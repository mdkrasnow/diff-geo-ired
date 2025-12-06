#!/usr/bin/env python3
"""
IRED Trajectory Logging System

This script runs IRED on multiple problem instances and logs the complete diffusion 
trajectories for downstream differential geometric analysis. It captures state vectors, 
energies, and error metrics at each diffusion step.

Compatible with Apple Silicon MPS backend and float32 dtype policy.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import gc
import psutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Trainer1D
from models import EBM, DiffusionWrapper
from dataset import Inverse


class TrajectoryLogger:
    """
    Logger for capturing IRED diffusion trajectories during inference.
    
    Hooks into the GaussianDiffusion1D.p_sample_loop method to record:
    - State vectors at each timestep
    - Energy values
    - Error metrics vs ground truth
    - Landscape parameters (k values)
    """
    
    def __init__(self, diffusion_model: GaussianDiffusion1D, device: str = 'auto'):
        """
        Initialize trajectory logger.
        
        Args:
            diffusion_model: IRED diffusion model to log trajectories from
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
        """
        self.diffusion_model = diffusion_model
        
        # Auto-detect device with Apple Silicon MPS support
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.diffusion_model = self.diffusion_model.to(self.device)
        
        # Storage for trajectory data
        self.reset_trajectory_storage()
        
    def reset_trajectory_storage(self):
        """Reset storage containers for new trajectory collection."""
        self.trajectories = {
            'problem_ids': [],
            'steps': [],
            'landscapes': [],
            'states': [],
            'energies': [],
            'error_metrics': []
        }
        
    def log_p_sample_loop_trajectory(self, batch_size: int, shape: Tuple, inp: torch.Tensor, 
                                   cond: torch.Tensor, mask: Optional[torch.Tensor], 
                                   problem_id: int, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Modified p_sample_loop that captures trajectory data during sampling.
        
        This method replicates the core sampling logic while adding trajectory logging.
        """
        device = self.diffusion_model.betas.device
        
        # Initialize with random noise or model-specific initialization
        if hasattr(self.diffusion_model.model, 'randn'):
            img = self.diffusion_model.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)
            
        x_start = None
        
        # Prepare landscape parameters (k values) - simplified version
        # For matrix inverse problems, we use different conditioning approaches
        num_timesteps = self.diffusion_model.num_timesteps
        landscapes = torch.linspace(0.1, 2.0, num_timesteps).to(device)
        
        # Trajectory storage for this problem
        problem_states = []
        problem_energies = []
        problem_errors = []
        problem_steps = []
        problem_landscapes = []
        
        # Main diffusion loop with trajectory logging
        iterator = reversed(range(0, num_timesteps))
        if self.diffusion_model.show_inference_tqdm:
            iterator = tqdm(iterator, desc='Trajectory logging', total=num_timesteps)
            
        for step_idx, t in enumerate(iterator):
            self_cond = x_start if self.diffusion_model.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
            
            # Handle conditional masking if provided
            cond_val = None
            if mask is not None:
                cond_val = self.diffusion_model.q_sample(x_start=inp, t=batched_times, 
                                                       noise=torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask
            
            # Perform diffusion step
            img_prev = img.clone()
            img, x_start = self.diffusion_model.p_sample(inp, img, t, self_cond, 
                                                       scale=False, with_noise=False)
            
            # Apply conditional masking again if needed
            if mask is not None:
                img = img * (1 - mask) + cond_val * mask
                
            # Inner loop optimization if enabled
            if self.diffusion_model.use_innerloop_opt:
                step_size = 20 if hasattr(self.diffusion_model, 'sudoku') and self.diffusion_model.sudoku else 5
                if t < 1:
                    img = self.diffusion_model.opt_step(inp, img, batched_times, mask, cond_val, 
                                                      step=step_size, sf=1.0)
                else:
                    img = self.diffusion_model.opt_step(inp, img, batched_times, mask, cond_val,
                                                      step=step_size, sf=1.0)
                img = img.detach()
            
            # Apply clipping based on problem type
            if self.diffusion_model.continuous:
                sf = 2.0
            elif hasattr(self.diffusion_model, 'shortest_path') and self.diffusion_model.shortest_path:
                sf = 0.1
            else:
                sf = 1.0
                
            max_val = self.diffusion_model.sqrt_alphas_cumprod[batched_times[0]].item() * sf
            img = torch.clamp(img, -max_val, max_val)
            
            # Log trajectory data for this step
            try:
                # State vector (current img)
                current_state = img.detach().cpu().numpy()
                
                # Energy computation
                with torch.no_grad():
                    energy = self.diffusion_model.model(inp, img, batched_times, return_energy=True)
                    energy_val = energy.detach().cpu().numpy()
                
                # Error metric vs ground truth
                error_metric = torch.norm(img - ground_truth, dim=-1).detach().cpu().numpy()
                
                # Current landscape parameter
                landscape_k = landscapes[step_idx].cpu().item()
                
                # Store trajectory data
                problem_states.append(current_state)
                problem_energies.append(energy_val)
                problem_errors.append(error_metric)
                problem_steps.append(num_timesteps - 1 - t)  # Forward step index
                problem_landscapes.append(landscape_k)
                
            except Exception as e:
                print(f"Warning: Failed to log trajectory at step {t}: {e}")
                continue
            
            # Prepare for next iteration
            img_unscaled = self.diffusion_model.predict_start_from_noise(img, batched_times, 
                                                                       torch.zeros_like(img))
            batched_times_prev = batched_times - 1
            
            if t != 0:
                alpha_prev = self.diffusion_model.sqrt_alphas_cumprod[batched_times_prev[0]]
                img = alpha_prev * img_unscaled
                
        # Store trajectory for this problem
        self.trajectories['problem_ids'].append(problem_id)
        self.trajectories['steps'].append(np.array(problem_steps))
        self.trajectories['landscapes'].append(np.array(problem_landscapes))
        self.trajectories['states'].append(np.array(problem_states))
        self.trajectories['energies'].append(np.array(problem_energies))
        self.trajectories['error_metrics'].append(np.array(problem_errors))
        
        return img
        
    def log_trajectories_batch(self, dataset, num_problems: int = 100, batch_size: int = 8) -> Dict:
        """
        Run IRED inference on multiple problems and collect trajectories.
        
        Args:
            dataset: Dataset to sample problems from
            num_problems: Number of problem instances to process
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing all trajectory data
        """
        print(f"Logging trajectories for {num_problems} problems with batch_size={batch_size}")
        
        self.reset_trajectory_storage()
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        problems_processed = 0
        total_memory_mb = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
                if problems_processed >= num_problems:
                    break
                    
                # Unpack batch data
                if len(batch_data) == 2:
                    inp_batch, target_batch = batch_data
                    mask_batch = None
                elif len(batch_data) == 3:
                    inp_batch, target_batch, mask_batch = batch_data
                else:
                    raise ValueError(f"Unexpected batch data format: {len(batch_data)} elements")
                
                # Move to device with proper dtype handling
                inp_batch = inp_batch.float().to(self.device)
                target_batch = target_batch.float().to(self.device) 
                if mask_batch is not None:
                    mask_batch = mask_batch.float().to(self.device)
                
                current_batch_size = inp_batch.shape[0]
                
                # Process each problem in the batch individually to capture trajectories
                for i in range(current_batch_size):
                    if problems_processed >= num_problems:
                        break
                        
                    # Extract single problem
                    inp_single = inp_batch[i:i+1]
                    target_single = target_batch[i:i+1] 
                    mask_single = mask_batch[i:i+1] if mask_batch is not None else None
                    
                    # Log trajectory for this problem
                    try:
                        result = self.log_p_sample_loop_trajectory(
                            batch_size=1,
                            shape=self.diffusion_model.out_shape,
                            inp=inp_single,
                            cond=target_single,
                            mask=mask_single,
                            problem_id=problems_processed,
                            ground_truth=target_single
                        )
                        
                        problems_processed += 1
                        
                        # Memory monitoring
                        if problems_processed % 10 == 0:
                            process = psutil.Process()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            total_memory_mb = max(total_memory_mb, memory_mb)
                            print(f"Processed {problems_processed}/{num_problems} problems, "
                                  f"Memory: {memory_mb:.1f} MB")
                            
                            # Force garbage collection to manage memory
                            gc.collect()
                            if self.device == 'mps':
                                torch.mps.empty_cache()
                            elif self.device == 'cuda':
                                torch.cuda.empty_cache()
                                
                    except Exception as e:
                        print(f"Error processing problem {problems_processed}: {e}")
                        continue
                        
        print(f"Successfully logged trajectories for {problems_processed} problems")
        print(f"Peak memory usage: {total_memory_mb:.1f} MB")
        
        return self.trajectories
        
    def save_trajectories(self, save_path: str, compress: bool = True):
        """
        Save trajectory data to npz file.
        
        Args:
            save_path: Path to save the trajectory data
            compress: Whether to use compression
        """
        print(f"Saving trajectories to {save_path}")
        
        # Prepare data for saving
        save_data = {}
        
        # Convert lists to numpy arrays for consistent storage
        save_data['problem_ids'] = np.array(self.trajectories['problem_ids'])
        save_data['num_problems'] = len(self.trajectories['problem_ids'])
        save_data['num_steps'] = len(self.trajectories['steps'][0]) if self.trajectories['steps'] else 0
        
        # Store trajectory sequences (these will be arrays of arrays)
        save_data['steps'] = np.array(self.trajectories['steps'], dtype=object)
        save_data['landscapes'] = np.array(self.trajectories['landscapes'], dtype=object)
        save_data['states'] = np.array(self.trajectories['states'], dtype=object)
        save_data['energies'] = np.array(self.trajectories['energies'], dtype=object)
        save_data['error_metrics'] = np.array(self.trajectories['error_metrics'], dtype=object)
        
        # Add metadata
        save_data['device'] = self.device
        save_data['timestamp'] = time.time()
        save_data['diffusion_steps'] = self.diffusion_model.num_timesteps
        
        # Save with appropriate compression
        if compress:
            np.savez_compressed(save_path, **save_data)
        else:
            np.savez(save_path, **save_data)
            
        print(f"Saved trajectory data: {len(self.trajectories['problem_ids'])} problems, "
              f"{save_data['num_steps']} steps each")


def load_trained_model(model_path: str, dataset) -> GaussianDiffusion1D:
    """
    Load a pre-trained IRED model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        dataset: Dataset instance for model configuration
        
    Returns:
        Loaded diffusion model
    """
    print(f"Loading model from {model_path}")
    
    # Create model architecture
    ebm_model = EBM(
        inp_dim=dataset.inp_dim,
        out_dim=dataset.out_dim,
    )
    model = DiffusionWrapper(ebm_model)
    
    # Create diffusion wrapper
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=32,
        objective='pred_noise',
        timesteps=100,
        sampling_timesteps=100,
        supervise_energy_landscape=False,
        use_innerloop_opt=True,
        show_inference_tqdm=False,
        continuous=True  # For matrix inverse problems
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        diffusion.load_state_dict(checkpoint['model'])
        print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    else:
        print(f"Warning: Model path {model_path} not found, using random initialization")
        
    return diffusion


def main():
    """
    Main function to run IRED trajectory logging.
    """
    print("Starting IRED Trajectory Logging System")
    print("=" * 50)
    
    # Configuration
    num_problems = 150  # Target 50-200 as specified
    batch_size = 8      # Memory-efficient batch size for Apple Silicon
    rank = 20           # Matrix rank for inverse problems
    
    # Create dataset - using matrix inverse as specified in assignment
    print(f"Creating matrix inverse dataset with rank {rank}")
    dataset = Inverse("train", rank, ood=False)
    dataset.inp_dim = rank * rank   # Flattened input matrix
    dataset.out_dim = rank * rank   # Flattened output inverse matrix
    
    # Try to load pre-trained model, fallback to random initialization
    model_path = "results/ds_inverse/model_mlp/model-1.pt"  # Common checkpoint location
    diffusion_model = load_trained_model(model_path, dataset)
    
    # Create trajectory logger
    logger = TrajectoryLogger(diffusion_model, device='auto')
    
    print(f"Model configuration:")
    print(f"  - Input dimension: {dataset.inp_dim}")
    print(f"  - Output dimension: {dataset.out_dim}")
    print(f"  - Diffusion steps: {diffusion_model.num_timesteps}")
    print(f"  - Device: {logger.device}")
    
    # Log trajectories
    start_time = time.time()
    trajectory_data = logger.log_trajectories_batch(
        dataset=dataset,
        num_problems=num_problems,
        batch_size=batch_size
    )
    
    elapsed_time = time.time() - start_time
    print(f"Trajectory logging completed in {elapsed_time:.2f} seconds")
    
    # Save results
    output_path = os.path.expanduser("~/documentation/results/ired_trajectories_raw.npz")
    logger.save_trajectories(output_path, compress=True)
    
    # Validation summary
    print("\nTrajectory Logging Summary:")
    print(f"  - Problems processed: {len(trajectory_data['problem_ids'])}")
    print(f"  - Steps per trajectory: {len(trajectory_data['steps'][0]) if trajectory_data['steps'] else 0}")
    print(f"  - Output file: {output_path}")
    print(f"  - File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Validate data completeness
    if trajectory_data['problem_ids']:
        states_shape = trajectory_data['states'][0].shape if trajectory_data['states'] else "N/A"
        energies_shape = trajectory_data['energies'][0].shape if trajectory_data['energies'] else "N/A"
        print(f"  - State vector shape per step: {states_shape}")
        print(f"  - Energy values shape per step: {energies_shape}")
    
    print("\nIRED trajectory logging completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrajectory logging interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError in trajectory logging: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)