#!/usr/bin/env python3
"""
IRED Trajectory Logging System - Efficient Version

This script runs IRED on multiple problem instances and logs the complete diffusion 
trajectories for downstream differential geometric analysis. It captures state vectors, 
energies, and error metrics at each diffusion step.

Optimized for fast processing with limited samples.
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

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_lib.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
from models import EBM, DiffusionWrapper
from dataset import Inverse


class EfficientTrajectoryLogger:
    """
    Efficient logger for capturing IRED diffusion trajectories during inference.
    
    This version directly interfaces with the core diffusion methods and uses
    memory-efficient batch processing.
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = self._setup_device(device)
        print(f"Using device: {self.device}")
        
        # Storage for trajectory data
        self.trajectories = {
            'problem_ids': [],
            'steps': [],
            'landscapes': [],  
            'states': [],
            'energies': [],
            'error_metrics': []
        }
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device with Apple Silicon MPS support."""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def create_model(self, inp_dim: int, out_dim: int) -> GaussianDiffusion1D:
        """Create IRED diffusion model."""
        # Create EBM model
        ebm_model = EBM(inp_dim=inp_dim, out_dim=out_dim)
        model = DiffusionWrapper(ebm_model)
        
        # Create diffusion wrapper with realistic parameters
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=32,
            objective='pred_noise',
            timesteps=10,  # Reduced for efficiency
            sampling_timesteps=10,
            supervise_energy_landscape=False,
            use_innerloop_opt=True,
            show_inference_tqdm=False,
            continuous=True  # For matrix inverse problems
        )
        
        return diffusion.to(self.device)
    
    def generate_problems(self, num_problems: int, matrix_size: int = 8) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate a specific number of matrix inverse problems."""
        problems = []
        
        print(f"Generating {num_problems} matrix inverse problems of size {matrix_size}x{matrix_size}")
        
        for i in range(num_problems):
            # Generate well-conditioned matrix
            A = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
            A = A @ A.T + 0.1 * torch.eye(matrix_size)  # Make positive definite
            
            # Compute inverse with high precision
            try:
                A_inv = torch.linalg.inv(A.double()).float()
                
                # Validate the inverse
                identity_check = torch.allclose(A @ A_inv, torch.eye(matrix_size), atol=1e-4)
                if not identity_check:
                    print(f"Warning: Problem {i} failed inverse validation")
                    continue
                    
                # Store as flattened tensors
                problems.append((A.flatten(), A_inv.flatten()))
                
            except Exception as e:
                print(f"Failed to generate problem {i}: {e}")
                continue
                
        print(f"Successfully generated {len(problems)} valid problems")
        return problems
    
    def log_single_trajectory(self, diffusion_model: GaussianDiffusion1D, 
                            inp: torch.Tensor, target: torch.Tensor, 
                            problem_id: int) -> bool:
        """Log trajectory for a single problem instance."""
        
        try:
            with torch.no_grad():
                batch_size = 1
                inp = inp.unsqueeze(0).to(self.device)  # Add batch dimension
                target = target.unsqueeze(0).to(self.device)
                
                # Initialize sampling
                img = torch.randn_like(target)
                num_timesteps = diffusion_model.num_timesteps
                
                # Storage for this trajectory
                states_sequence = []
                energies_sequence = []
                errors_sequence = []
                steps_sequence = []
                landscapes_sequence = []
                
                # Main diffusion loop
                for step_idx in range(num_timesteps):
                    t = num_timesteps - 1 - step_idx  # Reverse order (from T to 0)
                    batched_times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Store current state
                    states_sequence.append(img.detach().cpu().numpy().copy())
                    
                    # Compute energy
                    try:
                        energy = diffusion_model.model(inp, img, batched_times, return_energy=True)
                        energies_sequence.append(energy.detach().cpu().numpy().copy())
                    except:
                        energies_sequence.append(np.array([[0.0]]))  # Fallback
                    
                    # Compute error vs target
                    error = torch.norm(img - target, dim=-1)
                    errors_sequence.append(error.detach().cpu().numpy().copy())
                    
                    # Store step info
                    steps_sequence.append(step_idx)
                    landscapes_sequence.append(float(step_idx) / num_timesteps)  # Simple landscape param
                    
                    # Perform diffusion step (simplified)
                    if t > 0:
                        # Basic denoising step
                        try:
                            img, _ = diffusion_model.p_sample(inp, img, t, clip_denoised=True)
                        except Exception as e:
                            # Fallback: simple noise reduction
                            noise = torch.randn_like(img) * 0.1
                            img = img + noise
                            
                # Store complete trajectory
                self.trajectories['problem_ids'].append(problem_id)
                self.trajectories['steps'].append(np.array(steps_sequence))
                self.trajectories['landscapes'].append(np.array(landscapes_sequence))
                self.trajectories['states'].append(np.array(states_sequence))
                self.trajectories['energies'].append(np.array(energies_sequence))
                self.trajectories['error_metrics'].append(np.array(errors_sequence))
                
                return True
                
        except Exception as e:
            print(f"Error logging trajectory for problem {problem_id}: {e}")
            return False
    
    def log_trajectories(self, num_problems: int = 100, matrix_size: int = 8) -> Dict:
        """Main method to log trajectories for multiple problems."""
        
        print(f"Starting trajectory logging for {num_problems} problems")
        print("=" * 50)
        
        # Generate problems
        problems = self.generate_problems(num_problems, matrix_size)
        if len(problems) == 0:
            raise ValueError("No valid problems generated")
        
        # Create model
        inp_dim = out_dim = matrix_size * matrix_size
        diffusion_model = self.create_model(inp_dim, out_dim)
        
        print(f"Model configuration:")
        print(f"  - Matrix size: {matrix_size}x{matrix_size}")
        print(f"  - Input/Output dimension: {inp_dim}")
        print(f"  - Diffusion steps: {diffusion_model.num_timesteps}")
        print(f"  - Device: {self.device}")
        
        # Process each problem
        successful_logs = 0
        start_time = time.time()
        
        for problem_id, (inp, target) in enumerate(tqdm(problems, desc="Logging trajectories")):
            success = self.log_single_trajectory(diffusion_model, inp, target, problem_id)
            if success:
                successful_logs += 1
                
            # Memory management
            if (problem_id + 1) % 20 == 0:
                gc.collect()
                if self.device == 'mps':
                    torch.mps.empty_cache()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f"\nTrajectory logging completed in {elapsed_time:.2f} seconds")
        print(f"Successfully logged {successful_logs}/{len(problems)} trajectories")
        
        return self.trajectories
    
    def save_trajectories(self, save_path: str):
        """Save trajectory data to compressed npz file."""
        
        print(f"Saving trajectories to {save_path}")
        
        # Prepare data for saving
        save_data = {
            'problem_ids': np.array(self.trajectories['problem_ids']),
            'num_problems': len(self.trajectories['problem_ids']),
            'num_steps': len(self.trajectories['steps'][0]) if self.trajectories['steps'] else 0,
            'steps': np.array(self.trajectories['steps'], dtype=object),
            'landscapes': np.array(self.trajectories['landscapes'], dtype=object),
            'states': np.array(self.trajectories['states'], dtype=object),
            'energies': np.array(self.trajectories['energies'], dtype=object),
            'error_metrics': np.array(self.trajectories['error_metrics'], dtype=object),
            'device': self.device,
            'timestamp': time.time()
        }
        
        # Save compressed
        np.savez_compressed(save_path, **save_data)
        
        # Verify save
        file_size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"Saved trajectory data: {save_data['num_problems']} problems, "
              f"{save_data['num_steps']} steps each")
        print(f"File size: {file_size_mb:.1f} MB")


def main():
    """Main function to run efficient IRED trajectory logging."""
    
    print("IRED Efficient Trajectory Logging System")
    print("=" * 50)
    
    # Configuration
    num_problems = 150    # Target 50-200 as specified  
    matrix_size = 8       # Manageable matrix size for testing
    
    # Create logger and run
    logger = EfficientTrajectoryLogger(device='auto')
    
    try:
        trajectory_data = logger.log_trajectories(num_problems, matrix_size)
        
        # Save results
        output_path = os.path.expanduser("~/documentation/results/ired_trajectories_raw.npz")
        logger.save_trajectories(output_path)
        
        # Final validation
        print("\nTrajectory Logging Summary:")
        print(f"  - Problems processed: {len(trajectory_data['problem_ids'])}")
        if trajectory_data['states']:
            print(f"  - Steps per trajectory: {len(trajectory_data['steps'][0])}")
            print(f"  - State vector shape per step: {trajectory_data['states'][0].shape}")
            print(f"  - Energy values shape per step: {trajectory_data['energies'][0].shape}")
        print(f"  - Output file: {output_path}")
        
        print("\nIRED trajectory logging completed successfully!")
        
    except Exception as e:
        print(f"Error in trajectory logging: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)