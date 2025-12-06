"""
Trajectory Logging Infrastructure for IRED Optimization

This module provides tools for capturing and storing optimization trajectories
during the IRED diffusion process. Designed to integrate with GaussianDiffusion1D.opt_step()
for collecting trajectory data that will enable manifold learning analysis.

Key features:
- Memory-efficient logging with NumPy .npz compressed format
- Proper tensor-to-numpy conversion with deep copies
- Support for Sudoku state format: (729,) → (9,9,9) one-hot encoding
- Logs state, energy, problem_id, step, difficulty, success, landscape_idx
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
import copy
from pathlib import Path
import warnings


class TrajectoryLogger:
    """
    Main class for logging optimization trajectories during IRED diffusion.
    
    This logger captures state transitions, energy values, and metadata during
    the optimization process. Data is stored in memory-efficient format and
    can be saved to compressed NumPy archives.
    
    Usage:
        logger = TrajectoryLogger()
        trajectory_id = logger.init_trajectory_log(problem_id="sudoku_001", difficulty="hard")
        
        # During optimization loop:
        logger.append_step(trajectory_id, state, energy, step_num, success, landscape_idx)
        
        # Save when complete:
        logger.save_trajectory_log(trajectory_id, "trajectory_001.npz")
    """
    
    def __init__(self, max_trajectory_steps: int = 10000, max_memory_mb: float = 1000.0):
        """
        Initialize the trajectory logger.
        
        Args:
            max_trajectory_steps: Maximum number of steps per trajectory before warning
            max_memory_mb: Maximum memory usage in MB before warning
        """
        self.trajectories: Dict[str, Dict[str, List]] = {}
        self.trajectory_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_trajectory_steps = max_trajectory_steps
        self.max_memory_mb = max_memory_mb
        
    def init_trajectory_log(self, 
                          problem_id: str,
                          difficulty: Optional[str] = None,
                          landscape_idx: Optional[int] = None,
                          trajectory_id: Optional[str] = None) -> str:
        """
        Initialize a new trajectory log.
        
        Args:
            problem_id: Identifier for the problem being solved (e.g., "sudoku_001")
            difficulty: Problem difficulty level (e.g., "easy", "medium", "hard")
            landscape_idx: Energy landscape index for multi-landscape problems
            trajectory_id: Optional custom trajectory ID. If None, auto-generated.
            
        Returns:
            trajectory_id: Unique identifier for this trajectory
            
        Raises:
            ValueError: If trajectory_id already exists
        """
        if trajectory_id is None:
            trajectory_id = f"traj_{len(self.trajectories):06d}"
            
        if trajectory_id in self.trajectories:
            raise ValueError(f"Trajectory ID '{trajectory_id}' already exists")
            
        # Initialize trajectory data storage
        self.trajectories[trajectory_id] = {
            'states': [],
            'energies': [],
            'steps': [],
            'success_flags': [],
            'landscape_indices': [],
            'step_rejected_flags': [],
            'time_steps': []
        }
        
        # Store metadata
        self.trajectory_metadata[trajectory_id] = {
            'problem_id': problem_id,
            'difficulty': difficulty,
            'landscape_idx': landscape_idx,
            'start_time': np.datetime64('now'),
            'num_steps': 0,
            'completed': False
        }
        
        return trajectory_id
    
    def append_step(self,
                   trajectory_id: str,
                   state: Union[torch.Tensor, np.ndarray],
                   energy: Union[torch.Tensor, np.ndarray, float],
                   step: int,
                   success: bool = False,
                   landscape_idx: Optional[int] = None,
                   step_rejected: bool = False,
                   time_step: Optional[Union[torch.Tensor, np.ndarray, float]] = None) -> None:
        """
        Append a step to the trajectory log.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            state: Current state tensor/array. For Sudoku: (batch, 729) or (batch, 9, 9, 9)
            energy: Energy value at this step
            step: Step number in the optimization process
            success: Whether this step represents a successful solution
            landscape_idx: Energy landscape index for this step
            step_rejected: Whether this optimization step was rejected due to energy increase
            time_step: Diffusion time step (t) parameter, crucial for understanding optimization context
            
        Raises:
            ValueError: If trajectory_id doesn't exist
            RuntimeError: If tensor conversion fails
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory ID '{trajectory_id}' does not exist")
            
        traj = self.trajectories[trajectory_id]
        metadata = self.trajectory_metadata[trajectory_id]
        
        try:
            # Convert state to numpy with deep copy
            state_np = self._safe_tensor_to_numpy(state)
            
            # Handle Sudoku state format conversion: (729,) → (9,9,9)
            if state_np.ndim >= 2 and state_np.shape[-1] == 729:
                # Reshape from flat (729,) to (9,9,9) one-hot format
                batch_size = state_np.shape[0] if state_np.ndim > 1 else 1
                if state_np.ndim == 1:
                    state_np = state_np.reshape(1, 729)
                    batch_size = 1
                    
                # Convert to (batch, 9, 9, 9) format
                state_np = state_np.reshape(batch_size, 9, 9, 9)
            
            # Convert energy to numpy
            if isinstance(energy, (torch.Tensor, np.ndarray)):
                energy_np = self._safe_tensor_to_numpy(energy)
                if energy_np.ndim > 1:
                    energy_np = energy_np.flatten()
            else:
                energy_np = np.array([float(energy)])
            
            # Convert time_step to numpy if provided
            if time_step is not None:
                if isinstance(time_step, (torch.Tensor, np.ndarray)):
                    time_step_np = self._safe_tensor_to_numpy(time_step)
                    if time_step_np.ndim > 1:
                        time_step_np = time_step_np.flatten()
                else:
                    time_step_np = np.array([float(time_step)])
            else:
                time_step_np = np.array([0.0])  # Default value for missing time step
                
            # Store data with deep copies to avoid reference issues
            traj['states'].append(copy.deepcopy(state_np))
            traj['energies'].append(copy.deepcopy(energy_np))
            traj['steps'].append(int(step))
            traj['success_flags'].append(bool(success))
            traj['step_rejected_flags'].append(bool(step_rejected))
            traj['time_steps'].append(copy.deepcopy(time_step_np))
            
            # Use provided landscape_idx or fall back to metadata
            if landscape_idx is not None:
                traj['landscape_indices'].append(int(landscape_idx))
            elif metadata['landscape_idx'] is not None:
                traj['landscape_indices'].append(int(metadata['landscape_idx']))
            else:
                traj['landscape_indices'].append(0)  # Default
                
            # Update metadata
            metadata['num_steps'] = len(traj['steps'])
            
            # Check memory and step limits
            self._check_trajectory_limits(trajectory_id)
            
        except Exception as e:
            raise RuntimeError(f"Failed to append step to trajectory '{trajectory_id}': {e}")
    
    def save_trajectory_log(self,
                          trajectory_id: str,
                          output_path: Union[str, Path],
                          compress: bool = True,
                          mark_completed: bool = True) -> None:
        """
        Save trajectory log to a compressed NumPy archive.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            output_path: Path where to save the .npz file
            compress: Whether to use compression (recommended)
            mark_completed: Whether to mark this trajectory as completed
            
        Raises:
            ValueError: If trajectory_id doesn't exist or no data to save
            IOError: If file writing fails
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory ID '{trajectory_id}' does not exist")
            
        traj = self.trajectories[trajectory_id]
        metadata = self.trajectory_metadata[trajectory_id]
        
        if len(traj['steps']) == 0:
            raise ValueError(f"No trajectory data to save for '{trajectory_id}'")
        
        try:
            # Prepare data for saving
            save_data = {}
            
            # Convert lists to numpy arrays
            save_data['states'] = np.array(traj['states'])
            save_data['energies'] = np.array(traj['energies'])
            save_data['steps'] = np.array(traj['steps'], dtype=np.int32)
            save_data['success_flags'] = np.array(traj['success_flags'], dtype=np.bool_)
            save_data['landscape_indices'] = np.array(traj['landscape_indices'], dtype=np.int32)
            save_data['step_rejected_flags'] = np.array(traj['step_rejected_flags'], dtype=np.bool_)
            save_data['time_steps'] = np.array(traj['time_steps'])
            
            # Add metadata
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (str, np.datetime64)):
                        save_data[f'meta_{key}'] = str(value)
                    else:
                        save_data[f'meta_{key}'] = value
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to compressed numpy archive
            if compress:
                np.savez_compressed(output_path, **save_data)
            else:
                np.savez(output_path, **save_data)
                
            # Mark as completed if requested
            if mark_completed:
                metadata['completed'] = True
                metadata['end_time'] = np.datetime64('now')
                
        except Exception as e:
            raise IOError(f"Failed to save trajectory '{trajectory_id}' to '{output_path}': {e}")
    
    def load_trajectory_log(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a trajectory log from a NumPy archive.
        
        Args:
            input_path: Path to the .npz file to load
            
        Returns:
            Dictionary containing trajectory data and metadata
            
        Raises:
            IOError: If file reading fails
        """
        try:
            data = np.load(input_path, allow_pickle=False)
            
            # Separate trajectory data from metadata
            trajectory_data = {}
            metadata = {}
            
            for key in data.keys():
                if key.startswith('meta_'):
                    metadata[key[5:]] = data[key].item() if data[key].ndim == 0 else data[key]
                else:
                    trajectory_data[key] = data[key]
            
            return {
                'trajectory': trajectory_data,
                'metadata': metadata
            }
            
        except Exception as e:
            raise IOError(f"Failed to load trajectory from '{input_path}': {e}")
    
    def get_trajectory_info(self, trajectory_id: str) -> Dict[str, Any]:
        """
        Get information about a trajectory.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            
        Returns:
            Dictionary with trajectory statistics and metadata
            
        Raises:
            ValueError: If trajectory_id doesn't exist
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory ID '{trajectory_id}' does not exist")
            
        traj = self.trajectories[trajectory_id]
        metadata = self.trajectory_metadata[trajectory_id]
        
        info = dict(metadata)  # Copy metadata
        info.update({
            'num_steps': len(traj['steps']),
            'has_data': len(traj['steps']) > 0,
            'final_success': traj['success_flags'][-1] if traj['success_flags'] else False,
            'memory_usage_mb': self._estimate_memory_usage(trajectory_id)
        })
        
        if len(traj['energies']) > 0:
            energies = np.concatenate(traj['energies']) if len(traj['energies'][0]) > 1 else np.array([e[0] for e in traj['energies']])
            info.update({
                'initial_energy': float(energies[0]),
                'final_energy': float(energies[-1]),
                'min_energy': float(np.min(energies)),
                'energy_reduction': float(energies[0] - energies[-1])
            })
        
        return info
    
    def clear_trajectory(self, trajectory_id: str) -> None:
        """
        Clear a trajectory from memory.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            
        Raises:
            ValueError: If trajectory_id doesn't exist
        """
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory ID '{trajectory_id}' does not exist")
            
        del self.trajectories[trajectory_id]
        del self.trajectory_metadata[trajectory_id]
    
    def clear_all_trajectories(self) -> None:
        """Clear all trajectories from memory."""
        self.trajectories.clear()
        self.trajectory_metadata.clear()
    
    def list_trajectory_ids(self) -> List[str]:
        """Get list of all trajectory IDs."""
        return list(self.trajectories.keys())
    
    def _safe_tensor_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Safely convert tensor to numpy array with deep copy.
        
        Args:
            tensor: Input tensor or array
            
        Returns:
            NumPy array copy
            
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            if isinstance(tensor, torch.Tensor):
                # Move to CPU and detach from computation graph
                if tensor.requires_grad:
                    tensor = tensor.detach()
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                return tensor.numpy().copy()
            elif isinstance(tensor, np.ndarray):
                return tensor.copy()
            else:
                # Try to convert to numpy
                return np.array(tensor, copy=True)
        except Exception as e:
            raise RuntimeError(f"Failed to convert tensor to numpy: {e}")
    
    def _estimate_memory_usage(self, trajectory_id: str) -> float:
        """
        Estimate memory usage of a trajectory in MB.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            
        Returns:
            Estimated memory usage in MB
        """
        if trajectory_id not in self.trajectories:
            return 0.0
            
        traj = self.trajectories[trajectory_id]
        total_bytes = 0
        
        for states in traj['states']:
            total_bytes += states.nbytes
        for energies in traj['energies']:
            total_bytes += energies.nbytes if hasattr(energies, 'nbytes') else 8  # float64
        for time_steps in traj['time_steps']:
            total_bytes += time_steps.nbytes if hasattr(time_steps, 'nbytes') else 8  # float64
        
        # Add overhead for lists and metadata
        total_bytes += len(traj['steps']) * 18  # int32 + bool + bool + int32 + time_step overhead
        total_bytes += 1024  # metadata overhead
        
        return total_bytes / (1024 * 1024)
    
    def _check_trajectory_limits(self, trajectory_id: str) -> None:
        """
        Check trajectory size and memory limits, issue warnings if exceeded.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
        """
        if trajectory_id not in self.trajectories:
            return
            
        traj = self.trajectories[trajectory_id]
        num_steps = len(traj['steps'])
        memory_usage = self._estimate_memory_usage(trajectory_id)
        
        # Check step limit
        if num_steps > self.max_trajectory_steps:
            warnings.warn(
                f"Trajectory '{trajectory_id}' has {num_steps} steps, exceeding limit of {self.max_trajectory_steps}. "
                f"Consider saving and clearing trajectory to prevent memory issues.",
                UserWarning
            )
        
        # Check memory limit
        if memory_usage > self.max_memory_mb:
            warnings.warn(
                f"Trajectory '{trajectory_id}' using {memory_usage:.1f}MB, exceeding limit of {self.max_memory_mb}MB. "
                f"Consider saving and clearing trajectory to prevent memory exhaustion.",
                UserWarning
            )
        
        # Check total memory across all trajectories
        total_memory = sum(self._estimate_memory_usage(tid) for tid in self.trajectories.keys())
        if total_memory > self.max_memory_mb * 2:  # Allow 2x limit for multiple trajectories
            warnings.warn(
                f"Total memory usage across all trajectories: {total_memory:.1f}MB exceeds 2x limit. "
                f"Consider saving and clearing completed trajectories.",
                UserWarning
            )


# Convenience functions for simple usage

def init_trajectory_log(problem_id: str, 
                       difficulty: Optional[str] = None, 
                       landscape_idx: Optional[int] = None) -> str:
    """
    Initialize a new trajectory log using the global logger.
    
    Args:
        problem_id: Identifier for the problem being solved
        difficulty: Problem difficulty level
        landscape_idx: Energy landscape index
        
    Returns:
        trajectory_id: Unique identifier for this trajectory
    """
    global _global_logger
    if '_global_logger' not in globals():
        _global_logger = TrajectoryLogger()
    
    return _global_logger.init_trajectory_log(
        problem_id=problem_id, 
        difficulty=difficulty, 
        landscape_idx=landscape_idx
    )


def append_step(trajectory_id: str,
               state: Union[torch.Tensor, np.ndarray],
               energy: Union[torch.Tensor, np.ndarray, float],
               step: int,
               success: bool = False,
               landscape_idx: Optional[int] = None,
               step_rejected: bool = False,
               time_step: Optional[Union[torch.Tensor, np.ndarray, float]] = None) -> None:
    """
    Append a step to the trajectory log using the global logger.
    
    Args:
        trajectory_id: Unique identifier for the trajectory
        state: Current state tensor/array
        energy: Energy value at this step
        step: Step number in the optimization process
        success: Whether this step represents a successful solution
        landscape_idx: Energy landscape index
        step_rejected: Whether this optimization step was rejected due to energy increase
        time_step: Diffusion time step (t) parameter
    """
    global _global_logger
    if '_global_logger' not in globals():
        raise RuntimeError("No trajectory initialized. Call init_trajectory_log() first.")
    
    _global_logger.append_step(trajectory_id, state, energy, step, success, landscape_idx, step_rejected, time_step)


def save_trajectory_log(trajectory_id: str, output_path: Union[str, Path]) -> None:
    """
    Save trajectory log to a compressed NumPy archive using the global logger.
    
    Args:
        trajectory_id: Unique identifier for the trajectory
        output_path: Path where to save the .npz file
    """
    global _global_logger
    if '_global_logger' not in globals():
        raise RuntimeError("No trajectory initialized. Call init_trajectory_log() first.")
    
    _global_logger.save_trajectory_log(trajectory_id, output_path)


def get_global_logger() -> TrajectoryLogger:
    """Get the global trajectory logger instance."""
    global _global_logger
    if '_global_logger' not in globals():
        _global_logger = TrajectoryLogger()
    return _global_logger


# Example integration with GaussianDiffusion1D.opt_step()
def create_logging_hook(logger: TrajectoryLogger, trajectory_id: str):
    """
    Create a logging hook function for integration with opt_step().
    
    This function returns a callable that can be inserted into the opt_step()
    optimization loop to capture trajectory data.
    
    Args:
        logger: TrajectoryLogger instance
        trajectory_id: ID of the trajectory to log to
        
    Returns:
        Callable hook function
        
    Usage:
        logger = TrajectoryLogger()
        traj_id = logger.init_trajectory_log("sudoku_001", difficulty="hard")
        hook = create_logging_hook(logger, traj_id)
        
        # In opt_step() loop - call at multiple points to capture attempted and accepted steps:
        # Before step rejection: hook(inp, img_new, energy_new, step_i, success_flag, landscape_idx, False, t)
        # After step rejection: hook(inp, img, energy, step_i, success_flag, landscape_idx, step_rejected, t)
    """
    def logging_hook(inp: torch.Tensor,
                    img: torch.Tensor, 
                    energy: torch.Tensor,
                    step: int,
                    success: bool = False,
                    landscape_idx: Optional[int] = None,
                    step_rejected: bool = False,
                    time_step: Optional[torch.Tensor] = None):
        """Log a single optimization step."""
        # Log img tensor directly - this is the actual optimization variable being modified
        # inp tensor is not modified during optimization, so logging it would be misleading
        logger.append_step(trajectory_id, img, energy, step, success, landscape_idx, step_rejected, time_step)
    
    return logging_hook


if __name__ == "__main__":
    # Example usage
    import torch
    
    print("Testing trajectory logging...")
    
    # Initialize logger
    logger = TrajectoryLogger()
    
    # Start a new trajectory
    traj_id = logger.init_trajectory_log("test_sudoku_001", difficulty="hard", landscape_idx=0)
    print(f"Created trajectory: {traj_id}")
    
    # Simulate some optimization steps with new parameters
    batch_size = 2
    for step in range(5):
        # Create fake Sudoku state (batch_size, 729) - this represents img tensor only
        state = torch.randn(batch_size, 729) * 0.1
        energy = torch.tensor([10.0 - step * 2, 12.0 - step * 2.5]).unsqueeze(1)  # Decreasing energy
        time_step_val = torch.tensor([0.5 - step * 0.1])  # Simulated diffusion time step
        
        success = step == 4  # Success on final step
        step_rejected = step == 2  # Simulate rejection of step 2
        
        logger.append_step(traj_id, state, energy, step, success, 
                         landscape_idx=0, step_rejected=step_rejected, time_step=time_step_val)
    
    # Get trajectory info
    info = logger.get_trajectory_info(traj_id)
    print(f"Trajectory info: {info}")
    
    # Save trajectory
    output_path = "/tmp/test_trajectory.npz"
    logger.save_trajectory_log(traj_id, output_path)
    print(f"Saved trajectory to: {output_path}")
    
    # Load it back
    loaded = logger.load_trajectory_log(output_path)
    print(f"Loaded trajectory shape: {loaded['trajectory']['states'].shape}")
    
    print("Test completed successfully!")