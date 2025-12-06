# Trajectory Logging Infrastructure Design

## Overview

This document describes the design and implementation of the trajectory logging infrastructure for the IRED (Iterative Refinement Energy Diffusion) optimization system. The logging system captures optimization trajectories during the diffusion process to enable manifold learning analysis and trajectory visualization.

## Architecture

### Core Components

1. **TrajectoryLogger Class**: Main logging infrastructure
2. **Convenience Functions**: Simple interface for basic usage
3. **Integration Hooks**: For seamless integration with existing IRED code

### Key Design Principles

- **Memory Efficiency**: Uses NumPy compressed storage and smart memory management
- **Type Safety**: Handles tensor-to-numpy conversion with proper error handling
- **State Format Handling**: Special support for Sudoku (729,) → (9,9,9) one-hot conversion
- **Deep Copying**: Prevents reference issues and data corruption
- **Metadata Tracking**: Comprehensive logging of problem context

## Integration with IRED System

### Hook Point: GaussianDiffusion1D.opt_step()

The trajectory logging integrates with the main optimization loop in `diffusion_lib/denoising_diffusion_pytorch_1d.py` at lines 373-406:

```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
    with torch.enable_grad():
        for i in range(step):
            energy, grad = self.model(inp, img, t, return_both=True)
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            
            # ... optimization logic ...
            
            energy_new = self.model(inp, img_new, t, return_energy=True)
            # HOOK POINT: Log trajectory data here
            
            # ... step rejection logic ...
```

### State Format Requirements

#### Sudoku Problem Format
- **Input format**: `(batch_size, 729)` - flattened one-hot encoding
- **Storage format**: `(batch_size, 9, 9, 9)` - 3D one-hot tensor
- **Conversion**: Automatic reshape from 729-dimensional vector to 9×9×9 cube

The SudokuEBM model expects concatenated input and output:
```python
# In DiffusionWrapper.forward()
opt_variable = torch.cat([inp, opt_out], dim=-1)  # Shape: (batch, 1458)
# inp: (batch, 729) - problem state  
# opt_out: (batch, 729) - solution state
```

### Energy Computation

Energy is computed through the SudokuEBM model wrapped in a DiffusionWrapper:

```python
# models.py - DiffusionWrapper.forward()
energy = self.ebm(opt_variable, t)  # SudokuEBM returns (batch, 1)
opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]
```

The SudokuEBM architecture:
- Input: Concatenated inp+out reshaped to (batch, 18, 9, 9) 
- Architecture: conv1 → 3×ResBlock → conv5 with 384 filters
- Output: Energy values via `output.pow(2).sum(dim=[1, 2, 3])[:, None]`

## Data Schema

### Logged Fields

| Field | Type | Description |
|-------|------|-------------|
| `state` | `numpy.ndarray` | Full optimization state (inp + img concatenated) |
| `energy` | `numpy.ndarray` | Energy value(s) for current state |
| `problem_id` | `str` | Unique identifier for the problem instance |
| `step` | `int` | Step number within the optimization sequence |
| `difficulty` | `str` | Problem difficulty level ("easy", "medium", "hard") |
| `success` | `bool` | Whether this step represents a successful solution |
| `landscape_idx` | `int` | Energy landscape index for multi-landscape problems |

### File Format

Data is stored in NumPy's compressed `.npz` format:

```python
# Example saved data structure
{
    'states': ndarray,              # Shape: (num_steps, batch_size, 9, 9, 9)
    'energies': ndarray,            # Shape: (num_steps, batch_size)
    'steps': ndarray,               # Shape: (num_steps,) dtype=int32
    'success_flags': ndarray,       # Shape: (num_steps,) dtype=bool
    'landscape_indices': ndarray,   # Shape: (num_steps,) dtype=int32
    'meta_problem_id': str,
    'meta_difficulty': str,
    'meta_start_time': str,
    'meta_num_steps': int,
    'meta_completed': bool
}
```

## Usage Examples

### Basic Usage

```python
from src.trajectory_logging import TrajectoryLogger

# Initialize logger
logger = TrajectoryLogger()

# Start trajectory
traj_id = logger.init_trajectory_log(
    problem_id="sudoku_001", 
    difficulty="hard", 
    landscape_idx=0
)

# During optimization loop
for step in range(num_optimization_steps):
    # ... optimization code ...
    
    # Log the step
    full_state = torch.cat([inp, img], dim=-1)  # Combine input and output
    logger.append_step(
        trajectory_id=traj_id,
        state=full_state,
        energy=energy_tensor,
        step=step,
        success=(step == final_step and solved),
        landscape_idx=current_landscape
    )

# Save trajectory
logger.save_trajectory_log(traj_id, f"trajectories/{problem_id}_traj.npz")
```

### Integration Hook Pattern

```python
from src.trajectory_logging import create_logging_hook

# Create logging hook
logger = TrajectoryLogger()
traj_id = logger.init_trajectory_log("sudoku_001", difficulty="hard")
log_step = create_logging_hook(logger, traj_id)

# In opt_step() method - CORRECTED integration approach for scientific validity:
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
    with torch.enable_grad():
        for i in range(step):
            energy, grad = self.model(inp, img, t, return_both=True)
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            
            # Apply constraints
            if mask is not None:
                img_new = img_new * (1 - mask) + mask * data_cond
            
            # Clamp values
            max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
            img_new = torch.clamp(img_new, -max_val, max_val)
            
            energy_new = self.model(inp, img_new, t, return_energy=True)
            
            # LOG ATTEMPTED STEP (before rejection check)
            success = check_solution_validity(img_new)  # Custom validation  
            log_step(inp, img_new, energy_new, i, success, landscape_idx=0, step_rejected=False, time_step=t)
            
            # Check step rejection
            bad_step = (energy_new > energy)
            step_rejected = bad_step.any().item() if hasattr(bad_step, 'any') else bad_step
            
            # LOG FINAL STEP (after potential rejection)
            final_img = img_new.clone()
            final_img[bad_step] = img[bad_step]  # Revert rejected steps
            final_energy = energy_new.clone()
            final_energy[bad_step] = energy[bad_step]
            
            log_step(inp, final_img, final_energy, i, success, landscape_idx=0, step_rejected=step_rejected, time_step=t)
            
            # Update img for next iteration
            img = final_img.detach() if eval else final_img
```

### Convenience Functions

```python
from src.trajectory_logging import init_trajectory_log, append_step, save_trajectory_log

# Simplified interface using global logger
traj_id = init_trajectory_log("sudoku_001", difficulty="hard")

# In optimization loop - with new parameters for scientific validity
append_step(traj_id, img_state, energy, step, success=False, 
           landscape_idx=0, step_rejected=False, time_step=t)

# Save when done
save_trajectory_log(traj_id, "output.npz")
```

## Memory Management

### Memory Efficiency Features

1. **Deep Copying**: All tensors are converted to NumPy with `.copy()` to prevent reference issues
2. **Compression**: Uses `np.savez_compressed()` for ~50-70% size reduction
3. **Memory Monitoring**: `_estimate_memory_usage()` tracks trajectory size
4. **Cleanup**: `clear_trajectory()` and `clear_all_trajectories()` for memory management

### Memory Usage Estimates

For Sudoku problems:
- State: (9, 9, 9) float32 = ~3KB per state
- Energy: 1 float32 = 4 bytes per step  
- Metadata: ~100 bytes per step
- **Total**: ~3.1KB per step per trajectory

For 1000-step trajectory with batch_size=32:
- Uncompressed: ~100MB
- Compressed (.npz): ~30-50MB

## Error Handling

### Exception Types

1. **ValueError**: Invalid trajectory IDs, missing data
2. **RuntimeError**: Tensor conversion failures  
3. **IOError**: File I/O errors during save/load

### Validation

- Trajectory ID uniqueness checking
- Tensor format validation
- State shape verification for Sudoku format
- Memory usage monitoring

## Future Extensions

### Planned Features

1. **Streaming Mode**: For very long trajectories that don't fit in memory
2. **Parallel Logging**: Thread-safe logging for multi-GPU setups
3. **Compression Options**: Choice of compression algorithms (gzip, lz4, etc.)
4. **Trajectory Analysis**: Built-in trajectory visualization and analysis tools
5. **Format Converters**: Export to HDF5, Parquet for large-scale analysis

### Integration Points

1. **Training Loop**: Integrate with main training script for automatic logging
2. **Evaluation Pipeline**: Log trajectories during model evaluation
3. **Hyperparameter Sweeps**: Batch trajectory collection across parameter settings
4. **Manifold Learning**: Direct integration with dimensionality reduction algorithms

## Technical Notes

### Tensor Conversion Details

```python
def _safe_tensor_to_numpy(self, tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()  # Remove from computation graph
        if tensor.is_cuda:
            tensor = tensor.cpu()     # Move to CPU if on GPU
        return tensor.numpy().copy()  # Convert and deep copy
```

### State Format Handling

```python
# Sudoku format: (729,) → (9,9,9)
if state_np.ndim >= 2 and state_np.shape[-1] == 729:
    batch_size = state_np.shape[0]
    # Reshape to one-hot 3D format
    state_np = state_np.reshape(batch_size, 9, 9, 9)
```

### File Format Rationale

NumPy `.npz` format chosen because:
- **Native Python**: No external dependencies
- **Compressed**: Built-in gzip compression
- **Efficient**: Optimized for numerical data
- **Flexible**: Supports mixed data types (arrays, scalars, strings)
- **Compatible**: Works with PyTorch, TensorFlow, JAX

## Testing Strategy

### Unit Tests

1. **TrajectoryLogger Class**: All methods with various input types
2. **Tensor Conversion**: PyTorch tensors, NumPy arrays, edge cases
3. **File I/O**: Save/load roundtrip testing
4. **Memory Management**: Memory usage tracking, cleanup
5. **Error Handling**: All exception paths

### Integration Tests

1. **IRED Integration**: Test with actual GaussianDiffusion1D.opt_step()
2. **Large Trajectories**: Memory and performance testing
3. **Batch Processing**: Multiple trajectories simultaneously
4. **File Format**: Compatibility across NumPy versions

### Performance Benchmarks

Target performance metrics:
- **Logging Overhead**: <5% of optimization time
- **Memory Efficiency**: <10% overhead vs raw data
- **File I/O**: Save/load within 1 second for 1000-step trajectory
- **Scalability**: Handle 100+ concurrent trajectories

## Implementation Status

### Completed ✅
- Core TrajectoryLogger class
- Convenience functions  
- Tensor conversion with deep copying
- Sudoku state format handling
- NumPy .npz compressed storage
- Error handling and validation
- Memory usage estimation
- Documentation and examples

### Next Steps 🔄
- Integration testing with actual IRED optimization
- Performance benchmarking
- Unit test suite
- Example integration scripts
- Memory optimization for large trajectories