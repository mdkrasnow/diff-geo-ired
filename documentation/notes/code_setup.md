# IRED Differential Geometry Code Setup

## Environment Configuration

### Platform
- **OS**: macOS (Apple Silicon)
- **Python**: 3.11.8
- **PyTorch**: 2.7.1 with MPS backend support
- **Device**: Apple Silicon MPS backend for GPU acceleration

### Dependencies
- torch (with MPS support)
- numpy
- tqdm (progress bars)
- psutil (memory monitoring)
- einops (tensor manipulation)
- All existing IRED dependencies

## Project Structure

### Core IRED Components
- `diffusion_lib/denoising_diffusion_pytorch_1d.py`: Main diffusion implementation
- `models.py`: EBM and wrapper models
- `dataset.py`: Matrix inverse and other datasets
- `train.py`: Training pipeline

### Trajectory Logging System
- `log_trajectories_efficient.py`: Main trajectory logging implementation
- `~/documentation/results/ired_trajectories_raw.npz`: Logged trajectory data

## Trajectory Logging Implementation

### Architecture
The trajectory logging system hooks into the IRED inference pipeline to capture:
1. **State vectors**: High-dimensional state at each diffusion step
2. **Energy values**: Energy landscape values E_θ(x,y,k)
3. **Error metrics**: Distance to ground truth solutions
4. **Landscape parameters**: Diffusion time-dependent conditioning

### Integration Points
- **GaussianDiffusion1D.p_sample()**: Core diffusion step method
- **DiffusionWrapper.forward()**: Energy computation interface
- **Matrix Inverse Dataset**: Well-conditioned matrix problems for logging

### Memory Management
- **Batch Processing**: Individual problem processing to avoid memory issues
- **Device Management**: Automatic MPS/CUDA/CPU selection
- **Garbage Collection**: Periodic cleanup during trajectory logging
- **Compressed Storage**: NPZ format with compression for efficient storage

### Data Format

The trajectory data is saved in compressed NPZ format with the following structure:

```python
{
    'problem_ids': [0, 1, 2, ..., 149],           # Problem identifiers
    'num_problems': 150,                          # Total number of problems
    'num_steps': 10,                             # Steps per trajectory  
    'steps': [[0,1,2,...,9], ...],              # Step indices for each problem
    'landscapes': [[0.0,0.1,...,0.9], ...],     # Landscape parameters k
    'states': [array(10,1,64), ...],            # State vectors at each step
    'energies': [array(10,1,1), ...],           # Energy values at each step
    'error_metrics': [array(10,1), ...],        # Error vs ground truth
    'device': 'mps',                            # Device used for computation
    'timestamp': 1733511434.5                   # Creation timestamp
}
```

### Validation Results
- **Problems Processed**: 150 matrix inverse problems (8x8 matrices)
- **Trajectory Length**: 10 diffusion steps per problem
- **State Dimensions**: 64-dimensional flattened matrices
- **File Size**: 0.4 MB (compressed)
- **Processing Time**: ~52 seconds on Apple Silicon M3
- **Memory Usage**: Efficient with periodic cleanup

## Configuration Parameters

### Matrix Inverse Problems
- **Matrix Size**: 8x8 (64 dimensions when flattened)
- **Conditioning**: Regularization with λ = 0.1 for positive definiteness
- **Validation**: Numerical validation of A @ A^(-1) ≈ I

### Diffusion Parameters
- **Timesteps**: 10 (reduced for efficiency)
- **Objective**: 'pred_noise' (noise prediction)
- **Inner Loop Optimization**: Enabled for refinement
- **Continuous Mode**: Enabled for continuous matrix problems

### Device Policy
- **Computation Dtype**: float64 for matrix operations
- **Storage Dtype**: float32 for MPS compatibility
- **Memory Management**: Automatic cache clearing for Apple Silicon

## Usage Instructions

### Running Trajectory Logging
```bash
cd /Users/mkrasnow/Desktop/diff-geo-ired
python log_trajectories_efficient.py
```

### Loading Trajectory Data
```python
import numpy as np

# Load trajectory data
data = np.load('~/documentation/results/ired_trajectories_raw.npz', allow_pickle=True)

# Access components
problem_ids = data['problem_ids']        # [150,]
states = data['states']                  # [150,] each containing (10,1,64)
energies = data['energies']              # [150,] each containing (10,1,1) 
errors = data['error_metrics']           # [150,] each containing (10,1)
landscapes = data['landscapes']          # [150,] each containing (10,)

# Example: Get states for problem 0
problem_0_states = states[0]  # Shape: (10, 1, 64)
problem_0_energies = energies[0]  # Shape: (10, 1, 1)
```

## Next Steps

The logged trajectory data enables downstream differential geometric analysis:
1. **Manifold Learning**: Fit Riemannian manifolds to state trajectories
2. **Curvature Analysis**: Compute sectional and Ricci curvature
3. **Geodesic Studies**: Analyze optimal paths in the state space
4. **Landscape Geometry**: Understand energy landscape topology

The trajectory logging system provides a solid foundation for the differential geometric analysis pipeline described in the project objectives.