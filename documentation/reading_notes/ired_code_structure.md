# IRED Code Structure Analysis

## Overview
This document provides a detailed analysis of the IRED (Iterative Reasoning Energy Diffusion) codebase, focusing on the optimization loop structure, state representations, energy computation, and key hook points for trajectory logging.

**Repository**: `https://github.com/yilundu/ired_code_release`
**Location**: `external/ired/`
**Primary Files**: `train.py`, `models.py`, `diffusion_lib/denoising_diffusion_pytorch_1d.py`

## 1. Optimization Loop Location and Structure

### Primary Optimization Function: `GaussianDiffusion1D.opt_step()`
**File**: `external/ired/diffusion_lib/denoising_diffusion_pytorch_1d.py`
**Lines**: 373-406

```python
def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
    with torch.enable_grad():
        for i in range(step):  # Fixed 5 iterations by default
            # Compute energy and gradient
            energy, grad = self.model(inp, img, t, return_both=True)
            
            # Gradient descent step
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            
            # Apply constraints (masking for fixed variables)
            if mask is not None:
                img_new = img_new * (1 - mask) + mask * data_cond
            
            # Clamp to valid range based on diffusion schedule
            max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
            img_new = torch.clamp(img_new, -max_val, max_val)
            
            # Energy-based step acceptance (reject if energy increases)
            energy_new = self.model(inp, img_new, t, return_energy=True)
            bad_step = (energy_new > energy)
            img_new[bad_step] = img[bad_step]  # Revert bad steps
            
            img = img_new.detach() if eval else img_new
    return img
```

### Key Characteristics:
- **Fixed iteration budget**: 5 steps per optimization call
- **Energy-guided updates**: Rejects steps that increase energy
- **Constraint handling**: Supports masked variables (e.g., Sudoku clues)
- **Adaptive step size**: Based on diffusion timestep and scale factor
- **Deterministic evaluation**: Uses `.detach()` in eval mode

## 2. State Variable Representation and Shapes

### Continuous Tasks (Addition, Inverse, LowRank)
- **Shape**: `(batch_size, feature_dim)`
  - Addition: `(batch_size, 100)` - digit sequences
  - Inverse: `(batch_size, 400)` - flattened 20x20 matrix
  - LowRank: `(batch_size, rank*2)` - matrix factorization
- **Data type**: `torch.float32`
- **Range**: Typically normalized to `[-1, 1]`

### Sudoku Domain
- **Raw shape**: `(batch_size, 729)` = `81 positions × 9 digits`
- **Reshaped**: `(batch_size, 9, 9, 9)` = `spatial_h × spatial_w × digit_channels`
- **Encoding**: One-hot vectors, centered to `[-1, +1]` range
- **Constraints**: Mask for given clues (fixed during optimization)

### Graph Tasks (Connectivity, Planning)
- **Connectivity**: `(batch_size, 144)` = `12×12` adjacency matrix
- **Planning**: Variable shapes based on grid size and features
- **Node features**: Mixed continuous/discrete representations
- **Spatial encoding**: Often includes coordinate information

### Time Embedding
- **Shape**: `(batch_size,)` - scalar timestep per sample
- **Range**: `[0, diffusion_steps-1]` (typically 1000 steps)
- **Usage**: Controls noise level and optimization dynamics

## 3. Energy Computation Functions

### Core Energy-Based Models

#### 1. Generic EBM (`models.py:164-216`)
```python
class EBM(nn.Module):
    def forward(self, x, t):  # x = concatenated input+output
        # MLP with time-modulated features
        h = time_modulated_mlp(x, t)
        energy = self.fc4(h).pow(2).sum(dim=-1)[..., None]  # L2 norm squared
        return energy
```
- **Input**: Concatenated problem + current solution
- **Output**: Scalar energy per sample
- **Architecture**: 4-layer MLP with time conditioning

#### 2. Sudoku EBM (`models.py:328-439`)
```python
class SudokuEBM(nn.Module):
    def forward(self, x, t):
        # x: (batch, 729) -> (batch, 9, 9, 9)
        # CNN processing + constraint evaluation
        energy = output.pow(2).sum(dim=1).sum(dim=1).sum(dim=1)[:, None]
        return energy
```
- **Input**: One-hot encoded Sudoku state
- **Output**: Constraint violation energy
- **Architecture**: CNN with spatial convolutions

#### 3. Graph EBMs (`models.py:507+`)
- **GraphEBM**: Basic graph reasoning
- **GNNConvEBM**: Graph convolution networks
- **GNNConv1DEBMV2**: Advanced 1D graph processing

### Diffusion Wrapper Classes
**Purpose**: Interface between EBMs and diffusion process
**Key methods**:
- `return_energy=True`: Returns energy scalar
- `return_both=True`: Returns (energy, gradient) tuple
- Automatic differentiation for gradients

## 4. Convergence and Success Logic

### Energy-Based Convergence
- **No explicit threshold**: Uses fixed iteration budget
- **Step rejection mechanism**: Automatically rejects energy-increasing steps
- **Natural convergence**: Energy landscape guides toward local minima

### Success Evaluation (Domain-Specific)

#### Sudoku
- **Constraint checking**: Row, column, and box uniqueness
- **Hard constraints**: Given clues must remain fixed
- **Success metric**: All constraints satisfied

#### Graph Connectivity
- **Reachability**: Can traverse between specified nodes
- **Structure preservation**: Maintain required graph properties
- **Success metric**: Connectivity requirements met

#### Planning Tasks
- **Goal achievement**: Reach target state/position
- **Path validity**: Respect movement constraints
- **Success metric**: Task completion within budget

### Training vs Evaluation Success
- **Training**: Continuous energy minimization for landscape learning
- **Evaluation**: Discrete success/failure based on task completion

## 5. Training vs Evaluation Modes

### Training Mode (`forward()` in `denoising_diffusion_pytorch_1d.py:705`)
```python
def forward(self, inp, target, mask):
    # Standard denoising diffusion loss
    loss_mse = denoising_objective(...)
    
    if self.supervise_energy_landscape:
        # Energy landscape supervision
        data_sample = self.q_sample(x_start=target, t=t, noise=noise)
        fake_sample = optimization_with_noise(...)
        
        energy_real, energy_fake = self.model(data_sample), self.model(fake_sample)
        loss_energy = contrastive_energy_loss(energy_real, energy_fake)
        
        total_loss = loss_mse + loss_scale * loss_energy
    
    return total_loss
```

**Key features**:
- **Dual supervision**: Denoising + energy landscape
- **Contrastive learning**: Real data vs noisy optimized samples
- **Gradient flow**: Full backpropagation through optimization steps

### Evaluation Mode (`p_sample_loop()` in `denoising_diffusion_pytorch_1d.py:408`)
```python
@torch.no_grad()
def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
    img = torch.randn((batch_size, *shape), device=device)
    
    trajectory = []
    for t in reversed(range(self.num_timesteps)):
        img = self.p_sample(img, t, inp, cond, mask)
        if return_traj:
            trajectory.append(img.clone())
    
    return img, trajectory if return_traj else img
```

**Key features**:
- **No gradients**: Uses `@torch.no_grad()` decorator
- **Full trajectory**: Can return complete diffusion path
- **Deterministic**: Fixed random seed for reproducible results

## 6. Key Functions to Hook for Trajectory Logging

### Primary Hook Point: `opt_step()` Method
**Location**: `diffusion_lib/denoising_diffusion_pytorch_1d.py:373-406`

**Instrumentation strategy**:
```python
def opt_step_with_logging(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0):
    # Initialize trajectory storage
    trajectory_states = [img.detach().cpu().numpy()]
    trajectory_energies = []
    trajectory_gradients = []
    
    with torch.enable_grad():
        for i in range(step):
            # Compute energy and gradient
            energy, grad = self.model(inp, img, t, return_both=True)
            
            # LOG: Current state, energy, gradient
            trajectory_energies.append(energy.detach().cpu().numpy())
            trajectory_gradients.append(grad.detach().cpu().numpy())
            
            # Optimization step
            img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf
            
            # Constraint and energy checking...
            
            # LOG: Updated state
            trajectory_states.append(img_new.detach().cpu().numpy())
            
            img = img_new.detach() if eval else img_new
    
    return img, {
        'states': np.array(trajectory_states),
        'energies': np.array(trajectory_energies),
        'gradients': np.array(trajectory_gradients),
        'timestep': t.cpu().numpy(),
        'success': evaluate_success(img_new)
    }
```

### Secondary Hook Points

#### 1. Full Diffusion Trajectory (`p_sample_loop()`)
- **Purpose**: Capture complete generation process
- **Data**: State evolution across all diffusion timesteps
- **Usage**: Understanding long-term optimization dynamics

#### 2. Model-Specific Energy Functions
- **SudokuEBM.forward()**: Domain-specific energy landscape
- **GraphEBM.forward()**: Graph reasoning energy
- **Purpose**: Track energy values without gradients

#### 3. Training Loop (`Trainer1D.train()`)
- **Purpose**: Capture training dynamics and loss evolution
- **Data**: Energy landscape supervision effectiveness

### Recommended Data Structure
```python
trajectory_dataset = {
    'states': np.ndarray,        # (n_problems, n_steps, state_dim)
    'energies': np.ndarray,      # (n_problems, n_steps)
    'gradients': np.ndarray,     # (n_problems, n_steps, state_dim)
    'timesteps': np.ndarray,     # (n_problems,) diffusion timestep
    'success_flags': np.ndarray, # (n_problems,) task completion
    'metadata': {
        'domain': str,           # 'sudoku', 'connectivity', etc.
        'difficulty': np.ndarray, # Problem-specific difficulty scores
        'problem_ids': np.ndarray, # Unique identifiers
        'convergence_steps': np.ndarray, # Steps until convergence
        'initial_energy': np.ndarray,    # Starting energy values
        'final_energy': np.ndarray       # Final energy values
    }
}
```

## 7. Implementation Notes for Trajectory Collection

### Minimal Code Changes Required
1. **Modify `opt_step()`**: Add optional logging parameter
2. **Update `p_sample_loop()`**: Add trajectory return option (already exists)
3. **Instrument evaluation script**: Save logged data to files

### Data Volume Considerations
- **Sudoku**: ~729 features per state, ~5-10 opt steps, ~1000 diffusion steps
- **Estimated size**: ~10MB per 100 problems with full trajectories
- **Recommended**: Log optimization steps only (not all diffusion steps)

### Recommended Workflow
1. **Choose domain**: Sudoku (well-constrained, interpretable)
2. **Instrument `opt_step()`**: Add trajectory logging
3. **Run evaluation**: Generate 1000+ problem trajectories
4. **Save data**: Store as `.npz` files in `data/ired/`
5. **Validate**: Check trajectory quality and success rates

## 8. Expected Data Types and Tensor Shapes

### Sudoku Domain (Recommended for Analysis)
- **State shape**: `(batch_size, 729)` → reshape to `(batch_size, 9, 9, 9)`
- **Energy shape**: `(batch_size, 1)` → scalar per problem
- **Gradient shape**: `(batch_size, 729)` → same as state
- **Timestep**: `(batch_size,)` → diffusion timestep (0-999)
- **Success**: `(batch_size,)` → boolean completion flags

### Trajectory Sequence
- **Optimization steps**: 5 per diffusion timestep
- **Diffusion timesteps**: ~50-100 for evaluation (subset of 1000 total)
- **Total sequence length**: ~250-500 states per problem
- **Recommended logging**: Optimization steps only (~25 states per problem)

This analysis provides the foundation for implementing trajectory logging in the IRED codebase to support manifold learning analysis of optimization dynamics.