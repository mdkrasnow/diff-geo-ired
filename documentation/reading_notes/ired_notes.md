# IRED (Iterative Reasoning Energy Diffusion) Research Notes

## IRED Goal Summary

IRED (Iterative Reasoning Energy Diffusion) is a novel approach that learns to perform iterative reasoning tasks by modeling them as energy-based optimization problems. The core idea is to train diffusion models that can navigate complex energy landscapes to solve reasoning problems through iterative refinement. Instead of directly predicting solutions, IRED learns energy functions that guide optimization trajectories toward correct answers through gradient-based updates.

The method combines the representational power of deep neural networks with the iterative reasoning capabilities of energy-based models. During training, IRED supervises both the final solution and the intermediate optimization trajectory (energy landscape supervision), enabling the model to learn not just what the correct answer is, but how to systematically search for it through iterative refinement. This approach is particularly powerful for problems where the solution space has complex geometric structure and multiple valid reasoning paths.

## Application Domains

Based on the codebase analysis, IRED has been applied to several diverse domains:

### Discrete-Space Reasoning Tasks:
- **Sudoku**: 9x9 constraint satisfaction puzzles with logical reasoning requirements
- **Graph Connectivity**: Determining reachability between nodes in graph structures  
- **SAT Problems**: Boolean satisfiability constraint solving

### Continuous-Space Reasoning Tasks:
- **Matrix Addition**: Learning arithmetic operations in continuous vector spaces
- **Low-rank Matrix Decomposition**: Factorizing matrices into lower-dimensional representations
- **Matrix Inverse**: Computing matrix inverses through iterative optimization

### Planning Tasks:
- **Shortest Path**: Finding optimal routes in graph structures
- **List Sorting**: Organizing sequences through swap operations

## Optimization Trajectory for Sudoku (Chosen Domain)

For Sudoku, the optimization trajectory has the following structure:

### State Representation:
- **State x**: 9×9×9 tensor representing cell assignments as one-hot probability distributions
- **Dimensionality**: 729-dimensional flattened vector per state
- **State Space**: Continuous relaxation of discrete Sudoku assignments

### Energy Function:
- **Energy E(x)**: Scalar value measuring constraint violations and solution quality
- **Implementation**: SudokuEBM - Complex convolutional neural network that processes concatenated input/output 9×9×9 tensors
- **Architecture Details**: 
  - **Input Processing**: Concatenates input and output tensors (18 channels: 9+9) 
  - **Initial Convolution**: `conv1` with 3×3 kernel mapping 18→384 channels
  - **ResBlock Structure**: 3 pairs of ResBlocks (res1a/res1b, res2a/res2b, res3a/res3b) with 384 filters each
  - **Attention Layers**: 3 Attention modules (attn1, attn2, attn3) with 128-dim heads (currently commented out)
  - **Time Embedding**: 64-dim time embedding integrated via ResBlock modulation
  - **Final Convolution**: `conv5` with 1×1 kernel mapping 384→9 channels
- **Energy Computation**: `output = conv5(h); energy = output.pow(2).sum(dim=[1, 2, 3])` (L2 norm of final CNN output)

### Optimization Dynamics:
- **Initialization**: Random state `x_0 ~ U(0,1)^729` 
- **Complete opt_step() Process** (lines 373-404 in diffusion_lib):
  1. **Gradient Computation**: `energy, grad = model(inp, img, t, return_both=True)`
  2. **Gradient Step**: `img_new = img - step_size * grad * scale_factor`
  3. **Constraint Masking**: `img_new = img_new * (1 - mask) + mask * data_cond` (preserves given cells)
  4. **Value Clamping**: `img_new = torch.clamp(img_new, -max_val, max_val)` (enforces valid ranges)
  5. **Step Rejection Logic**: 
     - `energy_new = model(inp, img_new, t, return_energy=True)`
     - `bad_step = (energy_new > energy)` (energy must decrease)
     - `img_new[bad_step] = img[bad_step]` (reject steps that increase energy)
  6. **State Update**: `img = img_new` (accept successful step)
- **Step Count**: Typically 5-50 optimization steps per problem (20 steps for Sudoku)
- **Convergence**: Energy must monotonically decrease; steps rejected if energy increases

### Trajectory Properties:
- **Smoothness**: Continuous state evolution through probability simplex
- **Constraint Satisfaction**: Energy guides toward valid row/column/block constraints
- **Solution Convergence**: Final state approaches one-hot distributions for solved puzzles

## Loggable Information for Trajectory Analysis

Based on the codebase structure and energy optimization process, the following information can be systematically logged:

### Core State Variables:
- **state**: 729-dimensional continuous state vector at each step
- **energy**: Scalar energy value E(x_t) at each optimization step
- **step**: Integer step number in optimization trajectory (0 to max_steps)

### Problem Metadata:
- **problem_id**: Unique identifier for each Sudoku instance
- **difficulty**: Problem complexity measure (easy/medium/hard classification)
- **success**: Boolean indicating whether problem was successfully solved
- **landscape_idx**: Index for different energy landscape configurations

### Optimization Dynamics:
- **gradient_norm**: Magnitude of energy gradient ||∇E(x)|| at each step
- **state_change**: Distance moved in state space ||x_{t+1} - x_t||
- **convergence_rate**: Rate of energy decrease per step

### Solution Quality Metrics:
- **constraint_violations**: Number of Sudoku constraint violations remaining
- **solution_accuracy**: Fraction of correctly placed digits
- **board_completion**: Percentage of cells with confident assignments (>0.9 probability)

### Trajectory Characteristics:
- **path_length**: Total distance traveled in state space
- **energy_trajectory**: Complete sequence of energy values [E(x_0), E(x_1), ..., E(x_T)]
- **convergence_step**: Step number where solution is first reached (if any)
- **final_state**: Terminal state after optimization completion

This rich trajectory data enables analysis of:
- Energy landscape geometry and smoothness
- Optimization efficiency and convergence patterns  
- Relationship between problem difficulty and trajectory complexity
- Success/failure modes and their geometric signatures
- Manifold structure of the solution space

The logged trajectories provide high-dimensional time series data suitable for manifold learning techniques like Diffusion Maps to uncover the intrinsic geometric structure of the IRED optimization process.

## Trajectory Logging Implementation Notes

### Hook Placement for Phase 2 Development:
Based on the detailed SudokuEBM architecture analysis, optimal trajectory logging hook placement:

1. **Energy Function Hooks**:
   - **Pre-conv1**: Raw concatenated input/output tensors (batch_size, 18, 9, 9)
   - **Post-conv1**: Initial feature maps (batch_size, 384, 9, 9)  
   - **Post-ResBlock pairs**: After res1b, res2b, res3b (batch_size, 384, 9, 9)
   - **Pre-conv5**: Final feature representation before energy computation
   - **Energy output**: Scalar energy values for step rejection analysis

2. **Optimization Process Hooks**:
   - **Gradient computation**: Capture ∇E(x) at each opt_step iteration
   - **Step proposals**: img_new before/after masking and clamping operations  
   - **Step acceptance**: Track which steps are rejected by energy increase criterion
   - **State evolution**: Complete sequence of accepted states throughout trajectory

3. **Critical Dynamics to Log**:
   - **Step rejection rate**: Fraction of gradient steps rejected per trajectory
   - **Energy monotonicity**: Verification that energy decreases monotonically
   - **Constraint satisfaction**: How masking preserves given Sudoku cells
   - **Convergence patterns**: Relationship between energy landscape and solution quality

This detailed architectural understanding enables precise placement of logging hooks to capture the complete optimization trajectory dynamics for manifold analysis in Phase 2.