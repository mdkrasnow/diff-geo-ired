# IRED Trajectory Logging Dataset Design

## Overview

This document defines the specific data collection strategy for IRED (Iterative Reasoning Energy Diffusion) trajectories to enable differential geometric analysis. The goal is to capture how iterative reasoning evolves through state space during the diffusion process.

## Dataset Selection Strategy

### Option A: Matrix Inverse Problems (`--dataset inverse`)

**Problem Formulation:**
- Input: Flattened matrix `A ∈ ℝ^{n×n}` → `A_flat ∈ ℝ^{n²}`
- Output: Flattened matrix inverse `B = A^{-1}` → `B_flat ∈ ℝ^{n²}`
- State vector format: `B_t ∈ ℝ^{n²}` (flattened matrix at diffusion step t)

**Technical Details:**
- Default rank: 20 → state dimension = 400
- Input generation: Random symmetric positive definite matrix construction
- Error metric: MSE between predicted and true inverse
- Diffusion steps: 10 (default from train.py)

**State Vector Structure for Matrix Inverse:**
```
B_t = [b11_t, b12_t, ..., b1n_t, b21_t, ..., bnn_t] ∈ ℝ^{n²}
```
where `bij_t` represents the (i,j) entry of the predicted inverse matrix at step t.

### Option B: Shortest Path Planning (`--dataset shortest-path-1d`)

**Problem Formulation:**
- Input: Graph structure with start/end nodes
- Output: Sequence of node selections for optimal path
- State vector format: `s_t ∈ ℝ^N` (node scores at diffusion step t)

**Technical Details:**
- Default graph size: N nodes (varies by problem instance)
- Input: Concatenated start and end node one-hot encodings
- Output: Sequence of one-hot node selections (padded to 8 steps)
- Error metric: Binary cross-entropy on path correctness
- Diffusion steps: 10 (default from train.py)

**State Vector Structure for Planning:**
```
s_t = [score_1_t, score_2_t, ..., score_N_t] ∈ ℝ^N
```
where `score_i_t` represents the probability/energy of selecting node i at step t.

## Logging Strategy

### Required Data Fields

Based on assignment constraints, each logged trajectory must include:

1. **problem_id** (int): Unique identifier for each problem instance
2. **step** (int): Diffusion step index (0 to diffusion_steps)
3. **landscape** (string): Energy landscape identifier (varies by model)
4. **state** (array): State vector at current step (format depends on dataset choice)
5. **energy** (float): Energy value at current state
6. **error_metric** (float): Task-specific error measurement

### Dataset-Specific Logging Details

**For Matrix Inverse:**
```json
{
  "problem_id": 12345,
  "step": 5,
  "landscape": "matrix_inverse_energy",
  "state": [0.12, -0.03, 0.87, ...],  // length = n²
  "energy": -2.45,
  "error_metric": 0.023  // MSE from true inverse
}
```

**For Shortest Path Planning:**
```json
{
  "problem_id": 67890,
  "step": 3,
  "landscape": "path_planning_energy",
  "state": [0.02, 0.85, 0.13, ...],   // length = N (num nodes)
  "energy": -1.67,
  "error_metric": 0.15   // BCE from optimal path
}
```

## Experimental Parameters

### Data Collection Scale
- **Target instances**: 50-200 problem instances
- **Steps per instance**: 10 (diffusion_steps default)
- **Total data points**: 500-2000 trajectory points

### Model Configuration
- **Energy landscape supervision**: `--supervise-energy-landscape True`
- **Innerloop optimization**: `--use-innerloop-opt True`

### Dataset-Specific Commands

**Matrix Inverse Collection:**
```bash
python3 train.py --dataset inverse --rank 20 \
  --data-workers 4 --batch_size 2048 \
  --use-innerloop-opt True \
  --supervise-energy-landscape True \
  --diffusion_steps 10
```

**Shortest Path Collection:**
```bash
python3 train.py --dataset shortest-path-1d \
  --model gnn-conv-1d-v2 \
  --data-workers 2 --batch_size 512 \
  --use-innerloop-opt True \
  --supervise-energy-landscape True \
  --diffusion_steps 10
```

## Trajectory Analysis Framework

### State Space Characterization

1. **Dimensionality**: 
   - Matrix inverse: n² dimensions (default 400)
   - Planning: N dimensions (variable by graph size)

2. **Value Ranges**:
   - Matrix inverse: Continuous values, typically normalized [-1, 1]
   - Planning: Probability-like values [0, 1] after softmax

3. **Trajectory Properties**:
   - Length: Fixed at 10 steps per instance
   - Temporal structure: Discrete diffusion process
   - Terminal condition: Convergence to solution

### Energy Landscape Analysis

The energy landscapes vary by model type:

- **Matrix Inverse Models**: Quadratic energy surfaces based on matrix reconstruction loss
- **Planning Models**: Discrete energy based on path optimality and connectivity constraints

### Data Storage Format

Trajectories will be stored as JSON Lines format for efficient processing:

```
{"problem_id": 1, "step": 0, "landscape": "...", "state": [...], "energy": ..., "error_metric": ...}
{"problem_id": 1, "step": 1, "landscape": "...", "state": [...], "energy": ..., "error_metric": ...}
...
{"problem_id": 200, "step": 9, "landscape": "...", "state": [...], "energy": ..., "error_metric": ...}
```

## Implementation Dependencies

This design depends on:
- **Task 3.2**: Analysis methods for processing collected trajectories
- **Task 4.2**: Actual trajectory logging implementation in codebase

## Success Criteria

1. **Data Completeness**: All required fields logged for every trajectory step
2. **Scale Achievement**: 50-200 complete problem instances collected
3. **Format Consistency**: Standardized state vector representation
4. **Energy Capture**: Valid energy values at each diffusion step
5. **Error Tracking**: Meaningful error metrics throughout convergence

## Technical Considerations

### Memory Requirements
- Matrix inverse: ~400 floats per state × 10 steps × 200 instances = ~800K values
- Planning: ~N floats per state (variable) × 10 steps × 200 instances

### Computational Overhead
- Logging should not significantly impact training performance
- Consider batched logging to reduce I/O overhead
- Energy computation may require model forward passes

### Data Quality Assurance
- Validate state vector dimensions match expected format
- Check for NaN/infinite values in energy calculations  
- Verify error metrics decrease over diffusion steps (for successful instances)