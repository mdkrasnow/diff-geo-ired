# Methods and Case Study

## Geometric Interpretation of IRED

### IRED as Discrete Gradient Flow on Manifolds

IRED (Iterative Reasoning Energy Diffusion) can be interpreted geometrically as discrete gradient flow on a learned manifold structure in high-dimensional state space. The optimization process follows the fundamental equation:

```
y_{t+1} = y_t - α ∇_y E_θ(x, y_t, k_t)
```

where `y_t ∈ ℝ^n` represents the state vector at diffusion step `t`, `E_θ(x,y,k)` is the learned energy function parameterized by `θ`, and `k_t` is the time-varying landscape index parameter.

### State Space Manifold Structure

The IRED state space can be characterized as a Riemannian manifold `M` embedded in the ambient space `ℝ^n`, where:

- **Manifold Points**: Each state vector `y_t` represents a point on the solution manifold
- **Tangent Space**: The space `T_{y_t} M` of possible optimization directions at state `y_t`
- **Metric Structure**: Induced by the energy Hessian `∇²_y E_θ(x,y,k)`, defining local distance measurements
- **Curvature Properties**: Varying with landscape parameter `k` to control exploration vs. exploitation

### Energy Landscape Evolution

The landscape parameter `k` modulates the geometric properties of the energy surface:

- **Early Diffusion (large k)**: Smooth, low-curvature landscapes promoting global exploration
- **Later Diffusion (small k)**: Sharp, high-curvature landscapes enabling local refinement
- **Geometric Transition**: Continuous evolution from convex-like to multi-modal energy surfaces

### Connection to Discrete Gradient Flow

IRED updates represent discrete approximations to the continuous gradient flow equation:

```
dy/dt = -∇_y E_θ(x,y,k(t))
```

This flow has several key geometric properties:
1. **Energy Dissipation**: `dE/dt = -|∇_y E_θ|² ≤ 0`
2. **Orthogonality**: Flow lines intersect energy level sets perpendicularly
3. **Convergence**: Trajectories terminate at critical points (solutions)

## Case Study: Matrix Inverse Problems

### Problem Formulation and Geometric Structure

We analyze IRED trajectories on the matrix inverse computation task, which provides a mathematically well-founded case study with clear geometric interpretations. The problem structure is:

- **Input Space**: Symmetric positive definite matrices `A ∈ ℝ^{20×20}`
- **Output Space**: Matrix inverses `B = A^{-1} ∈ ℝ^{20×20}`
- **State Vector**: Flattened representation `y_t ∈ ℝ^{400}` (vectorized matrix)
- **Solution Manifold**: The space of valid inverse matrices with geometric structure inherited from the matrix group

### Mathematical Foundation

Matrix inversion exhibits rich differential geometric structure:

1. **Lie Group Structure**: The general linear group GL(n,ℝ) of invertible matrices forms a Lie group with well-defined geodesics
2. **Positive Definite Manifold**: Target matrices lie on the manifold of symmetric positive definite matrices, which has negative sectional curvature
3. **Natural Metric**: The Fisher information metric provides a canonical Riemannian structure

### Trajectory Analysis Framework

Each IRED optimization instance generates a discrete trajectory:
```
{y_0, y_1, y_2, ..., y_T}
```
where each `y_t` represents the flattened matrix estimate at diffusion step `t`, and `T = 10` is the total number of refinement steps.

## Trajectory Logging Methodology

### Data Collection Protocol

Trajectory data is collected during IRED training using the following experimental configuration:

```bash
python3 train.py --dataset inverse --rank 20 \
  --data-workers 4 --batch_size 2048 \
  --use-innerloop-opt True \
  --supervise-energy-landscape True \
  --diffusion_steps 10
```

### Logged Data Structure

Each trajectory point is recorded with the following fields:

1. **problem_id** (int): Unique identifier for each matrix inverse instance
2. **step** (int): Diffusion step index (0 ≤ step ≤ 10)
3. **landscape** (string): Energy landscape identifier for geometric analysis
4. **state** (array): 400-dimensional state vector (flattened matrix estimate)
5. **energy** (float): Energy value `E_θ(x,y_t,k_t)` at current state
6. **error_metric** (float): Mean squared error from true matrix inverse

### Data Storage Format

Trajectories are stored in JSON Lines format for efficient processing:

```json
{"problem_id": 1, "step": 0, "landscape": "matrix_inverse", "state": [...], "energy": -2.45, "error_metric": 0.82}
{"problem_id": 1, "step": 1, "landscape": "matrix_inverse", "state": [...], "energy": -3.21, "error_metric": 0.31}
...
```

### Collection Scale and Validation

- **Target Scale**: 100-200 problem instances providing 1000-2000 trajectory points
- **Validation Checks**: State vector dimensionality, energy monotonicity, convergence criteria
- **Quality Assurance**: NaN/infinity detection, error metric bounds verification

## Manifold Learning Pipeline

### Dimensionality Reduction Framework

Given the high-dimensional nature of matrix inverse states (400D), we implement a two-stage manifold learning pipeline:

**Stage 1: Linear Preprocessing**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
reduced_states = pca.fit_transform(trajectory_states)
```

**Stage 2: Nonlinear Manifold Learning**
```python
from sklearn.manifold import Isomap
isomap = Isomap(n_components=3, n_neighbors=10)
embedded_trajectories = isomap.fit_transform(reduced_states)
```

### Manifold Learning Methods

We apply three complementary manifold learning techniques to capture different aspects of trajectory geometry:

#### 1. Principal Component Analysis (PCA)
- **Purpose**: Linear dimensionality reduction and variance analysis
- **Geometric Insight**: Identifies principal directions of variation in state space
- **Implementation**: `sklearn.decomposition.PCA` with 50 components retaining >95% variance

#### 2. Isomap (Isometric Mapping)
- **Purpose**: Preserves geodesic distances on the trajectory manifold
- **Geometric Insight**: Reveals intrinsic distance structure and path connectivity
- **Parameters**: 
  - `n_neighbors=10`: Local neighborhood size for graph construction
  - `n_components=3`: 3D embedding for visualization
  - `metric='euclidean'`: Distance metric for neighborhood computation

#### 3. Laplacian Eigenmaps
- **Purpose**: Spectral embedding preserving local neighborhood structure
- **Geometric Insight**: Identifies smoothest coordinates on the trajectory manifold
- **Implementation**: `sklearn.manifold.SpectralEmbedding` with Gaussian kernel

### Pipeline Implementation

The complete manifold learning pipeline processes trajectory data as follows:

1. **Data Preparation**: Concatenate all trajectory points into a single dataset
2. **Preprocessing**: Apply PCA for initial dimensionality reduction
3. **Manifold Embedding**: Apply Isomap and Laplacian Eigenmaps in parallel
4. **Trajectory Reconstruction**: Separate embedded points back into individual trajectories
5. **Visualization**: Generate 3D plots of embedded trajectory curves

## Geometric Diagnostic Definitions

### Discrete Curvature Computation

For discrete trajectory points `{y_0, y_1, ..., y_T}`, we compute curvature using finite difference approximations:

#### Method 1: Three-Point Curvature
```python
def discrete_curvature(p_prev, p_curr, p_next):
    """Compute discrete curvature at p_curr using three consecutive points."""
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Compute curvature as angle change
    cos_theta = np.dot(v1_norm, v2_norm)
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
    
    return np.arccos(cos_theta)
```

#### Method 2: Menger Curvature
For three points forming a triangle, the Menger curvature provides a robust discrete approximation:

```python
def menger_curvature(p1, p2, p3):
    """Compute Menger curvature of triangle formed by three points."""
    # Compute side lengths
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3) 
    c = np.linalg.norm(p1 - p2)
    
    # Area using Heron's formula
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Menger curvature = 4 * area / (a * b * c)
    return 4 * area / (a * b * c)
```

### Path Length Metrics

#### Cumulative Arc Length
```python
def trajectory_arc_length(trajectory):
    """Compute cumulative arc length along trajectory."""
    distances = []
    for i in range(1, len(trajectory)):
        dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
        distances.append(dist)
    return np.cumsum(distances)
```

#### Energy-Weighted Path Length
```python
def energy_weighted_path_length(trajectory, energies):
    """Compute path length weighted by energy values."""
    weighted_length = 0
    for i in range(1, len(trajectory)):
        segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
        energy_weight = (energies[i] + energies[i-1]) / 2
        weighted_length += segment_length * energy_weight
    return weighted_length
```

### Geometric Convergence Diagnostics

#### Trajectory Smoothness Measure
```python
def trajectory_smoothness(trajectory):
    """Measure trajectory smoothness using second derivatives."""
    if len(trajectory) < 3:
        return 0
    
    second_derivatives = []
    for i in range(1, len(trajectory) - 1):
        # Discrete second derivative
        d2y = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
        second_derivatives.append(np.linalg.norm(d2y))
    
    return np.mean(second_derivatives)
```

#### Solution Manifold Distance
```python
def solution_manifold_distance(state, true_solution):
    """Compute distance to solution manifold (matrix inverse case)."""
    # Reshape to matrix form
    estimated_matrix = state.reshape(20, 20)
    true_matrix = true_solution.reshape(20, 20)
    
    # Frobenius norm distance
    return np.linalg.norm(estimated_matrix - true_matrix, 'fro')
```

### Analysis Pipeline Integration

The geometric diagnostics are computed for each trajectory and aggregated to provide insights into:

1. **Convergence Patterns**: How curvature and path length evolve during optimization
2. **Landscape Dependencies**: How geometric properties change with landscape parameter `k`
3. **Solution Quality Correlation**: Relationship between geometric measures and final solution accuracy
4. **Manifold Structure**: Intrinsic dimensionality and topological properties of trajectory space

## Experimental Validation Framework

### Result Files and Data Organization

All experimental results are stored in standardized formats within `~/documentation/results/`:

- **`trajectory_data.jsonl`**: Raw trajectory logging data
- **`manifold_embeddings.pkl`**: Saved manifold learning results  
- **`geometric_diagnostics.csv`**: Computed curvature and path length metrics
- **`convergence_analysis.json`**: Aggregated convergence statistics
- **`visualization_plots/`**: Generated trajectory and embedding visualizations

### Reproducibility and Validation

1. **Random Seed Control**: All algorithms use fixed random seeds for reproducibility
2. **Cross-Validation**: Geometric measures computed using multiple methods for robustness
3. **Numerical Stability**: Condition number monitoring and regularization for matrix operations
4. **Error Bounds**: Confidence intervals for all statistical measures

This comprehensive methodology provides a rigorous framework for analyzing the differential geometric properties of IRED optimization trajectories, with specific focus on the matrix inverse case study as a mathematically well-founded testbed for developing and validating geometric analysis techniques.