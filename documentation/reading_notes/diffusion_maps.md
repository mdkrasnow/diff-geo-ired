# Diffusion Maps: Algorithm and Application to IRED

## Intuition

Diffusion Maps is a nonlinear dimensionality reduction technique that discovers the intrinsic geometry of high-dimensional data by modeling it as a diffusion process on a manifold. The key insight is that data points that are similar (close together) should have high transition probabilities between them, while dissimilar points should have low transition probabilities. By analyzing the eigenvectors of the resulting diffusion operator, we can embed the data into a lower-dimensional space that preserves the intrinsic manifold structure and distances.

Unlike linear methods like PCA that find global linear structure, Diffusion Maps can capture complex nonlinear manifolds and is particularly effective at revealing the underlying geometry when data lies on curved or twisted structures. The method is also robust to noise and can handle data that lies on manifolds with varying local dimensionality. The "diffusion distance" between points captures the connectivity of the manifold better than Euclidean distance, making it ideal for analyzing trajectory data where the sequence of states follows complex paths through high-dimensional spaces.

## Algorithm

1. **Construct k-nearest neighbor graph**: For each data point xi, find its k nearest neighbors based on Euclidean distance in the original high-dimensional space. This creates a sparse graph structure that captures local neighborhood relationships.

2. **Compute pairwise similarities**: Calculate Gaussian kernel weights between connected points: W(xi, xj) = exp(-||xi - xj||²/ε), where ε is the bandwidth parameter. Points not connected in the k-NN graph have weight 0.

3. **Build degree-normalized transition matrix**: 
   - Compute row sums: d(xi) = Σj W(xi, xj)
   - Form the transition matrix: P(xi, xj) = W(xi, xj) / d(xi)
   - This creates a row-stochastic matrix representing transition probabilities

4. **Apply α-normalization (optional)**: Modify the kernel to K(xi, xj) = W(xi, xj) / (d(xi)^α × d(xj)^α) before renormalizing, which can help with non-uniform sampling density.

5. **Eigendecomposition**: Compute the top eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ and corresponding right eigenvectors ψ₁, ψ₂, ..., ψₙ of the transition matrix P.

6. **Construct diffusion coordinates**: The diffusion map embedding is given by: Ψₜ(xi) = [λ₁ᵗψ₁(xi), λ₂ᵗψ₂(xi), ..., λₘᵗψₘ(xi)], where t is the diffusion time parameter and m is the desired embedding dimension.

7. **Select embedding dimension**: Choose the number of significant eigenvectors based on the eigenvalue spectrum - typically where there's a gap in eigenvalues or where eigenvalues become small.

## Tuning Parameters

**k (number of neighbors)**: Controls the locality of the graph construction. Smaller k creates more local neighborhoods and can capture fine-grained structure, but may disconnect the graph. Larger k creates more global connectivity but may blur local structure. Typical range: 10-50 for most datasets. For trajectory data, k should be large enough to connect temporally adjacent states but small enough to preserve local geometry.

**ε (bandwidth/epsilon)**: Controls the scale of the Gaussian kernel and determines what distances are considered "close". Smaller ε creates sharper transitions and more local structure, while larger ε creates smoother transitions and more global structure. Common heuristics: median of k-th nearest neighbor distances, or adaptive selection based on local density. Critical parameter that significantly affects results.

**Number of eigenvectors/components**: Determines the dimensionality of the embedding space. Should be chosen based on the eigenvalue spectrum - look for gaps between eigenvalues or where eigenvalues decay to near zero. For visualization, typically 2-3 components. For analysis, may need more depending on the intrinsic dimensionality of the manifold.

**α (alpha normalization)**: Controls correction for non-uniform sampling density. α = 0 gives standard diffusion maps, α = 0.5 gives symmetric normalization, α = 1 gives Laplacian eigenmaps. For uniformly sampled data, α = 0 is usually sufficient. For non-uniform sampling, α = 0.5 or adaptive selection may help.

**Subsampling**: For very large datasets, may need to subsample points for computational efficiency. Can use random sampling or more sophisticated methods like landmark points. Trade-off between computational cost and preserving structure.

**Diffusion time t**: Controls the scale of diffusion process. t = 1 preserves local structure, larger t reveals more global structure by allowing diffusion to spread further. Often t = 1 is used, but can be tuned based on desired scale of analysis.

## Application to IRED

Diffusion Maps is particularly well-suited for analyzing IRED optimization trajectories because these trajectories represent paths through high-dimensional energy landscapes where the local geometry and connectivity are more important than global Euclidean distances. The energy function creates natural manifold structure where nearby states in the optimization process should be close in the embedding space.

For IRED trajectory analysis, Diffusion Maps can reveal several key insights: (1) The intrinsic dimensionality of the energy landscape - how many coordinates are needed to describe the essential optimization dynamics, (2) The smoothness and continuity of optimization paths - whether trajectories form coherent curves in the embedding space, (3) The relationship between energy and geometry - whether energy contours align with diffusion coordinates, and (4) Success vs failure patterns - whether successful and failed optimization runs occupy different regions of the manifold.

The temporal nature of IRED trajectories provides additional structure that Diffusion Maps can exploit. By treating each (problem, timestep) pair as a data point, we can analyze how the optimization process moves through the embedded space over time. This can reveal whether optimization follows predictable paths, whether there are bottlenecks or barriers in the landscape, and whether the embedding coordinates correlate with meaningful optimization metrics like energy decrease or convergence probability.

Parameter selection for IRED should consider the temporal structure: k should be large enough to connect temporally adjacent states within trajectories, ε should be chosen to capture the natural scale of state transitions during optimization, and the number of components should reflect the intrinsic dimensionality of the reasoning process rather than the high-dimensional state representation.

## Related Methods

**Laplacian Eigenmaps** is closely related to Diffusion Maps and can be obtained as a special case with α = 1 normalization. The main difference is that Laplacian Eigenmaps focuses on the graph Laplacian rather than the transition matrix, leading to slightly different preservation properties. Laplacian Eigenmaps tends to preserve locality better while Diffusion Maps provides better global structure preservation.

**t-SNE** and **UMAP** are other nonlinear dimensionality reduction methods but focus more on local neighborhood preservation for visualization rather than preserving diffusion distances or manifold geometry. They may be more suitable for visualization but less suitable for quantitative analysis of manifold properties.

**Isomap** and **Locally Linear Embedding (LLE)** are earlier manifold learning methods that also attempt to preserve intrinsic manifold structure but use different approaches (geodesic distances for Isomap, local linear reconstruction for LLE). Diffusion Maps generally provides better robustness to noise and more principled treatment of the manifold geometry.