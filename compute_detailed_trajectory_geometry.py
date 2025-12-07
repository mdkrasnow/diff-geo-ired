#!/usr/bin/env python3
"""
Compute detailed trajectory geometric properties for IRED analysis.
Extends the embedding analysis with path length and discrete curvature calculations.
Task 5.2 implementation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compute_path_length(trajectory_points):
    """
    Compute discrete path length as sum of Euclidean distances between consecutive points.
    
    Args:
        trajectory_points: Array of shape (n_points, n_dims) representing trajectory points
        
    Returns:
        Total path length as float
    """
    if len(trajectory_points) < 2:
        return 0.0
    
    # Compute differences between consecutive points
    diffs = np.diff(trajectory_points, axis=0)
    
    # Compute Euclidean distances using numpy linalg.norm
    distances = np.linalg.norm(diffs, axis=1)
    
    return float(np.sum(distances))

def compute_discrete_curvature(trajectory_points, epsilon=1e-8):
    """
    Compute discrete curvature for interior points only.
    
    Curvature formula: κ_t = |y_{t+1} - 2y_t + y_{t-1}| / |y_{t+1} - y_t|²
    with epsilon regularization for numerical stability.
    
    Args:
        trajectory_points: Array of shape (n_points, n_dims) representing trajectory points
        epsilon: Small value added to denominator to prevent division by zero
        
    Returns:
        List of curvature values for interior points (excludes endpoints)
    """
    if len(trajectory_points) < 3:
        return []
    
    curvatures = []
    
    # Only compute for interior points (skip endpoints)
    for t in range(1, len(trajectory_points) - 1):
        y_prev = trajectory_points[t-1]
        y_curr = trajectory_points[t]
        y_next = trajectory_points[t+1]
        
        # Second derivative approximation (discrete Laplacian)
        second_deriv = y_next - 2*y_curr + y_prev
        
        # First derivative approximation (forward difference)
        first_deriv = y_next - y_curr
        
        # Magnitude of second derivative (numerator)
        numerator = np.linalg.norm(second_deriv)
        
        # Squared magnitude of first derivative with epsilon regularization (denominator)
        denominator = np.linalg.norm(first_deriv)**2 + epsilon
        
        # Discrete curvature
        curvature = numerator / denominator
        curvatures.append(float(curvature))
    
    return curvatures

def analyze_trajectory_geometry_detailed(embedding, problem_indices, step_indices, embedding_name):
    """
    Analyze geometric properties for all trajectories in the given embedding space.
    
    Args:
        embedding: Array of shape (n_total_points, 2) with embedding coordinates
        problem_indices: Array mapping each point to its problem/trajectory ID
        step_indices: Array mapping each point to its step within the trajectory
        embedding_name: Name of the embedding ('PCA' or 'Isomap')
        
    Returns:
        Tuple of (summary_results, detailed_curvature_results)
    """
    print(f"=== Analyzing {embedding_name} Trajectory Geometry ===")
    
    summary_results = []
    detailed_curvature_results = []
    
    # Get unique problems
    unique_problems = np.unique(problem_indices)
    
    for problem_id in unique_problems:
        # Get points for this trajectory
        traj_mask = problem_indices == problem_id
        traj_points = embedding[traj_mask]
        traj_steps = step_indices[traj_mask]
        
        # Sort by step index to ensure correct order
        sort_indices = np.argsort(traj_steps)
        traj_points = traj_points[sort_indices]
        
        # Compute path length
        path_length = compute_path_length(traj_points)
        
        # Compute curvatures for interior points
        curvatures = compute_discrete_curvature(traj_points)
        
        # Summary statistics for curvatures
        if curvatures:
            mean_curvature = np.mean(curvatures)
            max_curvature = np.max(curvatures)
            min_curvature = np.min(curvatures)
            std_curvature = np.std(curvatures)
            median_curvature = np.median(curvatures)
        else:
            mean_curvature = max_curvature = min_curvature = std_curvature = median_curvature = 0.0
        
        # Summary record
        summary_record = {
            'trajectory_id': int(problem_id),
            'embedding_type': embedding_name,
            'path_length': path_length,
            'mean_curvature': mean_curvature,
            'max_curvature': max_curvature,
            'min_curvature': min_curvature,
            'std_curvature': std_curvature,
            'median_curvature': median_curvature,
            'num_points': len(traj_points),
            'num_curvature_points': len(curvatures)
        }
        summary_results.append(summary_record)
        
        # Detailed curvature records
        for i, curvature in enumerate(curvatures):
            detailed_record = {
                'trajectory_id': int(problem_id),
                'embedding_type': embedding_name,
                'point_index': i + 1,  # Interior point indices (1 to n-2)
                'curvature': curvature
            }
            detailed_curvature_results.append(detailed_record)
    
    print(f"Processed {len(unique_problems)} trajectories")
    print(f"Average path length: {np.mean([r['path_length'] for r in summary_results]):.4f}")
    print(f"Average mean curvature: {np.mean([r['mean_curvature'] for r in summary_results]):.6f}")
    
    return summary_results, detailed_curvature_results

def create_validation_plots(lengths_df, curvatures_df, output_dir):
    """Create validation plots for the geometric analysis."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Path length distributions
    for i, emb_type in enumerate(['PCA', 'Isomap']):
        subset = lengths_df[lengths_df['embedding_type'] == emb_type]
        axes[0, 0].hist(subset['path_length'], bins=20, alpha=0.6, label=f'{emb_type}', density=True)
    axes[0, 0].set_xlabel('Path Length')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Path Lengths')
    axes[0, 0].legend()

    # Mean curvature distributions
    for i, emb_type in enumerate(['PCA', 'Isomap']):
        subset = lengths_df[lengths_df['embedding_type'] == emb_type]
        axes[0, 1].hist(subset['mean_curvature'], bins=20, alpha=0.6, label=f'{emb_type}', density=True)
    axes[0, 1].set_xlabel('Mean Curvature')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Mean Curvatures')
    axes[0, 1].legend()

    # Path length vs mean curvature scatter
    for emb_type, color in [('PCA', 'blue'), ('Isomap', 'red')]:
        subset = lengths_df[lengths_df['embedding_type'] == emb_type]
        axes[0, 2].scatter(subset['path_length'], subset['mean_curvature'], 
                          alpha=0.6, color=color, label=emb_type, s=20)
    axes[0, 2].set_xlabel('Path Length')
    axes[0, 2].set_ylabel('Mean Curvature')
    axes[0, 2].set_title('Path Length vs Mean Curvature')
    axes[0, 2].legend()

    # Detailed curvature distributions
    for emb_type, color in [('PCA', 'blue'), ('Isomap', 'red')]:
        subset = curvatures_df[curvatures_df['embedding_type'] == emb_type]
        axes[1, 0].hist(subset['curvature'], bins=50, alpha=0.6, color=color, 
                       label=f'{emb_type}', density=True, range=(0, np.percentile(subset['curvature'], 95)))
    axes[1, 0].set_xlabel('Local Curvature')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Local Curvatures (95th percentile)')
    axes[1, 0].legend()

    # Curvature along trajectory examples
    example_trajectories = [0, 1, 2, 5, 10]
    for i, traj_id in enumerate(example_trajectories):
        if i >= 5:
            break
        
        pca_curvs = curvatures_df[(curvatures_df['trajectory_id'] == traj_id) & 
                                 (curvatures_df['embedding_type'] == 'PCA')]
        isomap_curvs = curvatures_df[(curvatures_df['trajectory_id'] == traj_id) & 
                                    (curvatures_df['embedding_type'] == 'Isomap')]
        
        if len(pca_curvs) > 0:
            axes[1, 1].plot(pca_curvs['point_index'], pca_curvs['curvature'], 
                           'o-', alpha=0.7, markersize=3, linewidth=1, label=f'PCA Traj {traj_id}' if i < 2 else '')
        if len(isomap_curvs) > 0:
            axes[1, 2].plot(isomap_curvs['point_index'], isomap_curvs['curvature'], 
                           's-', alpha=0.7, markersize=3, linewidth=1, label=f'Isomap Traj {traj_id}' if i < 2 else '')

    axes[1, 1].set_xlabel('Point Index (Interior Points)')
    axes[1, 1].set_ylabel('Curvature')
    axes[1, 1].set_title('PCA Curvature Along Trajectories (Examples)')
    if len(example_trajectories) <= 2:
        axes[1, 1].legend()

    axes[1, 2].set_xlabel('Point Index (Interior Points)')
    axes[1, 2].set_ylabel('Curvature') 
    axes[1, 2].set_title('Isomap Curvature Along Trajectories (Examples)')
    if len(example_trajectories) <= 2:
        axes[1, 2].legend()

    plt.tight_layout()
    
    plot_path = output_dir / 'trajectory_geometric_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_path

def main():
    """Main analysis function for Task 5.2"""
    print("="*60)
    print("🎯 TASK 5.2: TRAJECTORY GEOMETRIC PROPERTIES ANALYSIS")
    print("="*60)
    
    # Set up paths
    results_dir = Path('documentation/results')
    figures_dir = Path('documentation/figures')
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Load existing embedding analysis data
    print("\n📂 Loading trajectory embedding data...")
    data_path = results_dir / 'ired_embedding_analysis.npz'
    
    if not data_path.exists():
        print(f"❌ ERROR: Required data file not found: {data_path}")
        print("Please run the embedding analysis notebook first (Task 5.1)")
        return
    
    data = np.load(data_path, allow_pickle=True)
    
    # Extract data
    pca_embedding = data['pca_embedding']
    isomap_embedding = data['isomap_embedding']
    problem_indices = data['problem_indices']
    step_indices = data['step_indices']
    num_problems = int(data['num_problems'])
    num_steps = int(data['num_steps'])
    
    print(f"✅ Loaded embedding data:")
    print(f"   • PCA embedding: {pca_embedding.shape}")
    print(f"   • Isomap embedding: {isomap_embedding.shape}")
    print(f"   • Trajectories: {num_problems}")
    print(f"   • Steps per trajectory: {num_steps}")
    
    # Analyze both embeddings
    print(f"\n🔄 Computing geometric properties...")
    
    pca_summary, pca_detailed = analyze_trajectory_geometry_detailed(
        pca_embedding, problem_indices, step_indices, 'PCA'
    )

    isomap_summary, isomap_detailed = analyze_trajectory_geometry_detailed(
        isomap_embedding, problem_indices, step_indices, 'Isomap'
    )

    print(f"\n📊 Analysis complete:")
    print(f"   • Summary records: {len(pca_summary) + len(isomap_summary)}")
    print(f"   • Detailed curvature records: {len(pca_detailed) + len(isomap_detailed)}")

    # Combine results and save to CSV files as specified in the task
    print(f"\n💾 Saving results to CSV files...")
    
    # Combine summary results (path lengths)
    all_summary_results = pca_summary + isomap_summary
    lengths_df = pd.DataFrame(all_summary_results)

    # Save path length results
    lengths_csv_path = results_dir / 'ired_trajectory_lengths.csv'
    lengths_df.to_csv(lengths_csv_path, index=False)
    print(f"✅ Saved trajectory lengths to: {lengths_csv_path}")

    # Combine detailed curvature results
    all_curvature_results = pca_detailed + isomap_detailed
    curvatures_df = pd.DataFrame(all_curvature_results)

    # Save curvature results
    curvatures_csv_path = results_dir / 'ired_trajectory_curvatures.csv'
    curvatures_df.to_csv(curvatures_csv_path, index=False)
    print(f"✅ Saved trajectory curvatures to: {curvatures_csv_path}")

    # Display sample data
    print(f"\n📋 Data Preview:")
    print(f"\nTrajectory Lengths CSV (shape: {lengths_df.shape}):")
    print(lengths_df.head(10))

    print(f"\nTrajectory Curvatures CSV (shape: {curvatures_df.shape}):")
    print(curvatures_df.head(10))

    # Summary statistics by embedding type
    print(f"\n📈 Summary Statistics by Embedding Type:")
    summary_stats = lengths_df.groupby('embedding_type')[['path_length', 'mean_curvature']].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ])
    print(summary_stats)

    # Create validation plots
    print(f"\n📊 Creating validation plots...")
    plot_path = create_validation_plots(lengths_df, curvatures_df, figures_dir)
    print(f"✅ Saved plots to: {plot_path}")

    # Data quality validation
    print(f"\n🔍 Data Quality Validation:")
    print(f"   • Path lengths - Min: {lengths_df['path_length'].min():.6f}, Max: {lengths_df['path_length'].max():.3f}")
    print(f"   • Mean curvatures - Min: {lengths_df['mean_curvature'].min():.6f}, Max: {lengths_df['mean_curvature'].max():.6f}")
    print(f"   • Local curvatures - Min: {curvatures_df['curvature'].min():.6f}, Max: {curvatures_df['curvature'].max():.6f}")
    print(f"   • Negative curvatures: {(curvatures_df['curvature'] < 0).sum()} (should be 0)")
    print(f"   • Infinite curvatures: {np.isinf(curvatures_df['curvature']).sum()} (should be 0)")
    print(f"   • NaN curvatures: {np.isnan(curvatures_df['curvature']).sum()} (should be 0)")

    # Final summary
    print(f"\n" + "="*60)
    print("✨ TASK 5.2 COMPLETION SUMMARY")
    print("="*60)

    print(f"\n✅ COMPLETED OBJECTIVES:")
    print(f"   • Extended analysis with path length and discrete curvature calculations")
    print(f"   • Used numpy.linalg.norm for distance calculations")
    print(f"   • Applied epsilon=1e-8 regularization for division by zero prevention")
    print(f"   • Computed curvatures for interior points only (skipped endpoints)")
    print(f"   • Saved results in CSV format for easy analysis")

    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"   • {lengths_csv_path}")
    print(f"   • {curvatures_csv_path}")
    print(f"   • {plot_path}")

    print(f"\n📊 FINAL DATA SUMMARY:")
    print(f"   • Trajectories analyzed: {len(lengths_df)//2} per embedding ({len(lengths_df)} total records)")
    print(f"   • Embedding types: {', '.join(lengths_df['embedding_type'].unique())}")
    print(f"   • Curvature measurements: {len(curvatures_df)} interior points")
    
    for emb_type in ['PCA', 'Isomap']:
        subset = lengths_df[lengths_df['embedding_type'] == emb_type]
        avg_length = subset['path_length'].mean()
        avg_curvature = subset['mean_curvature'].mean()
        print(f"   • Average path length ({emb_type}): {avg_length:.4f}")
        print(f"   • Average mean curvature ({emb_type}): {avg_curvature:.6f}")

    print(f"\n✨ Ready for further geometric analysis and publication!")
    print("="*60)

if __name__ == '__main__':
    main()