#!/usr/bin/env python3
"""
Trajectory Geometric Property Analysis for IRED Dataset

Computes path lengths and discrete curvatures for trajectory data in both
PCA and nonlinear embedding spaces. Saves results as CSV files with robust
error handling and numerical stability measures.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

def compute_path_length(trajectory: np.ndarray) -> float:
    """
    Compute discrete path length as sum of Euclidean distances between consecutive points.
    
    Args:
        trajectory: Array of shape (n_points, n_dims) representing trajectory points
        
    Returns:
        Total path length as float
    """
    if len(trajectory) < 2:
        return 0.0
    
    # Compute differences between consecutive points
    diffs = np.diff(trajectory, axis=0)
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    return float(np.sum(distances))

def compute_discrete_curvature(trajectory: np.ndarray, epsilon: float = 1e-8) -> List[float]:
    """
    Compute discrete curvature for interior points only.
    
    Curvature formula: κ_t = |y_{t+1} - 2y_t + y_{t-1}| / |y_{t+1} - y_t|²
    with epsilon regularization for numerical stability.
    
    Args:
        trajectory: Array of shape (n_points, n_dims) representing trajectory points
        epsilon: Small value added to denominator to prevent division by zero
        
    Returns:
        List of curvature values for interior points (excludes endpoints)
    """
    if len(trajectory) < 3:
        return []
    
    curvatures = []
    
    # Only compute for interior points (skip endpoints)
    for t in range(1, len(trajectory) - 1):
        y_prev = trajectory[t-1]
        y_curr = trajectory[t]
        y_next = trajectory[t+1]
        
        # Second derivative approximation (discrete Laplacian)
        second_deriv = y_next - 2*y_curr + y_prev
        
        # First derivative approximation (forward difference)
        first_deriv = y_next - y_curr
        
        # Magnitude of second derivative (numerator)
        numerator = np.sqrt(np.sum(second_deriv**2))
        
        # Squared magnitude of first derivative with epsilon regularization (denominator)
        denominator = np.sum(first_deriv**2) + epsilon
        
        # Discrete curvature
        curvature = numerator / denominator
        curvatures.append(float(curvature))
    
    return curvatures

def load_trajectory_data(data_dir: str = ".") -> Tuple[Optional[Dict], Optional[Dict], Optional[List]]:
    """
    Load trajectory data from various possible sources.
    
    Args:
        data_dir: Directory to search for trajectory data files
        
    Returns:
        Tuple of (pca_trajectories, nonlinear_trajectories, trajectory_names)
    """
    pca_trajectories = None
    nonlinear_trajectories = None
    trajectory_names = None
    
    # Look for common trajectory data file patterns
    possible_files = [
        'trajectories.pkl',
        'trajectory_data.pkl', 
        'ired_trajectories.pkl',
        'embedding_data.pkl',
        'results.pkl'
    ]
    
    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                print(f'Loading trajectory data from {filepath}')
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                # Try to extract trajectory data based on common structures
                if isinstance(data, dict):
                    if 'pca_trajectories' in data:
                        pca_trajectories = data['pca_trajectories']
                    if 'nonlinear_trajectories' in data:
                        nonlinear_trajectories = data['nonlinear_trajectories']
                    if 'trajectory_names' in data:
                        trajectory_names = data['trajectory_names']
                    
                    # Check for other common keys
                    for key in data.keys():
                        if 'pca' in key.lower() and 'traj' in key.lower():
                            pca_trajectories = data[key]
                        elif 'nonlinear' in key.lower() and 'traj' in key.lower():
                            nonlinear_trajectories = data[key] 
                        elif 'name' in key.lower():
                            trajectory_names = data[key]
                
                if pca_trajectories is not None or nonlinear_trajectories is not None:
                    break
                    
            except Exception as e:
                print(f'Could not load {filepath}: {e}')
                continue
    
    # Look for results directory with trajectory data
    results_dir = os.path.join(data_dir, 'results')
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.pkl') and 'traj' in filename.lower():
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and len(data) > 0:
                        print(f'Found trajectory data in {filepath}')
                        if pca_trajectories is None:
                            pca_trajectories = data
                        elif nonlinear_trajectories is None:
                            nonlinear_trajectories = data
                        break
                except:
                    continue
    
    # Generate default trajectory names if not found
    if trajectory_names is None and (pca_trajectories is not None or nonlinear_trajectories is not None):
        reference_dict = pca_trajectories if pca_trajectories is not None else nonlinear_trajectories
        trajectory_names = list(reference_dict.keys())
    
    return pca_trajectories, nonlinear_trajectories, trajectory_names

def analyze_trajectory_geometry(trajectories_dict: Dict[str, np.ndarray], 
                              trajectory_names: List[str],
                              embedding_type: str) -> List[Dict]:
    """
    Analyze geometric properties for all trajectories in the given embedding space.
    
    Args:
        trajectories_dict: Dictionary mapping trajectory names to trajectory arrays
        trajectory_names: List of trajectory names to process
        embedding_type: Type of embedding ('PCA' or 'Nonlinear')
        
    Returns:
        List of dictionaries containing geometric properties for each trajectory
    """
    results = []
    
    for name in trajectory_names:
        if name not in trajectories_dict:
            print(f'Warning: trajectory {name} not found in {embedding_type} data')
            continue
            
        trajectory = trajectories_dict[name]
        
        # Ensure trajectory is a numpy array
        if not isinstance(trajectory, np.ndarray):
            try:
                trajectory = np.array(trajectory)
            except:
                print(f'Warning: could not convert trajectory {name} to array')
                continue
        
        # Handle degenerate cases
        if trajectory.size == 0:
            print(f'Warning: empty trajectory {name}')
            continue
        
        if len(trajectory.shape) != 2:
            print(f'Warning: trajectory {name} has unexpected shape {trajectory.shape}')
            continue
        
        try:
            # Compute path length
            path_length = compute_path_length(trajectory)
            
            # Compute curvatures for interior points  
            curvatures = compute_discrete_curvature(trajectory)
            
            # Summary statistics for curvatures
            if curvatures:
                mean_curvature = np.mean(curvatures)
                max_curvature = np.max(curvatures)
                min_curvature = np.min(curvatures)
                std_curvature = np.std(curvatures)
                median_curvature = np.median(curvatures)
            else:
                mean_curvature = max_curvature = min_curvature = std_curvature = median_curvature = 0.0
            
            result_row = {
                'trajectory_name': name,
                'embedding_type': embedding_type,
                'path_length': path_length,
                'mean_curvature': mean_curvature,
                'max_curvature': max_curvature,
                'min_curvature': min_curvature,
                'std_curvature': std_curvature,
                'median_curvature': median_curvature,
                'num_points': len(trajectory),
                'num_curvature_points': len(curvatures),
                'dimensionality': trajectory.shape[1] if len(trajectory.shape) > 1 else 1
            }
            
            results.append(result_row)
            
        except Exception as e:
            print(f'Error processing trajectory {name}: {e}')
            continue
    
    return results

def save_detailed_curvatures(trajectories_dict: Dict[str, np.ndarray], 
                           trajectory_names: List[str],
                           embedding_type: str,
                           output_path: str) -> None:
    """
    Save detailed per-point curvature data to CSV.
    
    Args:
        trajectories_dict: Dictionary mapping trajectory names to trajectory arrays
        trajectory_names: List of trajectory names to process  
        embedding_type: Type of embedding ('PCA' or 'Nonlinear')
        output_path: Path to save detailed curvature CSV
    """
    detailed_rows = []
    
    for name in trajectory_names:
        if name not in trajectories_dict:
            continue
            
        trajectory = trajectories_dict[name]
        if not isinstance(trajectory, np.ndarray):
            try:
                trajectory = np.array(trajectory)
            except:
                continue
                
        if trajectory.size == 0 or len(trajectory.shape) != 2:
            continue
            
        try:
            curvatures = compute_discrete_curvature(trajectory)
            
            # Create row for each curvature point
            for i, curvature in enumerate(curvatures):
                detailed_rows.append({
                    'trajectory_name': name,
                    'embedding_type': embedding_type,
                    'point_index': i + 1,  # Interior point indices (1 to n-2)
                    'curvature': curvature
                })
                
        except Exception as e:
            continue
    
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df.to_csv(output_path, index=False)
        print(f'Saved detailed curvatures to {output_path}')

def main():
    """Main analysis function"""
    print('IRED Trajectory Geometric Property Analysis')
    print('=' * 50)
    
    # Create output directory
    os.makedirs('documentation/results', exist_ok=True)
    
    # Load trajectory data
    print('Loading trajectory data...')
    pca_trajectories, nonlinear_trajectories, trajectory_names = load_trajectory_data()
    
    if pca_trajectories is None and nonlinear_trajectories is None:
        print('Error: No trajectory data found!')
        print('Please ensure trajectory data files exist in the current directory or results/ subdirectory')
        return
    
    all_results = []
    
    # Analyze PCA trajectories
    if pca_trajectories is not None:
        print(f'\nAnalyzing PCA trajectories ({len(pca_trajectories)} trajectories)...')
        pca_results = analyze_trajectory_geometry(pca_trajectories, trajectory_names, 'PCA')
        all_results.extend(pca_results)
        
        # Save PCA-specific results
        if pca_results:
            pca_df = pd.DataFrame(pca_results)
            pca_df.to_csv('documentation/results/ired_trajectory_lengths_pca.csv', index=False)
            print(f'Saved PCA trajectory summary for {len(pca_results)} trajectories')
            
            # Save detailed curvatures
            save_detailed_curvatures(pca_trajectories, trajectory_names, 'PCA',
                                   'documentation/results/ired_trajectory_curvatures_pca.csv')
    
    # Analyze nonlinear trajectories  
    if nonlinear_trajectories is not None:
        print(f'\nAnalyzing nonlinear trajectories ({len(nonlinear_trajectories)} trajectories)...')
        nonlinear_results = analyze_trajectory_geometry(nonlinear_trajectories, trajectory_names, 'Nonlinear')
        all_results.extend(nonlinear_results)
        
        # Save nonlinear-specific results
        if nonlinear_results:
            nonlinear_df = pd.DataFrame(nonlinear_results)
            nonlinear_df.to_csv('documentation/results/ired_trajectory_lengths_nonlinear.csv', index=False)
            print(f'Saved nonlinear trajectory summary for {len(nonlinear_results)} trajectories')
            
            # Save detailed curvatures
            save_detailed_curvatures(nonlinear_trajectories, trajectory_names, 'Nonlinear',
                                   'documentation/results/ired_trajectory_curvatures_nonlinear.csv')
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        
        # Save main output files as specified in assignment
        combined_df.to_csv('documentation/results/ired_trajectory_lengths.csv', index=False)
        combined_df.to_csv('documentation/results/trajectory_geometric_summary.csv', index=False)
        
        print(f'\nSaved combined geometric summary for {len(all_results)} total trajectory records')
        
        # Display summary statistics
        print('\nGeometric Property Summary by Embedding Type:')
        print('-' * 60)
        
        summary_stats = combined_df.groupby('embedding_type')[['path_length', 'mean_curvature', 'num_points']].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ])
        print(summary_stats)
        
        # Additional insights
        print('\nTrajectory Complexity Analysis:')
        print('-' * 40)
        for embedding_type in combined_df['embedding_type'].unique():
            subset = combined_df[combined_df['embedding_type'] == embedding_type]
            avg_length = subset['path_length'].mean()
            avg_curvature = subset['mean_curvature'].mean()
            avg_points = subset['num_points'].mean()
            
            print(f'{embedding_type}:')
            print(f'  Average path length: {avg_length:.4f}')
            print(f'  Average mean curvature: {avg_curvature:.6f}')
            print(f'  Average trajectory points: {avg_points:.1f}')
            print()
        
    else:
        print('Error: No trajectory results generated!')
        print('Please check trajectory data format and try again')

if __name__ == '__main__':
    main()