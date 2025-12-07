#\!/usr/bin/env python3
"""
Generate synthetic trajectory data for testing geometric property analysis.
"""

import numpy as np
import pickle
import os

def generate_synthetic_trajectories(n_trajectories=10, n_points=50, n_dims=3):
    """Generate synthetic trajectory data for testing"""
    trajectories = {}
    
    for i in range(n_trajectories):
        # Generate smooth trajectory with varying curvature
        t = np.linspace(0, 4*np.pi, n_points)
        
        # Create trajectory with different characteristics
        if i % 3 == 0:
            # Spiral trajectory (high curvature)
            trajectory = np.column_stack([
                t * np.cos(t),
                t * np.sin(t), 
                t * 0.1
            ])
        elif i % 3 == 1:
            # Straight line (zero curvature)
            trajectory = np.column_stack([
                t,
                t * 0.1,
                t * 0.05
            ])
        else:
            # Wavy trajectory (medium curvature)
            trajectory = np.column_stack([
                t,
                np.sin(t),
                np.cos(t * 0.5)
            ])
        
        # Add slight noise
        trajectory += np.random.normal(0, 0.01, trajectory.shape)
        
        # Ensure we have the right number of dimensions
        if n_dims \!= 3:
            if n_dims < 3:
                trajectory = trajectory[:, :n_dims]
            else:
                # Pad with zeros for higher dimensions
                padding = np.zeros((n_points, n_dims - 3))
                trajectory = np.column_stack([trajectory, padding])
        
        trajectories[f'trajectory_{i:02d}'] = trajectory
    
    return trajectories

def main():
    """Generate and save synthetic trajectory data"""
    print('Generating synthetic trajectory data for testing...')
    
    # Generate PCA trajectories (3D)
    pca_trajectories = generate_synthetic_trajectories(n_trajectories=15, n_points=100, n_dims=3)
    
    # Generate nonlinear trajectories (2D)
    nonlinear_trajectories = generate_synthetic_trajectories(n_trajectories=15, n_points=75, n_dims=2)
    
    # Create combined data structure
    trajectory_data = {
        'pca_trajectories': pca_trajectories,
        'nonlinear_trajectories': nonlinear_trajectories,
        'trajectory_names': list(pca_trajectories.keys())
    }
    
    # Save to pickle file
    os.makedirs('results', exist_ok=True)
    with open('results/test_trajectories.pkl', 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    print(f'Saved synthetic data:')
    print(f'  PCA trajectories: {len(pca_trajectories)} (3D)')
    print(f'  Nonlinear trajectories: {len(nonlinear_trajectories)} (2D)')
    print(f'  Saved to: results/test_trajectories.pkl')

if __name__ == '__main__':
    main()
