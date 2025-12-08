#!/usr/bin/env python3
"""
Create schematic figures for the IRED geometric analysis communication paper.
Author: Matt Krasnow, Dec 7 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style for academic publication
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 150

def create_pca_trajectories_schematic():
    """Create schematic showing PCA trajectory analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Generate synthetic trajectory data for visualization
    np.random.seed(42)
    n_trajectories = 12
    n_points = 11
    
    # Create trajectories that start dispersed and converge
    start_x = np.random.normal(0, 1.5, n_trajectories)
    start_y = np.random.normal(0, 1.5, n_trajectories)
    
    # Convergence point
    end_x, end_y = 3.5, 2.0
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        # Create smooth trajectory from start to convergence
        t = np.linspace(0, 1, n_points)
        x_traj = start_x[i] + (end_x - start_x[i]) * t**1.5
        y_traj = start_y[i] + (end_y - start_y[i]) * t**1.5
        
        # Add some noise for realism
        x_traj += 0.1 * np.random.normal(0, 1, n_points) * (1 - t)
        y_traj += 0.1 * np.random.normal(0, 1, n_points) * (1 - t)
        
        # Plot trajectory
        ax.plot(x_traj, y_traj, color=colors[i], alpha=0.8, linewidth=2)
        
        # Mark start and end points
        ax.scatter(x_traj[0], y_traj[0], color=colors[i], s=30, alpha=0.6)
        ax.scatter(x_traj[-1], y_traj[-1], color=colors[i], s=50, marker='*')
    
    # Add regions
    initialization_circle = plt.Circle((0, 0), 2.5, fill=False, 
                                     linestyle='--', alpha=0.5, color='red')
    ax.add_patch(initialization_circle)
    
    convergence_circle = plt.Circle((end_x, end_y), 0.8, fill=False,
                                  linestyle='--', alpha=0.5, color='green')
    ax.add_patch(convergence_circle)
    
    # Labels and annotations
    ax.text(0, -3.2, 'Initialization\nRegion', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(end_x, end_y-1.8, 'Convergence\nRegion', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlabel('First Principal Component (61.7% variance)')
    ax.set_ylabel('Second Principal Component (27.6% variance)')
    ax.set_title('PCA Trajectory Analysis: Matrix Inverse Optimization')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 5)
    ax.set_ylim(-4, 4)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.5, label='Initialization region'),
        mpatches.Patch(color='green', alpha=0.5, label='Convergence region'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Optimization trajectory')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/pca_trajectories_schematic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created: pca_trajectories_schematic.png")

def create_manifold_comparison_schematic():
    """Create schematic comparing PCA and Isomap manifold embeddings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Common trajectory data
    np.random.seed(123)
    n_traj = 8
    n_points = 11
    
    # Generate trajectories for PCA (more linear)
    for i in range(n_traj):
        t = np.linspace(0, 1, n_points)
        # Linear-like progression
        x_pca = -2 + 4*t + 0.3*np.random.normal(0, 1, n_points) * (1-t)
        y_pca = -1 + 2*t + 0.2*np.random.normal(0, 1, n_points) * (1-t)
        
        color = plt.cm.plasma(i / n_traj)
        ax1.plot(x_pca, y_pca, color=color, alpha=0.8, linewidth=2)
        ax1.scatter(x_pca[0], y_pca[0], color=color, s=25)
        ax1.scatter(x_pca[-1], y_pca[-1], color=color, s=40, marker='*')
    
    # Generate trajectories for Isomap (more curved)
    for i in range(n_traj):
        t = np.linspace(0, 1, n_points)
        # Curved progression
        angle_start = -np.pi/3 + i * np.pi/12
        radius = 2 + 0.3*i
        x_iso = radius * np.cos(angle_start + t * np.pi/2)
        y_iso = radius * np.sin(angle_start + t * np.pi/2)
        
        # Add noise
        x_iso += 0.2*np.random.normal(0, 1, n_points) * (1-t)
        y_iso += 0.2*np.random.normal(0, 1, n_points) * (1-t)
        
        color = plt.cm.plasma(i / n_traj)
        ax2.plot(x_iso, y_iso, color=color, alpha=0.8, linewidth=2)
        ax2.scatter(x_iso[0], y_iso[0], color=color, s=25)
        ax2.scatter(x_iso[-1], y_iso[-1], color=color, s=40, marker='*')
    
    # Format axes
    ax1.set_title('PCA Embedding\n(Linear, Extrinsic)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    ax2.set_title('Isomap Embedding\n(Nonlinear, Intrinsic)')
    ax2.set_xlabel('Isomap Dim 1')
    ax2.set_ylabel('Isomap Dim 2')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add text annotations
    ax1.text(0.02, 0.98, 'Path Length: 2.847\nSinuosity: 1.421', 
             transform=ax1.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.text(0.02, 0.98, 'Path Length: 3.214\nSinuosity: 1.612', 
             transform=ax2.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/manifold_comparison_schematic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created: manifold_comparison_schematic.png")

if __name__ == "__main__":
    create_pca_trajectories_schematic()
    create_manifold_comparison_schematic()
    print("All figures created successfully!")