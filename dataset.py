import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import multiprocessing as mp
import pickle
from multiprocessing import Pool
import traceback
import gc
import argparse
import time
import sys

try:
    from sat_dataset import Dataset as CustomDataset
except ImportError:
    from torch.utils.data import Dataset as CustomDataset


class Addition(data.Dataset):
    def __init__(self, digit=8, transform=None, batch_size=1, seed=42):
        self.dataset = None
        self.dataset_size = int(1e5)
        self.digit = digit
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed

    def gen_single_data(self):
        # Randomly generate a two numbers
        np.random.seed()
        a = np.random.randint(0, 10**(self.digit-1), 1)[0]
        b = np.random.randint(0, 10**(self.digit-1), 1)[0]
        sum_a_b = a + b

        x = np.array([a, b], dtype=np.float32)
        y = np.array(sum_a_b, dtype=np.float32)

        return x, y

    def __getitem__(self, index):
        x, y = self.gen_single_data()
        return x, y

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size


class LowRankDataset(data.Dataset):
    def __init__(self, dataset_size=1000, rank=10, h=100, ood=False):
        """
        Generate dataset of rank-h matrix completion problems

        :param dataset_size: Number of examples to generate
        :param rank: Rank of the ground truth matrix
        :param h: Dimension of the matrix (hxh)
        """
        self.dataset_size = dataset_size
        self.rank = rank
        self.h = h
        self.ood = ood

    def __getitem__(self, index):
        """
        Generate problem instance of x -> A such that AxA^T = x

        Returns:
        x - the full matrix of size h x h
        A_paths(str) - - the path of the image
        """

        h_rank = self.rank

        if self.ood:
            # matrix might have higher rank
            h_rank = self.rank + int(np.random.uniform(0, self.rank, 1)[0])
            h_rank = min(h_rank, self.h)

        A = np.random.uniform(-1, 1, (self.h, h_rank)).astype(np.float32)
        R_corrupt = A.dot(A.transpose())

        # add noise to the corrupted matrix
        R_corrupt = R_corrupt + np.random.uniform(-0.01, 0.01, (self.h, self.h)).astype(np.float32)

        return R_corrupt.flatten(), A.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size


class Inverse(data.Dataset):
    def __init__(self, dataset_size=1000, h=100, w=100, ood=False, log_trajectories=False):
        """
        Generate dataset of rank-h matrix completion problems

        :param dataset_size: Number of examples to generate
        :param h: Dimension of the matrix (hxh)
        :param log_trajectories: Enable trajectory logging for IRED research pipeline
        """
        self.dataset_size = dataset_size
        self.h = h
        self.w = w if w != h else h
        self.ood = ood
        self.log_trajectories = log_trajectories
        
        # Initialize trajectory logging storage
        if self.log_trajectories:
            self.trajectory_log = {
                'matrix_states': [],
                'conditioning_info': [],
                'inversion_steps': [],
                'timestamps': []
            }

    def __getitem__(self, index):
        """
        Generate problem instance of x -> A^-1 such that x is the input matrix and A^-1 is the inverse

        Returns:
        x - the full matrix of size h x h
        A_paths(str) - - the path of the image
        """
        import time
        
        # Log trajectory start time if logging enabled
        start_time = time.time() if self.log_trajectories else None

        R_corrupt = np.random.uniform(-1, 1, (self.h, self.w)).astype(np.float32)
        R_corrupt = R_corrupt.dot(R_corrupt.transpose())
        # R_corrupt = R_corrupt + 0.5 * np.eye(self.h)

        # Matrix conditioning with validation
        if self.ood:
            regularization = 0.1
            R_corrupt = R_corrupt + R_corrupt.transpose() + regularization * np.eye(self.h, dtype=np.float32)
        else:
            regularization = 0.5
            R_corrupt = R_corrupt + R_corrupt.transpose() + regularization * np.eye(self.h, dtype=np.float32)
        
        # Validate matrix conditioning before inversion
        condition_number = np.linalg.cond(R_corrupt)
        extra_reg = 0
        if condition_number > 1e12:
            # Add extra regularization for ill-conditioned matrices
            extra_reg = 1e-6
            R_corrupt = R_corrupt + extra_reg * np.eye(self.h, dtype=np.float32)
            condition_number = np.linalg.cond(R_corrupt)
            if condition_number > 1e12:
                raise ValueError(f"Matrix ill-conditioned: cond={condition_number:.2e}, regularization failed")
        
        # Log matrix state and conditioning info if enabled
        if self.log_trajectories:
            self.trajectory_log['matrix_states'].append({
                'index': index,
                'initial_matrix': R_corrupt.copy(),
                'matrix_shape': (self.h, self.w),
                'ood_mode': self.ood
            })
            self.trajectory_log['conditioning_info'].append({
                'condition_number': float(condition_number),
                'regularization': float(regularization),
                'extra_regularization': float(extra_reg)
            })
        
        # Compute inverse with float64 precision, validate result
        try:
            R_float64 = R_corrupt.astype(np.float64)
            R_inv_float64 = np.linalg.inv(R_float64)
            
            # Validate inverse accuracy in float64
            identity_check = np.allclose(R_float64 @ R_inv_float64, np.eye(self.h), atol=1e-12)
            if not identity_check:
                raise ValueError("Matrix inverse validation failed in float64")
                
            # Convert to float32 and validate precision loss is acceptable
            R = R_inv_float64.astype(np.float32)
            identity_check_f32 = np.allclose(R_corrupt @ R, np.eye(self.h), atol=1e-4)
            if not identity_check_f32:
                raise ValueError("Precision loss in float64→float32 conversion too large")
                
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix inversion failed: {e}")

        # Log inversion steps and timing if enabled
        if self.log_trajectories:
            end_time = time.time()
            self.trajectory_log['inversion_steps'].append({
                'float64_precision_used': True,
                'float32_validation_passed': identity_check_f32,
                'float64_validation_passed': identity_check,
                'inverse_matrix': R.copy()
            })
            self.trajectory_log['timestamps'].append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })

        return R_corrupt.flatten(), R.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e7)
    
    def get_trajectory_log(self):
        """
        Retrieve the complete trajectory log for IRED research analysis.
        
        Returns:
            dict: Complete trajectory log with matrix states, conditioning info, 
                  inversion steps, and timestamps
        """
        if not self.log_trajectories:
            raise ValueError("Trajectory logging not enabled. Set log_trajectories=True in constructor.")
        return self.trajectory_log
    
    def get_trajectory_summary(self):
        """
        Get summary statistics of logged trajectories for IRED research pipeline.
        
        Returns:
            dict: Summary statistics including trajectory counts, timing info, 
                  conditioning statistics
        """
        if not self.log_trajectories:
            raise ValueError("Trajectory logging not enabled. Set log_trajectories=True in constructor.")
        
        if not self.trajectory_log['timestamps']:
            return {"summary": "No trajectories logged yet"}
        
        durations = [t['duration'] for t in self.trajectory_log['timestamps']]
        condition_numbers = [c['condition_number'] for c in self.trajectory_log['conditioning_info']]
        
        return {
            'total_trajectories': len(self.trajectory_log['timestamps']),
            'timing_stats': {
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations)
            },
            'conditioning_stats': {
                'mean_condition_number': np.mean(condition_numbers),
                'std_condition_number': np.std(condition_numbers),
                'min_condition_number': np.min(condition_numbers),
                'max_condition_number': np.max(condition_numbers)
            },
            'matrix_dimensions': (self.h, self.w),
            'ood_mode': self.ood
        }
    
    def clear_trajectory_log(self):
        """
        Clear all logged trajectories to free memory.
        Useful for long-running IRED experiments.
        """
        if not self.log_trajectories:
            raise ValueError("Trajectory logging not enabled. Set log_trajectories=True in constructor.")
        
        self.trajectory_log = {
            'matrix_states': [],
            'conditioning_info': [],
            'inversion_steps': [],
            'timestamps': []
        }
    
    def save_trajectory_log(self, filepath):
        """
        Save trajectory log to file for IRED research analysis.
        
        Args:
            filepath (str): Path to save the trajectory log
        """
        if not self.log_trajectories:
            raise ValueError("Trajectory logging not enabled. Set log_trajectories=True in constructor.")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectory_log, f)


class Equation(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, dataset_size=1000, dim=10, num_circles=3, ood=False):
        """
        :param dataset_size: Number of examples to generate
        :param dim: Dimension of the input
        :param num_circles: Number of circle components to place
        """
        self.dataset_size = dataset_size
        self.dim = dim
        self.num_circles = num_circles
        self.ood = ood

    def __getitem__(self, index):
        """
        Generate problem instance of a trajectory optimization problem
        x -> path that passes through all circles

        Returns:
        circles - locations and radius of each circle [num_circles, 3] (x, y, r)
        path - trajectory that passes through all circles [path_length, 2] (x, y)
        """

        # Generate random circle positions
        circles = np.random.uniform(-5, 5, (self.num_circles, 3)).astype(np.float32)
        circles[:, 2] = np.abs(circles[:, 2]) + 0.1  # ensure positive radius

        if self.ood:
            # In OOD mode, make problem harder with smaller circles
            circles[:, 2] *= 0.5

        # Generate a path that visits all circles
        # Start from first circle center
        path = []
        current_pos = circles[0, :2]
        path.append(current_pos.copy())

        # Move through each circle
        for i in range(1, self.num_circles):
            target = circles[i, :2]
            direction = target - current_pos
            steps = int(np.linalg.norm(direction) * 5) + 5  # adaptive step count
            
            for j in range(steps):
                alpha = (j + 1) / steps
                pos = current_pos + alpha * direction
                path.append(pos.copy())
            
            current_pos = target

        path = np.array(path, dtype=np.float32)
        
        return circles.flatten(), path.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size


class Tree(data.Dataset):
    """Constructs a tree dataset with N nodes"""

    def __init__(self, dataset_size=1000, num_nodes=10, ood=False):
        """
        :param dataset_size: Number of examples to generate
        :param num_nodes: Number of nodes in the tree
        """
        self.dataset_size = dataset_size
        self.num_nodes = num_nodes
        self.ood = ood

    def generate_tree(self):
        """Generate a random tree structure"""
        # Create adjacency matrix
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        # Ensure it's a tree (connected, no cycles)
        nodes = list(range(self.num_nodes))
        connected = [0]  # Start with node 0
        unconnected = nodes[1:]
        
        while unconnected:
            # Pick random connected node and random unconnected node
            parent = np.random.choice(connected)
            child = np.random.choice(unconnected)
            
            # Add edge
            adj_matrix[parent, child] = 1
            adj_matrix[child, parent] = 1
            
            # Update sets
            connected.append(child)
            unconnected.remove(child)
        
        return adj_matrix

    def __getitem__(self, index):
        """
        Generate problem instance of finding shortest path in tree

        Returns:
        adj_matrix - adjacency matrix of the tree
        shortest_paths - shortest path distances between all pairs
        """

        adj_matrix = self.generate_tree()
        
        # Compute shortest paths using Floyd-Warshall
        dist_matrix = adj_matrix.copy()
        
        # Initialize distances
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    dist_matrix[i, j] = 0
                elif adj_matrix[i, j] == 0:
                    dist_matrix[i, j] = float('inf')
        
        # Floyd-Warshall algorithm
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        
        # Replace inf with large number for numerical stability
        dist_matrix[dist_matrix == float('inf')] = 999
        
        if self.ood:
            # In OOD mode, add some noise to distances
            noise = np.random.uniform(-0.1, 0.1, dist_matrix.shape).astype(np.float32)
            dist_matrix = dist_matrix + noise
            dist_matrix = np.maximum(dist_matrix, 0)  # ensure non-negative

        return adj_matrix.flatten(), dist_matrix.flatten()

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size