#!/usr/bin/env python3
"""
Quick test to understand tensor shapes and dimensions in the IRED codebase.
"""

import torch
import numpy as np
from dataset import Inverse

# Create a small dataset to test shapes
dataset = Inverse(dataset_size=10, h=5, w=5, ood=False)
dataset.inp_dim = 5 * 5  # Flattened input matrix
dataset.out_dim = 5 * 5  # Flattened output inverse matrix

# Test dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

print("Testing tensor shapes in IRED dataset:")
print("=" * 50)

for batch_idx, batch_data in enumerate(dataloader):
    if batch_idx >= 2:  # Only test first few batches
        break
        
    print(f"\nBatch {batch_idx}:")
    print(f"  Number of elements in batch: {len(batch_data)}")
    
    if len(batch_data) == 2:
        inp_batch, target_batch = batch_data
        print(f"  Input batch shape: {inp_batch.shape}")
        print(f"  Target batch shape: {target_batch.shape}")
        print(f"  Input dtype: {inp_batch.dtype}")
        print(f"  Target dtype: {target_batch.dtype}")
        
        # Test individual item
        inp_single = inp_batch[0:1]
        target_single = target_batch[0:1]
        print(f"  Single input shape: {inp_single.shape}")
        print(f"  Single target shape: {target_single.shape}")
        
    break

print("\nTesting sqrt_alphas_cumprod indexing:")
print("=" * 50)

# Simulate sqrt_alphas_cumprod tensor
num_timesteps = 100
sqrt_alphas_cumprod = torch.linspace(0.1, 1.0, num_timesteps)
print(f"sqrt_alphas_cumprod shape: {sqrt_alphas_cumprod.shape}")

# Test indexing scenarios
t = 50
batched_times = torch.full((1,), t, dtype=torch.long)
print(f"batched_times shape: {batched_times.shape}")
print(f"batched_times: {batched_times}")

# Test different indexing approaches
try:
    result1 = sqrt_alphas_cumprod[batched_times]
    print(f"sqrt_alphas_cumprod[batched_times] shape: {result1.shape}")
    print(f"sqrt_alphas_cumprod[batched_times]: {result1}")
except Exception as e:
    print(f"Error with sqrt_alphas_cumprod[batched_times]: {e}")

try:
    result2 = sqrt_alphas_cumprod[batched_times][0, 0]
    print(f"sqrt_alphas_cumprod[batched_times][0, 0]: {result2}")
except Exception as e:
    print(f"Error with sqrt_alphas_cumprod[batched_times][0, 0]: {e}")

try:
    result3 = sqrt_alphas_cumprod[batched_times[0]]
    print(f"sqrt_alphas_cumprod[batched_times[0]]: {result3}")
except Exception as e:
    print(f"Error with sqrt_alphas_cumprod[batched_times[0]]: {e}")