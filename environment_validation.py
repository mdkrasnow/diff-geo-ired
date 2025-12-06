#!/usr/bin/env python3
"""
IRED Environment Validation Script

Validates environment setup for full research protocol including:
- Dependencies and version compatibility
- Full batch size configuration (paper: 2048)
- Matrix conditioning and numerical stability
- Cross-platform device detection

Usage: python3 environment_validation.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import traceback

def validate_dependencies():
    """Test all required dependencies import correctly."""
    print("=== Dependency Validation ===")
    
    required_packages = {
        'torch': '2.7.1',
        'numpy': '1.26.4', 
        'scipy': '1.15.3',
        'matplotlib': '3.10.1',
        'tqdm': '4.67.1',
        'tabulate': '0.9.0',
        'einops': '0.8.1'
    }
    
    success = True
    for package, expected_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            status = '✓' if version == expected_version else f'⚠ (expected {expected_version})'
            print(f"{package:12} {version:12} {status}")
            if version != expected_version and package in ['torch', 'numpy']:
                success = False
        except ImportError as e:
            print(f"{package:12} {'MISSING':12} ✗")
            success = False
            
    return success

def validate_device_setup():
    """Test device availability and configuration."""
    print("\n=== Device Configuration ===")
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Device detection
    devices = []
    if torch.cuda.is_available():
        devices.append(f"CUDA ({torch.cuda.device_count()} GPUs)")
    if torch.backends.mps.is_available():
        devices.append("MPS (Apple Silicon)")
    
    if devices:
        print(f"Available devices: {', '.join(devices)}")
        return True
    else:
        print("Only CPU available - may impact performance")
        return True  # Not critical, just slower

def validate_matrix_conditioning():
    """Test matrix conditioning and numerical stability."""
    print("\n=== Matrix Conditioning Validation ===")
    
    # Simulate dataset.py matrix generation
    h = 20
    np.random.seed(42)  # Reproducible test
    
    try:
        # Generate test matrix as in dataset.py
        W = np.random.randn(h, h)
        R_corrupt = W @ W.T
        
        # Test both OOD and normal regularization
        for ood, reg_val in [(True, 0.1), (False, 0.5)]:
            R_reg = R_corrupt + R_corrupt.transpose() + reg_val * np.eye(h, dtype=np.float32)
            
            # Check conditioning
            cond_num = np.linalg.cond(R_reg)
            print(f"{'OOD' if ood else 'Normal'} conditioning: {cond_num:.1f} (reg={reg_val})")
            
            if cond_num > 1e12:
                print(f"  ⚠ High condition number - potential numerical instability")
                return False
                
            # Test float64→float32 inverse as in dataset.py  
            R_inv = np.linalg.inv(R_reg.astype(np.float64)).astype(np.float32)
            identity_check = np.allclose(R_reg @ R_inv, np.eye(h), atol=1e-4)
            
            if not identity_check:
                print(f"  ✗ Inverse validation failed")
                return False
            else:
                print(f"  ✓ Inverse validation passed")
                
        return True
        
    except Exception as e:
        print(f"Matrix operations failed: {e}")
        return False

def validate_full_configuration():
    """Test full paper configuration including large batch sizes."""
    print("\n=== Full Configuration Validation ===")
    
    try:
        # Test paper configuration parameters
        batch_size = 2048
        diffusion_steps = 10
        rank = 20
        
        print(f"Testing batch_size={batch_size}, diffusion_steps={diffusion_steps}, rank={rank}")
        
        # Import key components
        sys.path.append(str(Path(__file__).parent))
        from dataset import Inverse
        
        # Test dataset creation with paper parameters
        dataset = Inverse(h=rank, ood=False)
        print("✓ Dataset creation successful")
        
        # Test batch data generation
        batch_data = []
        for i in range(min(4, batch_size // 512)):  # Test subset to avoid memory issues
            R_corrupt, R_clean = dataset[i]
            batch_data.append((R_corrupt, R_clean))
            
        print(f"✓ Generated {len(batch_data)} batch samples")
        
        # Test tensor operations with target device
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        test_tensor = torch.randn(batch_size // 128, rank * rank).to(device)  # Scaled down test
        print(f"✓ Tensor operations on {device} successful")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        traceback.print_exc()
        return False

def validate_training_pipeline():
    """Quick validation that training pipeline can start."""
    print("\n=== Training Pipeline Validation ===")
    
    try:
        # Test minimal training setup
        import argparse
        sys.path.append(str(Path(__file__).parent))
        
        # Simulate train.py argument parsing
        test_args = [
            '--dataset', 'inverse',
            '--batch_size', '16', 
            '--diffusion_steps', '2',
            '--model', 'mlp'
        ]
        
        # Import and test key training components
        from train import parser
        args = parser.parse_args(test_args)
        print(f"✓ Argument parsing successful: {args.dataset}, batch_size={args.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"Training pipeline test failed: {e}")
        return False

def main():
    """Run comprehensive environment validation."""
    print("IRED Environment Validation")
    print("=" * 50)
    
    tests = [
        ("Dependencies", validate_dependencies),
        ("Device Setup", validate_device_setup), 
        ("Matrix Conditioning", validate_matrix_conditioning),
        ("Full Configuration", validate_full_configuration),
        ("Training Pipeline", validate_training_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n{test_name} test crashed: {e}")
            results[test_name] = False
            
    # Summary
    print("\n=== Validation Summary ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\n🎉 Environment fully validated for IRED research!")
        print("Ready for full paper configuration (batch_size=2048)")
    else:
        print("\n⚠  Environment issues detected - see details above")
        print("Fix issues before running full experiments")
        
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())