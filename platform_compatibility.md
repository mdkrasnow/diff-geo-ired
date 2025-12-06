# IRED Platform Compatibility Guide

## Tested Platforms

### ✅ Fully Tested
- **macOS 14.6.0 (Darwin 24.6.0) - Apple Silicon**
  - Python 3.11.8
  - PyTorch 2.7.1 with MPS backend
  - All dependencies working
  - Batch sizes: 16 (tested), 2048 (paper config)
  - Device: MPS acceleration available

### 🟨 Expected Compatible (Validation Needed)
- **Ubuntu 20.04+ / Linux x86_64**
  - Python 3.8+
  - PyTorch 2.7.1 with CUDA 11.8+ or CPU
  - All dependencies available via pip
  - Device: CUDA recommended, CPU fallback
  
- **Windows 10+ with WSL2**
  - Python 3.8+ in WSL2 Ubuntu environment
  - PyTorch with CUDA support
  - Native Windows support not tested

- **Google Colab / Kaggle**
  - Python 3.10+
  - Pre-installed PyTorch with CUDA
  - May require requirements.txt installation

## Device-Specific Notes

### Apple Silicon (M1/M2/M3)
- ✅ MPS backend provides GPU acceleration
- ✅ Float32 precision works correctly
- ✅ Matrix operations numerically stable
- ⚠️ Large batch sizes (2048+) may require memory management

### NVIDIA CUDA
- Expected to work with PyTorch CUDA build
- Float16 mixed precision may be available
- Large batch sizes should work better than MPS
- Requires CUDA 11.8+ for PyTorch 2.7.1

### CPU-Only
- ✅ All operations work on CPU
- ⚠️ Significantly slower training
- ⚠️ Large batch sizes may require reduced size
- NumPy operations are single-threaded (OPENBLAS_NUM_THREADS=1)

## Cross-Platform Validation Protocol

To validate IRED on new platforms:

### Step 1: Environment Setup
```bash
# Install requirements
pip install -r requirements.txt

# Run validation script
python3 environment_validation.py
```

### Step 2: Device Testing
```bash
# Test device detection
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"

# Test basic tensor operations
python3 -c "import torch; x = torch.randn(100, 100); print('Device test:', x.device, x.sum().item())"
```

### Step 3: Matrix Operations
```bash
# Test numerical stability
python3 -c "from dataset import Inverse; d = Inverse(h=20); print('Matrix test:', d[0][0].shape)"
```

### Step 4: Training Pipeline
```bash
# Test minimal training
python3 train.py --dataset inverse --batch_size 16 --diffusion_steps 2 --model mlp
```

### Step 5: Paper Configuration
```bash
# Test full paper config (if resources allow)
python3 train.py --dataset inverse --batch_size 2048 --diffusion_steps 10 --model mlp
```

## Known Issues and Workarounds

### Memory Issues
- **Large batch sizes**: Reduce batch_size if OOM errors
- **Apple Silicon**: Use batch_size ≤ 1024 for safety
- **CPU-only**: Use batch_size ≤ 512

### Dependency Issues
- **Missing packages**: Install via `pip install -r requirements.txt`
- **Version conflicts**: Use virtual environment
- **MKL warnings**: Cosmetic only, can be ignored

### Performance Issues
- **Slow training**: Ensure GPU acceleration is working
- **Memory warnings**: Monitor system memory usage
- **Numerical instability**: Check matrix conditioning validation

## Reporting Platform Issues

When reporting platform compatibility issues, include:

1. **System Info**:
   ```bash
   python3 -c "import sys, platform; print(f'Python: {sys.version}'); print(f'Platform: {platform.platform()}')"
   ```

2. **PyTorch Info**:
   ```bash
   python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Devices: CUDA={torch.cuda.is_available()}, MPS={torch.backends.mps.is_available()}')"
   ```

3. **Error Details**: Full error traceback and environment_validation.py output

4. **Configuration**: batch_size, dataset, model used when error occurred

## Future Platform Support

- **AMD ROCm**: Planned testing with PyTorch ROCm builds
- **Intel XPU**: Testing when PyTorch Intel GPU support matures
- **Cloud Platforms**: Systematic testing on AWS, GCP, Azure ML
- **ARM64 Linux**: Testing on ARM64 Linux servers

This guide will be updated as new platforms are tested and validated.