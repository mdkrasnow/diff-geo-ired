# Environment Setup Documentation

## Overview
This document describes the Python virtual environment setup for the unsupervised-ired project, including all required dependencies for manifold learning, data analysis, and visualization.

## Environment Details

### Python Version
- **Version**: Python 3.11.8
- **Location**: /opt/anaconda3/bin/python3
- **Compatibility**: Meets requirement for Python 3.8+

### Virtual Environment
- **Location**: `.venv/` (project root)
- **Creation Command**: `python3 -m venv .venv`
- **Activation**: `source .venv/bin/activate`

## Installed Packages

### Core Scientific Computing
- **NumPy**: 2.3.5 - Numerical computing arrays and operations
- **SciPy**: 1.16.3 - Scientific computing algorithms and tools
- **Scikit-learn**: 1.7.2 - Machine learning and data mining

### Visualization
- **Matplotlib**: 3.10.7 - 2D plotting and visualization

### Jupyter Environment
- **Jupyter**: 1.1.1 - Interactive notebook environment
- **IPython Kernel**: 7.1.0 - Jupyter kernel support
- **Kernel Name**: `unsupervised-ired`
- **Display Name**: "Unsupervised IRED"
- **Kernel Location**: `/Users/mkrasnow/Library/Jupyter/kernels/unsupervised-ired`

### Specialized Libraries
- **PyDiffMap**: 0.2.0.1 - Diffusion maps implementation for manifold learning
  - Additional dependency: numexpr 2.14.1 for performance optimization

## Installation Process

### 1. Virtual Environment Creation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Core Package Installation
```bash
pip install numpy scipy scikit-learn matplotlib jupyter ipykernel
```

### 3. Optional Package Installation
```bash
pip install pydiffmap
```

### 4. Jupyter Kernel Registration
```bash
python -m ipykernel install --user --name unsupervised-ired --display-name "Unsupervised IRED"
```

## Verification

All packages were successfully installed and tested:
- ✅ NumPy 2.3.5 - Basic array operations
- ✅ SciPy 1.16.3 - Scientific functions
- ✅ Scikit-learn 1.7.2 - ML algorithms
- ✅ Matplotlib 3.10.7 - Plotting capabilities
- ✅ PyDiffMap - Diffusion map algorithms accessible
- ✅ Jupyter kernel registered and available

## Usage Notes

### Activating the Environment
Always activate the virtual environment before working on the project:
```bash
source .venv/bin/activate
```

### Jupyter Notebooks
The "Unsupervised IRED" kernel is available in Jupyter Lab/Notebook for interactive development with all dependencies pre-installed.

### Package Compatibility
All packages are compatible with Python 3.11.8 and work together without conflicts. The environment provides a complete foundation for:
- Manifold learning research
- Data analysis and preprocessing  
- Interactive visualization
- Jupyter-based development

## Dependencies Summary

```
numpy==2.3.5
scipy==1.16.3
scikit-learn==1.7.2
matplotlib==3.10.7
jupyter==1.1.1
ipykernel==7.1.0
pydiffmap==0.2.0.1
numexpr==2.14.1  # pydiffmap dependency
```

All additional transitive dependencies were automatically resolved and installed during the setup process.

## Installation Success
Environment setup completed successfully on December 6, 2024, with all required and optional packages functional and tested.