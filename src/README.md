# Source Directory - Running Guide

This directory contains all the source code, datasets, and implementation details for the ML_Studies repository. Follow this guide to set up your environment and run the machine learning models effectively.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Running the Models](#running-the-models)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

## Quick Start

```bash
# 1. Navigate to the src directory
cd src

# 2. Create the virtual environment (if not already created)
python -m venv ml_venv

# 3. Activate the virtual environment (if not already active)
source ml_venv/bin/activate # Linux/macOS
# OR
ml_venv\Scripts\activate    # Windows

# 4. Install dependencies (if not already installed)
pip install -r requirements.txt

# 5. Launch Jupyter Notebook
jupyter notebook

# 6. Navigate to any algorithm folder and open the .ipynb file
```

## Environment Setup

### Option 1: Creating Your Own Virtual Environment

If you prefer to create a fresh environment:

```bash
# Create new virtual environment
python -m venv my_ml_env

# Activate it
source my_ml_env/bin/activate  # Linux/macOS
my_ml_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

If you prefer Conda environments:

```bash
# Create conda environment
conda create -n ml_studies python=3.9

# Activate environment
conda activate ml_studies

# Install dependencies
pip install -r requirements.txt
# OR install conda packages:
conda install jupyter pandas numpy scikit-learn matplotlib seaborn
conda install tensorflow imbalanced-learn -c conda-forge
```

## Dependencies

### Core Requirements (from requirements.txt)

```txt
imbalanced-learn>=0.8.0 # Handling imbalanced datasets
ipykernel>=6.0.0        # Jupyter kernel support
matplotlib>=3.4.0       # Plotting and visualization
numpy>=1.21.0           # Numerical computing
pandas>=1.3.0           # Data manipulation and analysis
scikit-learn>=1.0.0     # Machine learning algorithms
seaborn>=0.11.0         # Statistical data visualization
tensorflow>=2.10.0      # Deep learning (Neural Networks only)
```

### System Requirements

- **Python**: 3.7+ (recommended: 3.9+)
- **Memory**: 4GB+ RAM (8GB+ recommended for larger datasets)
- **Storage**: ~2GB for dependencies and datasets
- **OS**: Windows, macOS, or Linux

### Installing Dependencies

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install individually for more control
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn ipykernel
pip install tensorflow  # Only needed for Neural Networks

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages imported successfully!')"
```

## Running the Models

### Method 1: Jupyter Notebook (Recommended)

This is the primary way to run and interact with the implementations:

```bash
# Start Jupyter Notebook server
jupyter notebook

# Your browser should automatically open to http://localhost:8888
# Navigate to the algorithms/ folder
# Click on any algorithm folder (e.g., knn/)
# Open the .ipynb file (e.g., knn.ipynb)
# Run cells sequentially using Shift+Enter
```

### Method 2: JupyterLab (Alternative Interface)

If you prefer JupyterLab's interface:

```bash
# Install JupyterLab (if not already installed)
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Navigate to algorithms/ and open any .ipynb file
```

### Method 3: VS Code with Jupyter Extension

If you're using VS Code:

1. Install the Jupyter extension for VS Code
2. Open the src folder in VS Code
3. Select the correct Python interpreter (ml_venv/bin/python)
4. Open any .ipynb file and run cells interactively

### Method 4: Google Colab (Cloud-based)

Upload individual notebooks to Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the .ipynb file you want to run
3. Upload the corresponding dataset files to Colab
4. Modify file paths in the notebook to match Colab structure
5. Install dependencies: `!pip install scikit-learn imbalanced-learn`

## Directory Structure

```
src/
├── README.md                    # This file - detailed running instructions
├── requirements.txt             # Python dependencies
├── ml_venv/                     # Python virtual environment
│   ├── bin/ (or Scripts/ on Windows)
│   ├── lib/
│   └── pyvenv.cfg
├── algorithms/                  # Algorithm implementations
│   └── ...
└── datasets/                    # Curated datasets for experiments
    └── ...
```

## Troubleshooting

### Common Issues and Solutions

#### **Issue 1: "ModuleNotFoundError" when importing packages**

```bash
# Solution: Ensure virtual environment is activated and dependencies installed
source ml_venv/bin/activate  # Activate environment
pip install -r requirements.txt  # Install dependencies
```

#### **Issue 2: Jupyter can't find the correct Python kernel**

```bash
# Solution: Register your environment as a Jupyter kernel
python -m ipykernel install --user --name=ml_studies --display-name="ML Studies"
# Then select "ML Studies" kernel in Jupyter
```

#### **Issue 3: TensorFlow installation issues (Neural Networks)**

```bash
# For CPU-only installation
pip install tensorflow-cpu

# For Apple Silicon Macs
pip install tensorflow-macos

# For older systems, use specific version
pip install tensorflow==2.10.0
```

#### **Issue 4: "Permission denied" when activating virtual environment**

```bash
# On Linux/macOS, make activation script executable
chmod +x ml_venv/bin/activate
```

#### **Issue 5: Jupyter notebook won't start**

```bash
# Check if Jupyter is installed in the correct environment
which jupyter
# If not found, install it
pip install jupyter notebook

# Or try launching with full path
python -m jupyter notebook
```

#### **Issue 6: Out of memory errors**

- **For large datasets**: Consider using a subset for initial testing
- **Neural Networks**: Reduce batch size in the training configuration
- **General**: Close other applications to free up RAM

### Dataset-Specific Issues

#### **Missing Dataset Files**
All datasets are included in the repository. If files are missing:

```bash
# Check if you're in the correct directory
pwd  # Should show path ending in /src
ls datasets/  # Should show all dataset folders

# If datasets are missing, re-clone the repository
git clone https://github.com/gabrieldelmonte/ML_Studies.git
```

## Performance Tips

### Faster Execution
- **Close unnecessary applications** to free up RAM
- **Use smaller dataset samples** for initial testing and learning

### Better Jupyter Experience
```bash
# Install useful Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable variable inspector
jupyter nbextension enable varInspector/main
```

### Resource Monitoring
```python
# Monitor memory usage in notebooks
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Time execution of cells
%%time
# Your code here
```

### Optimization for Different Use Cases

#### **Learning/Education**
- Start with smaller datasets (Car Evaluation, Tic-Tac-Toe)
- Focus on understanding code rather than performance
- Run one algorithm at a time

#### **Research/Comparison**
- Use consistent random seeds across algorithms
- Run full datasets for accurate comparisons
- Save results for later analysis

#### **Development/Extension**
- Create copies of notebooks before modifying
- Use version control for tracking changes
- Test with smaller datasets before full runs

---

*For algorithm-specific details, mathematical foundations, and detailed explanations, check the README.md file in each algorithm folder.*
