# Installation Guide

## Prerequisites

- **Python 3.8 or higher** (3.9+ recommended)
- **pip** (Python package manager)

Check your Python version:
```bash
python --version
```

## Installation Steps

### 1. Clone/Navigate to Project Directory

```bash
cd e:\climb-path
```

### 2. (Recommended) Create Virtual Environment

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### Linux/Mac
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

#### Option A: Install All Packages (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Install CPU-Only PyTorch (Smaller Download)
If you only have CPU (no GPU):
```bash
# Install CPU-only PyTorch first (smaller, faster)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
```

#### Option C: Install Minimal Dependencies
If you only want to train the model (no API):
```bash
pip install torch pandas numpy scikit-learn tensorboard tqdm transformers
```

### 4. Verify Installation

Run the test script:
```bash
python test_model.py
```

You should see:
```
======================================================================
TESTING AUTOREGRESSIVE CLIMB PATH MODEL
======================================================================

1. Testing imports...
   ✓ All imports successful

2. Testing tokenizer...
   ✓ Tokenizer initialized (vocab_size=45)

...

ALL TESTS PASSED! ✓
```

## Installation Time Estimates

| Method | Download Size | Install Time |
|--------|---------------|--------------|
| **Full install** | ~2-3 GB | 5-10 minutes |
| **CPU-only PyTorch** | ~150 MB | 2-5 minutes |
| **Minimal** | ~200 MB | 2-5 minutes |

## Troubleshooting

### "pip: command not found"
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

### "Permission denied"
```bash
# Use --user flag
pip install --user -r requirements.txt
```

### "No matching distribution found for torch>=2.1.0"
Your Python version might be too old. Upgrade to Python 3.8+:
```bash
python --version  # Check current version
```

### PyTorch Installation Issues

If PyTorch fails to install, visit https://pytorch.org/get-started/locally/ and select:
- **PyTorch Build**: Stable
- **Your OS**: Windows/Linux/Mac
- **Package**: Pip
- **Language**: Python
- **Compute Platform**: CPU (or CUDA if you have GPU)

Then run the generated command, for example:
```bash
# CPU only (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### "Out of disk space"
PyTorch is large (~2GB). Free up space or use CPU-only version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Package Descriptions

| Package | Purpose | Size |
|---------|---------|------|
| **torch** | Deep learning framework | ~2 GB |
| **transformers** | Transformer utilities | ~500 MB |
| **pandas** | Data manipulation | ~50 MB |
| **numpy** | Numerical computing | ~20 MB |
| **tensorboard** | Training visualization | ~100 MB |
| **tqdm** | Progress bars | ~1 MB |
| **scikit-learn** | Train/val/test splits | ~30 MB |
| **matplotlib/seaborn** | Plotting (optional) | ~50 MB |
| **fastapi/uvicorn** | API (optional) | ~20 MB |

## Next Steps

After successful installation:

1. **Verify data is ready**:
   ```bash
   # Check if data files exist
   dir data\moonboard_*.csv  # Windows
   ls data/moonboard_*.csv   # Linux/Mac
   ```

2. **Run quick test**:
   ```bash
   python test_model.py
   ```

3. **Start training**:
   ```bash
   # CPU training
   train_cpu.bat  # Windows
   bash train_cpu.sh  # Linux/Mac
   ```

4. **Monitor progress**:
   ```bash
   tensorboard --logdir runs/climb_path_cpu
   ```

## Updating Dependencies

To update all packages to latest versions:
```bash
pip install --upgrade -r requirements.txt
```

## Uninstalling

To remove the virtual environment:
```bash
# Deactivate first
deactivate

# Remove directory
rmdir /s venv  # Windows
rm -rf venv    # Linux/Mac
```

## Alternative: Conda Installation

If you use Anaconda/Miniconda:
```bash
# Create conda environment
conda create -n climb-path python=3.9

# Activate environment
conda activate climb-path

# Install PyTorch
conda install pytorch cpuonly -c pytorch

# Install other packages
pip install -r requirements.txt
```

---

**You're ready to train!** See `QUICKSTART.md` or `CPU_TRAINING_GUIDE.md` for next steps.
