# Windows Setup Guide

## Quick Fix for PowerShell Issues

You're seeing two common Windows issues:
1. PowerShell script execution policy blocking venv activation
2. Packages not installed yet

## Solution: Use Command Prompt Instead (Easiest)

### Option A: Command Prompt (Recommended for Windows)

1. **Open Command Prompt** (not PowerShell):
   - Press `Win + R`
   - Type `cmd`
   - Press Enter

2. **Navigate to project:**
   ```cmd
   cd E:\climb-path
   ```

3. **Create virtual environment:**
   ```cmd
   python -m venv venv
   ```

4. **Activate virtual environment:**
   ```cmd
   venv\Scripts\activate.bat
   ```
   You should see `(venv)` in your prompt.

5. **Install packages:**
   ```cmd
   pip install -r requirements.txt
   ```

6. **Run training:**
   ```cmd
   .\train_cpu.bat
   ```

### Option B: Fix PowerShell (If You Prefer PowerShell)

1. **Enable script execution** (run PowerShell as Administrator):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Then activate venv:**
   ```powershell
   venv\Scripts\Activate.ps1
   ```

3. **Install packages:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run training:**
   ```powershell
   .\train_cpu.bat
   ```

### Option C: Skip Virtual Environment (Quick & Dirty)

If you just want to get started quickly:

```cmd
# Install directly to system Python
pip install -r requirements.txt

# Run training
.\train_cpu.bat
```

**Warning**: This installs packages globally. Virtual environment is better practice.

## Step-by-Step for Command Prompt

```cmd
REM 1. Navigate to project
cd E:\climb-path

REM 2. Create virtual environment
python -m venv venv

REM 3. Activate it
venv\Scripts\activate.bat

REM 4. Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

REM 5. Install packages
pip install -r requirements.txt

REM 6. Verify installation
python test_model.py

REM 7. Start training
.\train_cpu.bat
```

## Troubleshooting

### "python: command not found"
Python not in PATH. Try:
```cmd
py -m venv venv
py -m pip install -r requirements.txt
```

### "pip: command not found"
Use:
```cmd
python -m pip install -r requirements.txt
```

### Still having issues?
Install packages without virtual environment:
```cmd
pip install torch pandas numpy scikit-learn tensorboard tqdm transformers --user
```

Then run:
```cmd
python backend\training\train_autoregressive.py --device cpu --batch_size 16 --num_workers 0
```
