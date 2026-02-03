# Climb Path Generation Workflow

## Core Pipeline

### 1. **Data Quality Filtering**
```bash
python check_data_quality.py
```
- Filters MoonBoard data by quality score
- Creates `moonboard_train_quality.csv` and `moonboard_test_quality.csv`
- Removes low-quality paths

### 2. **Reorder with Y-Axis**
```bash
python reorder_all_datasets.py
```
- Reorders holds by Y-coordinate (bottom to top)
- Creates files in `data_reordered/` folder
- Essential for proper path representation

### 3. **Check Grade Balance**
```bash
python analyze_grade_distribution.py
```
- Analyzes grade distribution
- Creates `moonboard_train_quality_filtered.csv` (removes rare grades)
- Ensures balanced training data

### 4. **Fine-tune DistilGPT-2**
```bash
.\train_distilgpt2.bat
```
- Trains DistilGPT-2 on climbing paths
- Uses endpoint conditioning (grade + start/end → intermediate holds)
- Saves checkpoints to `checkpoints/distilgpt2_climb/`
- **Time**: ~2-3 hours on CPU, ~20 minutes on GPU

### 5. **Generate Paths**
```bash
.\generate_paths.bat
```
- Interactive path generation
- Specify grade, start hold, end hold
- Uses trained model from step 4

### 6. **Gradio Web Server**
```bash
.\start_web_interface.bat
```
- Full web interface with visualization
- Generate and visualize paths
- Interactive MoonBoard display

---

## Essential Files

### Scripts
- `check_data_quality.py` - Quality filtering
- `reorder_all_datasets.py` - Y-axis reordering
- `analyze_grade_distribution.py` - Grade balance analysis
- `visualize_path.py` - Path visualization
- `web_interface.py` - Gradio web interface

### Batch Files
- `train_distilgpt2.bat` - Training
- `generate_paths.bat` - Generation
- `start_web_interface.bat` - Web server
- `start_server.bat` - API server
- `start_frontend.bat` - Frontend server

### Training Code
- `backend/training/finetune_huggingface.py` - DistilGPT-2 fine-tuning
- `backend/training/generate_with_gpt2.py` - Path generation

### Datasets (data_reordered/)
- `moonboard_train_quality_filtered.csv` - Training data (balanced)
- `moonboard_test_quality.csv` - Test data
- `moonboard_val_quality.csv` - Validation data
- `moonboard_train_quality.csv` - Original quality-filtered training data

### Documentation
- `README.md` - Project overview
- `QUICKSTART.md` - Quick start guide
- `INSTALL.md` - Installation instructions
- `WINDOWS_SETUP.md` - Windows-specific setup
- `ENDPOINT_CONDITIONING.md` - Endpoint conditioning explanation
- `GRADE_BALANCE_GUIDE.md` - Grade balance guide
- `QUALITY_FILTERING_GUIDE.md` - Quality filtering guide
- `WORKFLOW.md` - This file

---

## Quick Start

```bash
# 1. Check data quality
python check_data_quality.py

# 2. Reorder datasets
python reorder_all_datasets.py

# 3. Check grade balance
python analyze_grade_distribution.py

# 4. Train model
.\train_distilgpt2.bat

# 5. Generate paths
.\generate_paths.bat

# 6. Start web interface
.\start_web_interface.bat
```

---

## File Structure

```
climb-path/
├── backend/
│   ├── training/
│   │   ├── finetune_huggingface.py    # DistilGPT-2 training
│   │   └── generate_with_gpt2.py      # Path generation
│   ├── models/                         # Model definitions
│   ├── data_processing/                # Data processing
│   └── api/                            # API server
├── data_reordered/
│   ├── moonboard_train_quality_filtered.csv  # Training (balanced)
│   ├── moonboard_test_quality.csv            # Testing
│   └── moonboard_val_quality.csv             # Validation
├── checkpoints/
│   └── distilgpt2_climb/              # Trained models
├── frontend/                           # React frontend
├── check_data_quality.py              # Quality filtering
├── reorder_all_datasets.py            # Y-axis reordering
├── analyze_grade_distribution.py      # Grade analysis
├── visualize_path.py                  # Visualization
├── web_interface.py                   # Gradio interface
├── train_distilgpt2.bat              # Training script
├── generate_paths.bat                 # Generation script
└── start_web_interface.bat           # Web server
```
