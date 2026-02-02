# Backend

Machine learning backend for climb path generation.

## Directory Structure

- **data_processing/** - Scripts for data cleaning, analysis, and preparation
- **models/** - Neural network architectures for path generation
- **training/** - Training scripts and utilities
- **api/** - REST API endpoints for frontend integration

## Data Processing Scripts

### `analyze_data.py`
Analyzes the MoonBoard dataset and provides statistics on grades, holds, and problem distribution.

### `clean_data_for_training.py`
Cleans and prepares data for model training:
- Parses hold coordinates
- Validates data quality
- Creates train/val/test splits
- Exports in multiple formats (CSV, JSON, JSONL)

### `filter_quality_data.py`
Filters dataset to include only high-quality routes (benchmark status, high repeats).

### `analyze_hold_distances.py`
Analyzes reachability distances between consecutive holds in the dataset.

### `reorder_paths_by_reachability.py` & `smart_reorder_paths.py`
Reorders holds in paths to optimize for reachability and climbing progression.

## Running Scripts

All scripts should be run from the project root:

```bash
# From e:\climb-path\
python backend/data_processing/analyze_data.py
python backend/data_processing/clean_data_for_training.py
```

## Model Approaches

### ✅ Implemented: Autoregressive Transformer (LegoACE-inspired)
**Location**: `models/`

A custom transformer model that generates climb paths token-by-token, inspired by LegoACE's sequential Lego assembly approach.

**Key Features**:
- Autoregressive generation with grade conditioning
- Logits processors for structural constraints
- 45-token vocabulary (BOS, grades, x/y coords, EOS)
- ~2.5M parameters (efficient and trainable on single GPU)

**See**: `models/README.md` and `QUICKSTART.md` for details

### Future Approaches
1. **Conditional VAE** - Generate paths conditioned on difficulty
2. **Diffusion Model** - Iterative refinement of climb paths
3. **Reinforcement Learning** - Learn from climber feedback

## API Endpoints (Planned)

- `POST /generate` - Generate a new climb path
- `GET /validate` - Validate a climb path
- `POST /predict-difficulty` - Predict difficulty of a path
