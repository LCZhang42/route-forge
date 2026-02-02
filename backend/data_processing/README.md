# Data Processing Scripts

Scripts for analyzing and preparing MoonBoard climbing data.

## Scripts

### `analyze_data.py`
Provides comprehensive analysis of the dataset:
- Grade distribution
- Benchmark vs regular problems
- Popularity statistics (repeats)
- Hold position analysis

**Usage:**
```bash
python backend/data_processing/analyze_data.py
```

### `clean_data_for_training.py`
Prepares clean training data:
- Parses hold coordinates from string format
- Validates coordinate ranges (MoonBoard: 11x18 grid)
- Filters by quality (minimum repeats)
- Creates train/val/test splits (80/10/10)
- Exports multiple formats for different model types

**Outputs:**
- `data/moonboard_cleaned.csv` - Full cleaned dataset
- `data/moonboard_cleaned.json` - JSON format
- `data/moonboard_llm_format.jsonl` - LLM training format
- `data/moonboard_train.csv` - Training set
- `data/moonboard_val.csv` - Validation set
- `data/moonboard_test.csv` - Test set

**Usage:**
```bash
python backend/data_processing/clean_data_for_training.py
```

### `examine_data_structure.py`
Examines data structure and suggests model architectures.

### `convert_pickle_to_excel.py`
Converts raw pickle file to Excel format.

## Data Format

**Input Features:**
- `grade` - Difficulty level (6B to 8B+)
- `is_benchmark` - Quality indicator
- `repeats` - Number of times climbed

**Output/Target:**
- `start_holds` - Starting positions [[x,y], ...]
- `mid_holds` - Intermediate holds [[x,y], ...]
- `end_holds` - Finishing holds [[x,y], ...]
- `full_path` - Complete sequence of holds

**Coordinate System:**
- X: 0-10 (11 columns)
- Y: 1-17 (18 rows, 0-indexed from bottom)
