# Quality-Based Training Guide

## Your Proposed Strategies ✓

All four strategies you mentioned make excellent sense:

### ✅ 1. Filter by repeats >= 10
**Why it works**: Removes untested/low-quality routes, focuses on proven climbs

### ✅ 2. Use is_benchmark=True
**Why it works**: Community-vetted routes, guaranteed quality, great for prototyping

### ✅ 3. Use repeats as quality weight
**Why it works**: Popular routes influence model more, learns what makes routes "good"

### ✅ 4. Stratify by grade AND quality
**Why it works**: Balanced representation across difficulty and quality levels

## Implementation

### Step 1: Filter Your Data

Run the quality filtering script:
```bash
python backend/data_processing/filter_quality_data.py
```

This creates:
- `data/moonboard_train_quality.csv` - Routes with >= 10 repeats
- `data/moonboard_train_benchmark.csv` - Benchmark routes only
- `data/quality_weights.csv` - Quality weights for training

### Step 2: Choose Training Strategy

#### **Strategy A: Benchmark-Only (Quick Prototype)**
```bash
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_benchmark.csv \
    --val_csv data/moonboard_val_benchmark.csv \
    --device cpu \
    --batch_size 16 \
    --num_epochs 30
```

**Best for**: Quick testing, high-quality subset, faster training

#### **Strategy B: Quality-Filtered (Recommended)**
```bash
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --device cpu \
    --batch_size 16 \
    --num_epochs 50
```

**Best for**: Balanced quality and quantity, proven routes

#### **Strategy C: Full Dataset (Maximum Data)**
```bash
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train.csv \
    --val_csv data/moonboard_val.csv \
    --device cpu \
    --batch_size 16 \
    --num_epochs 100
```

**Best for**: Maximum diversity, learning from all routes

## Quality Weighting Implementation

I've created `weighted_dataset.py` that implements quality-based sampling:

```python
from models.weighted_dataset import create_weighted_dataloader

# Create dataloader with quality weighting
train_loader = create_weighted_dataloader(
    csv_path='data/moonboard_train_quality.csv',
    tokenizer=tokenizer,
    batch_size=32,
    use_quality_weights=True,  # Popular routes sampled more often
)
```

**How it works**:
- Routes with more repeats get higher sampling probability
- Uses log scale to avoid extreme weights
- Popular routes (100+ repeats) appear ~2-3x more often in training

## Expected Results by Strategy

| Strategy | Dataset Size | Training Time | Quality | Diversity |
|----------|--------------|---------------|---------|-----------|
| **Benchmark** | ~500-1000 | Fastest | Highest | Lower |
| **Quality (>= 10)** | ~5000-7000 | Medium | High | Good |
| **Full** | ~11000 | Slowest | Mixed | Highest |

## Stratification Benefits

The filtering script uses stratified sampling by `grade × quality_bin`:

```python
# Creates strata like:
'6B_low', '6B_medium', '6B_high',
'7A_low', '7A_medium', '7A_high',
...
```

**Benefits**:
- Ensures all grades represented
- Balances easy popular routes vs hard rare routes
- Prevents model bias toward easy/popular combinations

## Recommendations

### For Your First Training Run

1. **Start with quality-filtered** (>= 10 repeats):
   ```bash
   python backend/data_processing/filter_quality_data.py
   python backend/training/train_autoregressive.py \
       --train_csv data/moonboard_train_quality.csv \
       --val_csv data/moonboard_val_quality.csv \
       --device cpu \
       --batch_size 16 \
       --num_epochs 50
   ```

2. **Why this is best**:
   - ✓ Removes noise (0-9 repeat routes)
   - ✓ Still has good dataset size (~5-7k routes)
   - ✓ Proven, climbable routes
   - ✓ Stratified by grade and quality

### Progressive Training Strategy

```
Phase 1: Benchmark-only (30 epochs)
   ↓ Quick prototype, verify model works
   
Phase 2: Quality-filtered (50 epochs)
   ↓ Train on proven routes
   
Phase 3: Full dataset (100 epochs)
   ↓ Fine-tune with all data for diversity
```

## Quality Metrics to Track

When training, monitor:

1. **Loss by grade**: Does model learn all difficulties?
2. **Loss by quality bin**: Does it handle rare routes?
3. **Generated route quality**: Do outputs look climbable?

## Files Created

- ✅ `backend/data_processing/filter_quality_data.py` - Filtering script
- ✅ `backend/models/weighted_dataset.py` - Quality-weighted sampling
- ✅ `QUALITY_FILTERING_GUIDE.md` - This guide

## Summary

Your proposed strategies are **excellent** and well-thought-out:

1. ✅ **Filtering by repeats** removes noise
2. ✅ **Benchmark subset** great for prototyping
3. ✅ **Quality weighting** emphasizes popular routes
4. ✅ **Stratification** ensures balanced representation

All four can be combined for optimal results!
