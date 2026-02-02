# Reachability Loss for Training

## Overview

Reachability loss has been added to the training pipeline to teach the model to generate physically reachable climbing sequences. This complements the inference-time reachability constraints.

## Why Add Reachability Loss?

**Problem**: Even with properly sequenced training data, the model might learn to generate unreachable sequences because:
- The cross-entropy loss only cares about predicting the correct next token
- It doesn't understand physical constraints
- Some training data may still have violations (2.32% after reordering)

**Solution**: Add a loss term that explicitly penalizes unreachable hold transitions during training.

## Implementation

### Three Loss Types

#### 1. Standard Reachability Loss (default)
```python
ReachabilityLoss(tokenizer, max_reach=5.0, penalty_scale=1.0)
```
- Linear penalty for violations
- Penalty = max(0, distance - max_reach)
- Simple and interpretable

#### 2. Soft Reachability Loss
```python
SoftReachabilityLoss(tokenizer, max_reach=5.0, penalty_scale=0.5, sharpness=0.5)
```
- Exponential penalty: exp(sharpness * violation) - 1
- Lenient on small violations, harsh on large ones
- Better gradient flow

#### 3. Adaptive Reachability Loss
```python
AdaptiveReachabilityLoss(tokenizer, initial_reach=6.0, final_reach=4.5, penalty_scale=0.5)
```
- Adjusts max_reach based on path progress
- Larger reach at start (setting up)
- Tighter constraints at finish (precise moves)

### How It Works

```python
# For each hold after the first 4:
window = last_4_holds  # Approximate body position
next_hold = current_hold

# Calculate minimum distance to any hold in window
distances = [euclidean_distance(next_hold, h) for h in window]
min_distance = min(distances)

# Penalty if exceeds max_reach
if min_distance > max_reach:
    penalty = min_distance - max_reach  # Linear
    # or
    penalty = exp(sharpness * (min_distance - max_reach)) - 1  # Exponential
```

## Usage

### Training with Reachability Loss (default)

```bash
# Standard reachability loss (default, enabled)
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --num_epochs 50

# Soft reachability loss (exponential penalty)
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --reachability_loss_type soft \
    --num_epochs 50

# Adaptive reachability loss (adjusts with progress)
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --reachability_loss_type adaptive \
    --num_epochs 50

# Disable reachability loss (for comparison)
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --no_reachability_loss \
    --num_epochs 50

# Adjust loss weight
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train_quality.csv \
    --val_csv data/moonboard_val_quality.csv \
    --reachability_loss_weight 0.5 \
    --num_epochs 50
```

## Loss Composition

The total training loss is now:

```
Total Loss = CE Loss 
           + 0.5 * Vertical Loss 
           + 0.3 * Start Loss 
           + 0.3 * End Loss 
           + 0.3 * Reachability Loss  (NEW!)
```

Where:
- **CE Loss**: Cross-entropy (predict next token correctly)
- **Vertical Loss**: Encourage upward progression
- **Start Loss**: Start holds should be low (Y ≤ 6)
- **End Loss**: End holds should be high (Y ≥ 12)
- **Reachability Loss**: Holds should be within reach (≤ 5.0 units)

## Monitoring

During training, you'll see reachability loss in:

1. **Progress bar**:
```
Epoch 1 [Train]: 100%|██| 415/415 [03:45<00:00]
loss: 2.3456, ce: 1.8234, vert: 0.2145, reach: 0.1234, lr: 0.000100
```

2. **Epoch summary**:
```
CE Loss: 1.8234 | Vertical Loss: 0.2145 | 
Start Loss: 0.0456 | End Loss: 0.0321 | 
Reachability Loss: 0.1234
```

3. **TensorBoard**:
- `Train/Reachability_Loss` - per batch
- Compare with other loss components

## Expected Impact

### Before Reachability Loss
- Model may generate unreachable sequences
- Violations: ~2-3% in training data
- Inference constraints needed to filter

### After Reachability Loss
- Model learns to avoid unreachable transitions
- Should generate fewer violations naturally
- Inference constraints act as safety net
- Better generalization to new grades

## Hyperparameter Tuning

### Loss Weight (default: 0.3)
- **Too low (< 0.1)**: Model ignores reachability
- **Good range (0.2-0.5)**: Balanced learning
- **Too high (> 0.8)**: May sacrifice other objectives

### Max Reach (default: 5.0)
- Based on data analysis (95th percentile)
- **Tighter (4.0-4.5)**: More conservative, fewer violations
- **Looser (5.5-6.0)**: More creative, allows bigger moves

### Loss Type
- **Standard**: Best for most cases, interpretable
- **Soft**: Better gradients, use if training is unstable
- **Adaptive**: Use if you want easier starts, harder finishes

## Validation

After training with reachability loss, validate by:

1. **Generate paths** and check violation rate:
```bash
python backend/training/generate_paths.py \
    --checkpoint checkpoints/climb_path/best.pt \
    --grade 7A \
    --num_samples 100 \
    --save_json generated_paths.json
```

2. **Analyze violations**:
```python
from backend.models.reachability_constraints import validate_path_reachability

violations = 0
for path in generated_paths:
    is_valid, invalid_indices = validate_path_reachability(path['holds'], max_reach=5.0)
    if not is_valid:
        violations += 1

print(f"Violation rate: {violations / len(generated_paths) * 100:.1f}%")
```

3. **Compare with/without**:
- Train two models (with and without reachability loss)
- Generate 100 paths from each
- Compare violation rates

## Files

- **Loss implementation**: `backend/models/reachability_loss.py`
- **Training integration**: `backend/training/train_autoregressive.py`
- **Inference constraints**: `backend/models/reachability_processor.py`
- **Data reordering**: `backend/data_processing/smart_reorder_paths.py`

## Summary

Reachability loss teaches the model to generate physically plausible climbing sequences during training, rather than just filtering at inference time. This should result in:

✅ Fewer violations in generated paths
✅ Better understanding of physical constraints
✅ More realistic climbing sequences
✅ Better generalization to unseen grades

The loss is enabled by default with sensible parameters based on data analysis.
