# Vertical Progression Training Enhancement

## Overview

The training script has been enhanced to address the issue where the model was generating flat horizontal paths instead of realistic vertical climbing paths. The enhancements include:

1. **Vertical Progression Loss** - Encourages Y-coordinate variety and upward movement
2. **Start Position Constraints** - Ensures paths begin at lower positions (Y ≤ 6)
3. **End Position Constraints** - Ensures paths end at higher positions (Y ≥ 12)

## Implementation Details

### 1. Vertical Progression Loss

The `compute_vertical_progression_loss()` method implements three components:

- **Downward Penalty**: Penalizes large downward moves (> 2 units), while allowing small downward adjustments
- **Upward Progression**: Encourages minimum vertical gain of 8 units from start to end
- **Flat Move Penalty**: Discourages consecutive moves with minimal Y-coordinate change

**Weight**: 0.5 (configurable via `self.vertical_loss_weight`)

### 2. Start Position Constraint

The `compute_position_constraint_loss()` method ensures the first hold has Y ≤ 6, matching the training data distribution where paths typically start at Y values between 2-5.

**Weight**: 0.3 (configurable via `self.start_constraint_weight`)

### 3. End Position Constraint

Ensures the last hold has Y ≥ 12, matching the training data where paths typically end at Y values between 15-17 (often at Y=17, the top of the wall).

**Weight**: 0.3 (configurable via `self.end_constraint_weight`)

## Training Data Analysis

From the sample data analysis:
- **Start positions**: Typically Y ∈ [2, 5]
- **End positions**: Typically Y ∈ [15, 17]
- **Vertical gain**: Usually 10-15 units
- **Example path**: `[[5, 4], [3, 8], [1, 10], [6, 12], [8, 15], [6, 17]]` shows Y progression: 4→8→10→12→15→17

## Loss Function

The total loss is now:

```
Total Loss = CE_Loss + 
             0.5 × Vertical_Loss + 
             0.3 × Start_Loss + 
             0.3 × End_Loss
```

Where:
- **CE_Loss**: Cross-entropy loss for next-token prediction
- **Vertical_Loss**: Encourages vertical progression
- **Start_Loss**: Penalizes high starting positions
- **End_Loss**: Penalizes low ending positions

## Monitoring Training

The training script now logs additional metrics to TensorBoard:

- `Train/CE_Loss` - Cross-entropy loss
- `Train/Vertical_Loss` - Vertical progression loss
- `Train/Start_Loss` - Start position constraint loss
- `Train/End_Loss` - End position constraint loss
- `Val/CE_Loss`, `Val/Vertical_Loss`, `Val/Start_Loss`, `Val/End_Loss` - Validation metrics

## Usage

Train the model as usual:

```bash
# Windows
train_cpu.bat

# Linux/Mac
./train_cpu.sh
```

Or with custom parameters:

```bash
python backend/training/train_autoregressive.py \
    --train_csv data/moonboard_train.csv \
    --val_csv data/moonboard_val.csv \
    --num_epochs 50 \
    --batch_size 32
```

## Tuning Loss Weights

If you need to adjust the balance between different loss components, modify these values in the `Trainer.__init__()` method:

```python
self.vertical_loss_weight = 0.5      # Increase to emphasize vertical progression
self.start_constraint_weight = 0.3   # Increase to enforce lower start positions
self.end_constraint_weight = 0.3     # Increase to enforce higher end positions
```

## Expected Results

After training with these constraints, the model should generate paths that:

1. ✅ Start at lower Y positions (Y ≤ 6)
2. ✅ Progress vertically upward throughout the climb
3. ✅ End at higher Y positions (Y ≥ 12, ideally 15-17)
4. ✅ Have realistic vertical variety (not flat horizontal paths)
5. ✅ Match the distribution of benchmark climbing problems

## Evaluation

The evaluation script has also been updated to track all loss components, providing comprehensive metrics on test data.

### Running Evaluation

Evaluate a trained model:

```bash
python backend/training/evaluate_model.py --checkpoint checkpoints/climb_path/best.pt
```

### Evaluation Metrics

The evaluation script now reports:

- **Total Loss** - Combined loss (CE + vertical + constraints)
- **Cross-Entropy Loss** - Next-token prediction loss
- **Vertical Loss** - Vertical progression quality
- **Start Position Loss** - How well paths start at low positions
- **End Position Loss** - How well paths end at high positions
- **Perplexity** - Model confidence metric

Example output:
```
============================================================
TEST SET RESULTS
============================================================
Total Loss:           2.3456
Cross-Entropy Loss:   2.1234
Vertical Loss:        0.1234
Start Position Loss:  0.0567
End Position Loss:    0.0421
Perplexity:           8.3456
Num Batches:          79
Num Tokens:           45,678
============================================================
```

Lower vertical, start, and end losses indicate the model generates more realistic climbing paths.

## Verification

After training, test the model with:

```bash
python test_model.py
```

Check that generated paths show proper vertical progression similar to training examples.
