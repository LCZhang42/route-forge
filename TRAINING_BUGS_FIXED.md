# Training Bugs Fixed

## Critical Bug #1: BOS Token Conflict with Padding ⚠️

**Severity:** CRITICAL - Model was not learning at all

**Root Cause:**
- `BOS_TOKEN = 0` was the same value as padding
- Loss function used `ignore_index=0` to skip padding tokens
- This caused ALL BOS tokens to be ignored in loss calculation
- Model never learned to predict tokens after BOS

**Symptoms:**
- Loss completely flat at 4.241 (random baseline)
- Validation loss identical across all epochs (4.241)
- No improvement over 50 epochs

**Fix Applied:**
- Added dedicated `PAD_TOKEN = 0`
- Shifted all tokens: `BOS_TOKEN = 1`, grades start at 2
- Updated vocabulary size from 45 to 46 tokens
- Now padding is ignored, but BOS contributes to loss

**Files Modified:**
- `backend/models/tokenizer.py` - Added PAD_TOKEN, shifted vocabulary
- `backend/models/dataset.py` - Use PAD_TOKEN for padding

---

## Bug #2: Learning Rate Warmup Issues

**Severity:** HIGH - Learning rate was near-zero

**Root Cause:**
- `current_step` incremented AFTER calculating learning rate
- `get_lr()` read from `optimizer.param_groups[0]['lr']` which was being overwritten
- Created feedback loop where LR kept getting multiplied by warmup factor

**Symptoms:**
- Learning rate started at ~5e-12 (essentially 0)
- Learning rate dropped to 0 after warmup period
- Model couldn't update weights effectively

**Fix Applied:**
- Store `base_lr` separately to avoid feedback loop
- Increment `current_step` BEFORE calculating learning rate
- Use `base_lr` in warmup calculation instead of reading from optimizer

**Files Modified:**
- `backend/training/train_autoregressive.py` - Fixed warmup calculation

---

## Additional Improvements

### Early Error Detection
Added sanity checks after epoch 1 to catch issues early:
- ✓ NaN/Inf detection
- ✓ Learning rate verification (catches lr=0)
- ✓ Gradient flow check
- ✓ Loss validity check

### Quick Testing Script
Created `test_training.bat` to run 3-epoch test before full training:
- Verifies setup in ~5-10 minutes
- Catches configuration errors early
- Saves time before committing to 50-epoch runs

### Training Visualization
Created `visualize_training.py` and `visualize_training.bat`:
- Real-time training curve analysis
- Learning rate schedule visualization
- Checkpoint performance comparison
- Summary statistics

---

## Action Required

**IMPORTANT:** You must delete old checkpoints and logs before retraining:

```bash
# Delete old training artifacts
rmdir /s /q checkpoints\climb_path_cpu
rmdir /s /q runs\climb_path_cpu
```

Then restart training:
```bash
# Test first (recommended)
test_training.bat

# Full training
train_cpu.bat
```

The model should now:
- ✅ Show learning rate ramping from 0 to 0.0001 over 500 steps
- ✅ Show decreasing loss over epochs
- ✅ Achieve validation loss well below 4.0
- ✅ Actually learn to generate climb paths
