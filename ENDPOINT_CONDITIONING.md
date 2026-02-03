# Endpoint Conditioning for Climb Path Generation

## Overview

The DistilGPT-2 model is trained with **endpoint conditioning**, which means:

**Input:** Grade + Start Hold + End Hold  
**Output:** Intermediate Holds

This approach is more practical because:
- ✅ You specify where to start and where to finish
- ✅ The model generates the path between these points
- ✅ More control over the generated routes
- ✅ Better for route setting (you know the endpoints)

## Data Format

### Training Format
```
GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
```

**Components:**
- `GRADE`: Difficulty grade (6B to 8B+)
- `START`: Starting hold coordinates (x, y)
- `END`: Finishing hold coordinates (x, y)
- `MID`: Intermediate holds that the model predicts

### Example
```
Input:  GRADE: 7A | START: (0,4) | END: (8,17) | MID:
Output: GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
```

The model learns to fill in the intermediate holds based on:
- The difficulty grade
- The starting position
- The ending position
- Patterns learned from real climbing routes

## Training

### Quick Start (Recommended)

```bash
train_distilgpt2.bat
```

This will:
- Use DistilGPT-2 (82M parameters, lightweight)
- Train with endpoint conditioning format
- Include safety checks for NaN/Inf values
- Save checkpoints every 500 steps
- Complete in ~1-2 hours on CPU, ~20 minutes on GPU

### Training Parameters

Default settings in `train_distilgpt2.bat`:
- **Model**: DistilGPT-2 (82M parameters)
- **Batch size**: 16
- **Epochs**: 10
- **Learning rate**: 5e-5
- **Max length**: 256 tokens

### Safety Features

The training script includes automatic checks for:
- ⚠️ **NaN detection**: Stops training if loss becomes NaN
- ⚠️ **Inf detection**: Stops training if loss becomes infinite
- ⚠️ **Gradient monitoring**: Tracks gradient flow
- 📊 **TensorBoard logging**: Real-time training visualization

## Generation

### Interactive Generation

```bash
generate_paths.bat
```

You'll be prompted to enter:
1. **Grade** (e.g., `7A`) - or press Enter for random
2. **Start hold** (e.g., `0,4`) - or press Enter for random
3. **End hold** (e.g., `8,17`) - or press Enter for random

### Command Line Examples

**Generate with specific endpoints:**
```bash
python backend/training/generate_with_gpt2.py ^
    --grade 7A ^
    --start_hold 0,4 ^
    --end_hold 8,17 ^
    --num_samples 5
```

**Generate with grade only (random endpoints):**
```bash
python backend/training/generate_with_gpt2.py ^
    --grade 7B ^
    --num_samples 10
```

**Fully random generation:**
```bash
python backend/training/generate_with_gpt2.py --num_samples 5
```

**Adjust creativity:**
```bash
python backend/training/generate_with_gpt2.py ^
    --grade 7A ^
    --start_hold 2,3 ^
    --end_hold 7,16 ^
    --temperature 1.0 ^
    --num_samples 10
```

### Generation Parameters

- `--temperature`: Controls randomness
  - `0.5` = Conservative, safe paths
  - `0.8` = Balanced (default)
  - `1.0` = Creative, varied paths
  - `1.2` = Very creative, may be unrealistic

- `--top_k`: Top-k sampling (default: 50)
- `--top_p`: Nucleus sampling (default: 0.95)

## Model Architecture

### DistilGPT-2 Specifications

| Feature | Value |
|---------|-------|
| Parameters | 82M |
| Layers | 6 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Vocabulary | ~50k tokens |
| Context Length | 1024 tokens |

### Why DistilGPT-2?

- **Lightweight**: 33% smaller than GPT-2
- **Fast**: Trains 60% faster than GPT-2
- **Efficient**: Uses less memory
- **Effective**: Retains 97% of GPT-2's performance
- **Pre-trained**: Already understands sequences

## Troubleshooting

### "Probability tensor contains inf/nan"

This error is prevented by:
1. **SafetyCallback**: Monitors loss for NaN/Inf
2. **Gradient clipping**: Prevents exploding gradients
3. **Warmup steps**: Gradual learning rate increase
4. **Weight decay**: Regularization to prevent instability

If you still encounter this:
- Reduce learning rate: `--learning_rate 2e-5`
- Increase warmup: `--warmup_steps 1000`
- Reduce batch size: `--batch_size 8`

### Poor Generation Quality

If generated paths are invalid:
1. **Train longer**: `--num_epochs 20`
2. **Lower temperature**: `--temperature 0.6`
3. **Check training loss**: Should be < 2.0 after 10 epochs
4. **Verify data**: Ensure training data is clean

### Out of Memory

If training fails with OOM:
1. Reduce batch size: `--batch_size 8`
2. Reduce max length: `--max_length 128`
3. Use gradient accumulation: `--gradient_accumulation_steps 2`

### Slow Training

To speed up training:
1. Use GPU (automatically detected)
2. Enable mixed precision: `--fp16` (requires CUDA)
3. Increase batch size if memory allows: `--batch_size 32`

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir checkpoints/distilgpt2_climb/logs
```

Then open: http://localhost:6006

**Metrics to watch:**
- **Loss**: Should decrease steadily (target: < 2.0)
- **Learning rate**: Should warm up then stay constant
- **Gradient norm**: Should be stable (not exploding)

### Checkpoints

Models are saved to:
```
checkpoints/distilgpt2_climb/
├── checkpoint-500/     # Every 500 steps
├── checkpoint-1000/
├── checkpoint-1500/
├── final/              # Final trained model
├── logs/               # TensorBoard logs
└── config.json         # Training configuration
```

## Example Output

```
Sample 1:
  Generated: GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
  Parsed Grade: 7A
  Start Hold: [0, 4]
  End Hold: [8, 17]
  Intermediate Holds: [[1, 7], [3, 11], [5, 13]]
  Total holds: 5
```

## Advantages of Endpoint Conditioning

| Aspect | Endpoint Conditioning | Full Path Generation |
|--------|----------------------|---------------------|
| **Control** | High - specify start/end | Low - fully random |
| **Practical** | Very - matches route setting | Less - need to filter |
| **Training** | Easier - focused task | Harder - more complex |
| **Quality** | Better - constrained problem | Variable - unconstrained |
| **Use Case** | Route setting, puzzles | Exploration, variety |

## Next Steps

1. **Train the model**: Run `train_distilgpt2.bat`
2. **Monitor progress**: Check TensorBoard
3. **Generate paths**: Use `generate_paths.bat`
4. **Evaluate quality**: Test with different grades and endpoints
5. **Fine-tune**: Adjust temperature and sampling parameters

## Files

- `backend/training/finetune_huggingface.py` - Training script with endpoint conditioning
- `backend/training/generate_with_gpt2.py` - Generation script with endpoint support
- `train_distilgpt2.bat` - Training batch script
- `generate_paths.bat` - Interactive generation script

## Comparison: Before vs After

### Before (Full Path)
```
Input:  GRADE: 7A | PATH:
Output: GRADE: 7A | PATH: (0,4) (1,7) (3,11) (5,13) (8,17)
```
- ❌ No control over start/end
- ❌ May generate unsuitable endpoints
- ❌ Less practical for route setting

### After (Endpoint Conditioning)
```
Input:  GRADE: 7A | START: (0,4) | END: (8,17) | MID:
Output: GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
```
- ✅ Full control over start/end
- ✅ Generates paths between specified points
- ✅ Perfect for route setting
- ✅ More practical and usable
