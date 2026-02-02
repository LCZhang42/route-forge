# Quick Start Guide - Autoregressive Climb Path Generation

This guide will help you train and use the LegoACE-inspired autoregressive model for generating MoonBoard climbing routes.

## Prerequisites

```bash
# Install dependencies
cd backend
pip install -r requirements.txt
```

## Step 1: Prepare Data

Your data should already be cleaned from the previous preprocessing step. Verify you have:

```
data/
├── moonboard_train.csv
├── moonboard_val.csv
└── moonboard_test.csv
```

Each CSV should contain:
- `grade`: Grade string (6B, 6B+, ..., 8B+)
- `full_path`: List of [x, y] coordinates

## Step 2: Train the Model

### Basic Training

```bash
cd backend
python training/train_autoregressive.py
```

This will:
- Train for 50 epochs (default)
- Save checkpoints to `checkpoints/climb_path/`
- Log metrics to TensorBoard

### Monitor Training

```bash
tensorboard --logdir runs/climb_path
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Learning rate schedule
- Other metrics

### Custom Training Configuration

```bash
python training/train_autoregressive.py \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --num_layers 8 \
    --warmup_steps 2000
```

**Recommended settings for better results:**
- Increase `d_model` to 512 for more capacity
- Increase `num_layers` to 8-12 for complex patterns
- Use larger `batch_size` if you have GPU memory
- Train for 100+ epochs for best performance

## Step 3: Generate Climb Paths

### Generate with Visualization

```bash
python training/generate_paths.py \
    --grade 7A \
    --num_samples 5 \
    --visualize
```

Output:
```
Climb Path - Grade: 7A
Number of holds: 5

   01234567890
  +-----------+
17|.....5.....|
16|...........|
15|...........|
14|...........|
13|...4.......|
12|...........|
11|..3........|
10|...........|
 9|...........|
 8|.2.........|
 7|...........|
 6|...........|
 5|...........|
 4|1..........|
 3|...........|
 2|...........|
 1|...........|
  +-----------+
```

### Generate Multiple Grades

```bash
# Easy routes
python training/generate_paths.py --grade 6B --num_samples 10 --visualize

# Hard routes
python training/generate_paths.py --grade 8A+ --num_samples 10 --visualize
```

### Control Creativity

```bash
# Conservative (more realistic)
python training/generate_paths.py --grade 7A --temperature 0.5

# Creative (more diverse)
python training/generate_paths.py --grade 7A --temperature 1.5
```

### Save to JSON

```bash
python training/generate_paths.py \
    --grade 7B+ \
    --num_samples 20 \
    --temperature 0.8 \
    --save_json outputs/7B+_routes.json
```

## Step 4: Use in Python

```python
import torch
from models import (
    ClimbPathTokenizer,
    ClimbPathTransformerWithGeneration,
    ClimbPathLogitsProcessor,
    MinHoldsLogitsProcessor,
    MaxHoldsLogitsProcessor,
)

# Load model
tokenizer = ClimbPathTokenizer()
model = ClimbPathTransformerWithGeneration(vocab_size=45)

checkpoint = torch.load('checkpoints/climb_path/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Setup constraints
logits_processors = [
    ClimbPathLogitsProcessor(tokenizer),
    MinHoldsLogitsProcessor(tokenizer, min_holds=3),
    MaxHoldsLogitsProcessor(tokenizer, max_holds=20),
]

# Generate
grade_token = tokenizer.encode_grade('7A')
tokens = model.generate_with_processors(
    grade_token=grade_token,
    tokenizer=tokenizer,
    logits_processors=logits_processors,
    temperature=1.0,
    device='cuda',
)

# Decode
grade, holds = tokenizer.decode(tokens.cpu().numpy())
print(f"Grade: {grade}")
print(f"Holds: {holds}")
```

## Understanding the Model

### Architecture
- **Type**: Transformer decoder (autoregressive)
- **Size**: ~2.5M parameters (default config)
- **Vocabulary**: 45 tokens (BOS, 14 grades, 11 x-coords, 17 y-coords, EOS)
- **Sequence**: `[BOS, grade, x1, y1, x2, y2, ..., EOS]`

### How It Works

1. **Input**: Grade token (e.g., "7A" → token 6)
2. **Generation**: Model predicts next token autoregressively
3. **Constraints**: Logits processors ensure valid tokens at each position
4. **Output**: Sequence of holds `[(x1, y1), (x2, y2), ...]`

### Key Features

✅ **Structural Validity**: Logits processors ensure only valid coordinates  
✅ **Grade Conditioning**: Generates routes matching target difficulty  
✅ **Variable Length**: Handles 3-30 holds naturally  
✅ **Spatial Learning**: Discovers reachability patterns from data  

## Tips for Best Results

### Training
1. **More data = better**: Use all 11k+ routes
2. **Train longer**: 100+ epochs for convergence
3. **Larger model**: Try d_model=512, num_layers=8
4. **Monitor validation**: Stop if val_loss stops improving

### Generation
1. **Temperature tuning**: 
   - 0.5-0.7 for realistic routes
   - 0.8-1.0 for balanced
   - 1.2-1.5 for creative/diverse
2. **Constraints**: Keep enabled for valid routes
3. **Multiple samples**: Generate 10-20 to pick best
4. **Grade matching**: Model learns grade patterns from data

## Troubleshooting

### Model generates invalid sequences
- Ensure logits processors are enabled
- Check tokenizer is working correctly
- Verify training data quality

### Training loss not decreasing
- Reduce learning rate (try 5e-5)
- Increase warmup steps
- Check data preprocessing

### Out of memory
- Reduce batch_size
- Reduce max_seq_len
- Use smaller model (d_model=128, num_layers=4)

## Next Steps

1. **Evaluate quality**: Compare generated routes to real ones
2. **Add constraints**: Implement reachability distance checks
3. **Fine-tune**: Adjust on high-quality routes only
4. **Deploy**: Integrate with web frontend
5. **Extend**: Add start/end position conditioning

## Comparison with LegoACE

Your model is **simpler and more efficient** than LegoACE:

| Metric | LegoACE | Your Model |
|--------|---------|------------|
| Tokens per item | 5 | 2 |
| Vocabulary | ~16,000 | 45 |
| Model size | LLaMA (billions) | Custom (millions) |
| Training time | Days (multi-GPU) | Hours (single GPU) |
| Domain | 3D assembly | 2D climbing |

The same **core principles** apply:
- Autoregressive generation
- Structural constraints via logits processors
- Conditional generation (grade vs text)
- Variable-length sequences

---

**You're now ready to generate climbing routes!** 🧗‍♂️
