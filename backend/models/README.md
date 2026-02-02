# Autoregressive Climb Path Generation Model

Transformer-based model for generating MoonBoard climbing routes inspired by LegoACE's sequential generation approach.

## Architecture Overview

### Core Concept
The model generates climb paths **token-by-token** in an autoregressive manner, similar to how LegoACE generates Lego brick sequences. Each hold is represented as 2 tokens `[x, y]`, making it simpler than Lego's 5-token representation `[x, y, z, rotation, type]`.

### Token Vocabulary (45 tokens)

```
Token 0:      BOS (beginning of sequence)
Tokens 1-14:  Grade tokens (6B, 6B+, 6C, ..., 8B+)
Tokens 15-25: X coordinates (0-10)
Tokens 26-43: Y coordinates (1-17)
Token 44:     EOS (end of sequence)
```

### Sequence Format

```
[BOS, grade_token, x1, y1, x2, y2, ..., xn, yn, EOS]
```

Example for grade 7A with 3 holds:
```
[0, 6, 15, 30, 18, 35, 20, 40, 44]
 │  │   │   │   │   │   │   │   └─ EOS
 │  │   │   │   │   │   │   └───── y3 = 14 (token 40 → y=14)
 │  │   │   │   │   │   └───────── x3 = 5  (token 20 → x=5)
 │  │   │   │   └───────────────── y2 = 9  (token 35 → y=9)
 │  │   │   └───────────────────── x2 = 3  (token 18 → x=3)
 │  │   └───────────────────────── y1 = 4  (token 30 → y=4)
 │  └───────────────────────────── x1 = 0  (token 15 → x=0)
 └──────────────────────────────── grade = 7A (token 6)
```

## Model Components

### 1. Tokenizer (`tokenizer.py`)
- Encodes/decodes between climb paths and token sequences
- Handles grade encoding and coordinate mapping
- Validates token positions

### 2. Transformer Model (`climb_transformer.py`)
- **Architecture**: Transformer decoder with causal masking
- **Default config**: 6 layers, 256 hidden dim, 8 attention heads
- **Parameters**: ~2.5M (much smaller than typical LLMs)
- **Input**: Token sequence
- **Output**: Next-token logits

### 3. Logits Processors (`logits_processor.py`)
Enforce structural constraints during generation:

- **ClimbPathLogitsProcessor**: Ensures valid token types at each position
  - Position 0: Only BOS
  - Position 1: Only grade tokens
  - Even positions: Only X coordinates or EOS
  - Odd positions: Only Y coordinates or EOS

- **MinHoldsLogitsProcessor**: Prevents early termination (min 3 holds)
- **MaxHoldsLogitsProcessor**: Forces termination after max holds (default 30)

### 4. Dataset (`dataset.py`)
- Loads preprocessed MoonBoard data
- Converts to token sequences
- Handles padding and batching

## Training

### Quick Start

```bash
# Train with default settings
python backend/training/train_autoregressive.py

# Custom configuration
python backend/training/train_autoregressive.py \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --num_layers 8
```

### Training Details

- **Loss**: Cross-entropy (teacher forcing)
- **Optimizer**: AdamW with warmup
- **Warmup**: 1000 steps
- **Gradient clipping**: Max norm 1.0
- **Logging**: TensorBoard

### Expected Performance

With 11k+ training examples:
- Training converges in ~30-50 epochs
- Validation loss should reach ~1.5-2.0
- Model learns spatial constraints from data

## Generation

### Basic Usage

```bash
# Generate 5 paths for grade 7A
python backend/training/generate_paths.py \
    --grade 7A \
    --num_samples 5 \
    --visualize

# Save to JSON
python backend/training/generate_paths.py \
    --grade 6B+ \
    --num_samples 10 \
    --temperature 0.8 \
    --save_json outputs/generated_paths.json
```

### Generation Parameters

- **temperature**: Controls randomness (0.5 = conservative, 1.5 = creative)
- **min_holds**: Minimum path length (default: 3)
- **max_holds**: Maximum path length (default: 30)
- **constraints**: Enable/disable logits processors

### Example Output

```
Climb Path - Grade: 7A
Number of holds: 5

   01234567890
  +-----------+
17|...........|
16|...........|
15|...........|
14|...........|
13|.....3.....|
12|...........|
11|...2.......|
10|...........|
 9|...........|
 8|...........|
 7|.1.........|
 6|...........|
 5|...........|
 4|0..........|
 3|...........|
 2|...........|
 1|...........|
  +-----------+
   01234567890

Hold sequence:
  1. [0, 4]
  2. [1, 7]
  3. [3, 11]
  4. [5, 13]
  5. [8, 17]
```

## Python API

```python
from models import (
    ClimbPathTokenizer,
    ClimbPathTransformerWithGeneration,
    ClimbPathLogitsProcessor,
)

# Initialize
tokenizer = ClimbPathTokenizer()
model = ClimbPathTransformerWithGeneration(vocab_size=45)
model.load_state_dict(torch.load('checkpoints/best.pt')['model_state_dict'])

# Generate
logits_processors = [ClimbPathLogitsProcessor(tokenizer)]
grade_token = tokenizer.encode_grade('7A')

tokens = model.generate_with_processors(
    grade_token=grade_token,
    tokenizer=tokenizer,
    logits_processors=logits_processors,
    temperature=1.0,
)

# Decode
grade, holds = tokenizer.decode(tokens)
print(f"Generated {len(holds)} holds for grade {grade}")
```

## Comparison with LegoACE

| Aspect | LegoACE | Our Model |
|--------|---------|-----------|
| **Domain** | 3D Lego assembly | 2D climbing routes |
| **Tokens per item** | 5 (x, y, z, rotation, type) | 2 (x, y) |
| **Vocabulary size** | ~16,000 | 45 |
| **Constraints** | Physical stability | Reachability |
| **Conditioning** | Text/image | Grade difficulty |
| **Model size** | LLaMA-based (large) | Custom transformer (small) |

## Advantages

✅ **Simpler than Lego**: 2D grid vs 3D space  
✅ **Efficient**: Small vocabulary and model  
✅ **Constrained**: Logits processors ensure validity  
✅ **Flexible**: Variable-length sequences  
✅ **Learnable**: Discovers spatial patterns from data  

## Future Improvements

- Add reachability constraints (distance between consecutive holds)
- Multi-task learning (predict difficulty + generate path)
- Reinforcement learning with climber feedback
- Condition on start/end positions
- Add hold type information (if available)

## Files

```
backend/models/
├── __init__.py              # Package exports
├── tokenizer.py             # Token encoding/decoding
├── climb_transformer.py     # Model architecture
├── logits_processor.py      # Constraint enforcement
├── dataset.py               # Data loading
└── README.md                # This file

backend/training/
├── train_autoregressive.py # Training script
└── generate_paths.py        # Generation script
```
