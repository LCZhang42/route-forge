# Autoregressive Climb Path Generation Model

## Overview

This document describes the implementation of an autoregressive transformer model for MoonBoard climb path generation, inspired by the LegoACE paper's approach to sequential Lego brick assembly.

## Motivation: Why LegoACE's Approach Works for Climbing

### LegoACE Core Concept
LegoACE generates 3D Lego structures by predicting brick placements sequentially:
- Each brick = 5 tokens: `[x, y, z, rotation_id, brick_type_id]`
- Autoregressive generation with structural constraints
- Conditioning on text descriptions or images
- Logits processors enforce physical validity

### Adaptation to Climb Paths

The climbing route generation problem maps naturally to this approach:

| **LegoACE (Lego Assembly)** | **Our Model (Climb Paths)** |
|------------------------------|------------------------------|
| 3D brick placement | 2D hold placement |
| 5 tokens per brick | **2 tokens per hold** |
| Physical stability constraints | Reachability constraints |
| Text/image conditioning | **Grade conditioning** |
| ~16,000 token vocabulary | **45 token vocabulary** |
| LLaMA-based (billions of params) | **Custom transformer (millions)** |

**Key Advantage**: Climbing is simpler than Lego assembly! We only need 2D coordinates with no rotation or type information.

## Architecture

### Token Vocabulary (45 tokens)

```
Token ID    Type            Description
--------    ----            -----------
0           BOS             Beginning of sequence
1-14        Grade           14 climbing grades (6B to 8B+)
15-25       X-coordinate    Grid positions 0-10
26-43       Y-coordinate    Grid positions 1-17
44          EOS             End of sequence
```

### Sequence Format

```
[BOS, grade_token, x₁, y₁, x₂, y₂, ..., xₙ, yₙ, EOS]
```

**Example**: Grade 7A with holds at (0,4), (1,7), (3,11)
```
[0, 6, 15, 30, 16, 33, 18, 37, 44]
 │  │  │   │   │   │   │   │   └─ EOS
 │  │  │   │   │   │   │   └───── y₃ = 11 (token 37 → 11)
 │  │  │   │   │   │   └───────── x₃ = 3  (token 18 → 3)
 │  │  │   │   │   └───────────── y₂ = 7  (token 33 → 7)
 │  │  │   │   └───────────────── x₂ = 1  (token 16 → 1)
 │  │  │   └───────────────────── y₁ = 4  (token 30 → 4)
 │  │  └───────────────────────── x₁ = 0  (token 15 → 0)
 │  └──────────────────────────── grade = 7A (token 6)
 └─────────────────────────────── BOS
```

### Model Architecture

```
Input: Token IDs [batch_size, seq_len]
  ↓
Embedding Layer (vocab_size=45 → d_model=256)
  ↓
Positional Encoding (sinusoidal)
  ↓
Transformer Decoder (6 layers, 8 heads)
  - Self-attention with causal masking
  - Feedforward networks (dim=1024)
  - Layer normalization
  - Dropout (0.1)
  ↓
Output Projection (d_model → vocab_size)
  ↓
Logits [batch_size, seq_len, vocab_size]
```

**Parameters**: ~2.5M (default config)
- Much smaller than LegoACE's LLaMA-based model
- Trainable on single GPU in hours

## Constraint Enforcement: Logits Processors

Following LegoACE's approach, we use logits processors to enforce structural validity during generation.

### 1. ClimbPathLogitsProcessor

Ensures correct token types at each position:

```python
Position 0:  Only BOS token (ID 0)
Position 1:  Only grade tokens (IDs 1-14)
Position 2:  X-coords or EOS (IDs 15-25, 44)
Position 3:  Y-coords or EOS (IDs 26-43, 44)
Position 4:  X-coords or EOS
Position 5:  Y-coords or EOS
...
```

This is analogous to LegoACE's `DynamicRangeMaskingProcessor` which enforces:
```python
# LegoACE's approach
if cur_len % 5 in [0, 1, 2]:  # x, y, z coordinates
    return position_range
elif cur_len % 5 == 3:  # rotation
    return rotation_range
elif cur_len % 5 == 4:  # brick type
    return brick_type_range
```

### 2. MinHoldsLogitsProcessor

Prevents premature termination (minimum 3 holds).

### 3. MaxHoldsLogitsProcessor

Forces termination after maximum holds (default 30).

## Training

### Loss Function

Cross-entropy with teacher forcing:
```python
# Predict next token given previous tokens
shift_logits = logits[:, :-1, :]  # All but last prediction
shift_labels = input_ids[:, 1:]    # All but first token

loss = CrossEntropyLoss(shift_logits, shift_labels)
```

### Optimization

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Warmup**: 1000 steps with linear warmup
- **Gradient clipping**: Max norm 1.0
- **Batch size**: 32 (default)

### Training Data

- **Size**: 11,000+ climb paths
- **Split**: 80% train, 10% val, 10% test
- **Augmentation**: None (future: rotation, mirroring)

## Generation

### Autoregressive Sampling

```python
# Start with BOS and grade
sequence = [BOS, grade_token]

while len(sequence) < max_length:
    # Forward pass
    logits = model(sequence)
    
    # Apply logits processors
    next_logits = logits[-1] / temperature
    for processor in logits_processors:
        next_logits = processor(sequence, next_logits)
    
    # Sample next token
    probs = softmax(next_logits)
    next_token = sample(probs)
    
    # Append and check for EOS
    sequence.append(next_token)
    if next_token == EOS:
        break

return sequence
```

### Sampling Strategies

- **Temperature**: Controls randomness
  - 0.5-0.7: Conservative, realistic routes
  - 0.8-1.0: Balanced
  - 1.2-1.5: Creative, diverse routes

- **Top-k/Top-p**: Can be added for controlled diversity

## Comparison with LegoACE

### Similarities ✓

1. **Autoregressive generation**: Token-by-token prediction
2. **Structural constraints**: Logits processors enforce validity
3. **Conditional generation**: Grade (ours) vs text/image (theirs)
4. **Variable length**: Natural handling of different sequence lengths
5. **Learned patterns**: Model discovers spatial relationships from data

### Differences

1. **Simpler domain**: 2D vs 3D
2. **Fewer tokens per item**: 2 vs 5
3. **Smaller vocabulary**: 45 vs ~16,000
4. **Custom architecture**: Small transformer vs LLaMA
5. **Faster training**: Hours vs days

### Why This Works

Both Lego assembly and climb path generation share key properties:
- **Sequential structure**: Items placed one after another
- **Spatial constraints**: Each item depends on previous positions
- **Discrete space**: Finite set of valid positions
- **Conditional generation**: Output depends on high-level specification

## Results & Evaluation

### Expected Performance

After training on 11k+ routes:
- **Validation loss**: ~1.5-2.0
- **Valid sequences**: >95% with logits processors
- **Grade matching**: Model learns difficulty patterns
- **Spatial coherence**: Holds form climbable sequences

### Qualitative Evaluation

Generated paths should exhibit:
- ✓ Valid grid coordinates (0-10, 1-17)
- ✓ Reasonable hold spacing (learned from data)
- ✓ Upward progression (typical climbing pattern)
- ✓ Grade-appropriate complexity

### Future Improvements

1. **Reachability constraints**: Add distance-based logits processor
2. **Physics-aware**: Model body positions and movements
3. **Multi-task learning**: Joint difficulty prediction + generation
4. **Reinforcement learning**: Fine-tune with climber feedback
5. **Data augmentation**: Mirror/rotate existing routes

## Implementation Files

```
backend/models/
├── tokenizer.py              # Token encoding/decoding
├── climb_transformer.py      # Model architecture
├── logits_processor.py       # Constraint enforcement
├── dataset.py                # Data loading
└── test_tokenizer.py         # Unit tests

backend/training/
├── train_autoregressive.py   # Training script
└── generate_paths.py         # Generation script
```

## Usage Examples

### Training
```bash
python backend/training/train_autoregressive.py \
    --batch_size 64 \
    --num_epochs 100 \
    --d_model 512 \
    --num_layers 8
```

### Generation
```bash
python backend/training/generate_paths.py \
    --grade 7A \
    --num_samples 10 \
    --temperature 0.8 \
    --visualize
```

### Python API
```python
from models import ClimbPathTokenizer, ClimbPathTransformerWithGeneration

tokenizer = ClimbPathTokenizer()
model = ClimbPathTransformerWithGeneration(vocab_size=45)
model.load_state_dict(torch.load('checkpoints/best.pt'))

grade_token = tokenizer.encode_grade('7A')
tokens = model.generate_with_processors(
    grade_token=grade_token,
    tokenizer=tokenizer,
    logits_processors=[...],
)

grade, holds = tokenizer.decode(tokens)
```

## Conclusion

The LegoACE approach translates excellently to climb path generation. The sequential, constraint-based generation paradigm is a natural fit for creating valid climbing routes. Our implementation is simpler and more efficient than LegoACE while maintaining the core benefits of autoregressive generation with structural constraints.

**Key Takeaway**: The same principles that enable sequential Lego assembly can generate climbing routes—just with a simpler 2D representation and smaller model.

---

**References**:
- LegoACE Paper: [docs/legoACE.pdf]
- LegoACE Code: https://github.com/xh38/LegoACE-code
- Implementation: [backend/models/]
