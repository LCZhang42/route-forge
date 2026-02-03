# Quick Start Guide - DistilGPT-2 Climb Path Generation

This guide will help you train and use the fine-tuned DistilGPT-2 model for generating MoonBoard climbing routes with endpoint conditioning.

## Prerequisites

```bash
# Install dependencies
cd backend
pip install -r requirements.txt
```

## Step 1: Prepare Data

### Quality Filtering
```bash
python check_data_quality.py
```

### Reorder by Y-axis
```bash
python reorder_all_datasets.py
```

### Check Grade Balance
```bash
python analyze_grade_distribution.py
```

This creates:
```
data_reordered/
├── moonboard_train_quality_filtered.csv  # 6,610 balanced routes
├── moonboard_test_quality.csv
└── moonboard_val_quality.csv
```

## Step 2: Train the Model

### Basic Training

```bash
# Windows
train_distilgpt2.bat

# Linux/Mac
python backend/training/finetune_huggingface.py
```

This will:
- Fine-tune DistilGPT-2 for 10 epochs
- Save checkpoints to `checkpoints/distilgpt2_climb/`
- Log metrics to TensorBoard
- **Time**: ~2-3 hours on CPU, ~20 minutes on GPU

### Monitor Training

```bash
tensorboard --logdir checkpoints/distilgpt2_climb/logs
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Learning rate schedule
- Target: validation loss < 2.0

### Custom Training Configuration

```bash
python backend/training/finetune_huggingface.py \
    --batch_size 8 \
    --num_epochs 15 \
    --learning_rate 5e-5 \
    --max_length 128
```

**Recommended settings:**
- Reduce `batch_size` to 4-8 for CPU training
- Reduce `max_length` to 128 for faster training
- Use 5-10 epochs (more doesn't help much)
- Default settings work well for most cases

## Step 3: Generate Climb Paths

### Interactive Generation

```bash
# Windows
generate_paths.bat

# Linux/Mac
python backend/training/generate_with_gpt2.py
```

You'll be prompted to enter:
- Grade (e.g., 7A)
- Start hold (e.g., 0,4)
- End hold (e.g., 8,17)

### Command Line Generation

```bash
python backend/training/generate_with_gpt2.py \
    --grade 7A \
    --start_hold 0,4 \
    --end_hold 8,17 \
    --num_samples 5
```

Output:
```
Sample 1:
  Generated: GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
  Parsed Grade: 7A
  Start Hold: [0, 4]
  End Hold: [8, 17]
  Intermediate Holds: [[1, 7], [3, 11], [5, 13]]
  Total holds: 5
```

### Generate Multiple Grades

```bash
# Easy routes
python backend/training/generate_with_gpt2.py --grade 6B+ --num_samples 10

# Hard routes
python backend/training/generate_with_gpt2.py --grade 7C+ --num_samples 10
```

### Control Creativity

```bash
# Conservative (more realistic)
python backend/training/generate_with_gpt2.py --grade 7A --temperature 0.6

# Creative (more diverse)
python backend/training/generate_with_gpt2.py --grade 7A --temperature 1.2
```

## Step 4: Use in Python

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model
model = GPT2LMHeadModel.from_pretrained('checkpoints/distilgpt2_climb/final')
tokenizer = GPT2Tokenizer.from_pretrained('checkpoints/distilgpt2_climb/final')
model.eval()

# Generate
prompt = "GRADE: 7A | START: (0,4) | END: (8,17) | MID:"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=256,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)

# Decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
# Output: GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)
```

## Understanding the Model

### Architecture
- **Type**: Fine-tuned DistilGPT-2 (Causal Language Model)
- **Size**: 82M parameters (6 layers, 768 hidden size)
- **Vocabulary**: ~50k tokens (GPT-2 tokenizer)
- **Format**: Text-based climbing path representation

### How It Works

1. **Input**: `GRADE: 7A | START: (0,4) | END: (8,17) | MID:`
2. **Generation**: Model predicts intermediate holds as text
3. **Endpoint Conditioning**: Start and end positions guide generation
4. **Output**: `GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)`

### Key Features

✅ **Endpoint Conditioning**: Control start and end positions  
✅ **Grade Conditioning**: Generates routes matching target difficulty  
✅ **Pre-trained Knowledge**: Leverages GPT-2's sequence understanding  
✅ **Fast Training**: 2-3 hours on CPU, 20 minutes on GPU  

## Tips for Best Results

### Training
1. **Use quality-filtered data**: Balanced grades, high-quality routes
2. **Monitor validation loss**: Should be < 2.0 after 10 epochs
3. **Don't overtrain**: 10 epochs is usually enough
4. **Check grade balance**: Use `analyze_grade_distribution.py`

### Generation
1. **Temperature tuning**: 
   - 0.6-0.8 for realistic routes
   - 0.8-1.0 for balanced (default)
   - 1.0-1.2 for creative/diverse
2. **Specify endpoints**: Control start/end for better results
3. **Multiple samples**: Generate 5-10 to pick best
4. **Grade matching**: Model respects grade conditioning well

## Troubleshooting

### Model generates invalid paths
- Check training data quality
- Verify grade balance (use `analyze_grade_distribution.py`)
- Try lower temperature (0.6-0.7)

### Training loss not decreasing
- Ensure validation loss < 2.0 after 10 epochs
- Check data format (should be endpoint conditioning format)
- Verify CSV files are in `data_reordered/`

### Training too slow
- Reduce batch_size to 4-8 for CPU
- Reduce max_length to 128
- Reduce num_epochs to 5
- See TRAINING_SPEED_GUIDE.md

## Next Steps

1. **Use Gradio interface**: `start_web_interface.bat`
2. **Visualize paths**: `python visualize_path.py`
3. **Experiment with endpoints**: Try different start/end positions
4. **Adjust temperature**: Find the sweet spot for your use case
5. **Deploy**: Integrate with web frontend or API

## Model Advantages

**Why DistilGPT-2?**

| Feature | Benefit |
|---------|--------|
| Pre-trained | Understands sequences out of the box |
| Lightweight | 82M params (fast training/inference) |
| Text-based | Natural format for climbing paths |
| Endpoint conditioning | Control start/end positions |
| Fast training | 2-3 hours on CPU |

**Key principles**:
- Causal language modeling (next-token prediction)
- Endpoint conditioning (specify start/end)
- Grade conditioning (control difficulty)
- Text-based representation (easy to parse)

---

**You're now ready to generate climbing routes!** 🧗‍♂️
