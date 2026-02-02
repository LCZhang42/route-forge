# Climb Path Generator 🧗‍♂️

AI-powered MoonBoard climbing route generation system using an autoregressive transformer model. Generates realistic climbing paths of varying difficulty levels (6B to 8B+) with built-in reachability constraints and quality filtering.

## 🎯 Features

- ✅ **Autoregressive Transformer Model** - LegoACE-inspired architecture for sequential hold generation
- ✅ **Grade-Conditioned Generation** - Generate routes for specific difficulty levels (6B to 8B+)
- ✅ **Reachability Constraints** - Physics-based distance checks ensure climbable routes
- ✅ **Quality Filtering** - Train on high-quality, benchmark routes for better outputs
- ✅ **Interactive Web Interface** - Gradio-based UI for easy route generation and visualization
- ✅ **Vertical Progression Loss** - Ensures routes progress upward naturally
- ✅ **CPU & GPU Training** - Optimized for both CPU-only and GPU-accelerated training

## 🖼️ Gradio Web Interface

<!-- Screenshot placeholder - Add your Gradio app screenshot here -->
![Gradio Web Interface](docs/gradio_screenshot.png)
*Interactive web interface for generating and visualizing climbing routes*

## 📁 Project Structure

```
climb-path/
├── backend/
│   ├── data_processing/     # Data cleaning, analysis, and preparation
│   ├── models/              # Model architectures, tokenizer, dataset loaders
│   ├── training/            # Training, evaluation, and generation scripts
│   └── api/                 # FastAPI endpoints (optional)
├── data/                    # MoonBoard 2016 dataset (train/val/test splits)
├── docs/                    # Architecture documentation and research notes
├── frontend/                # React-based visualization (optional)
├── checkpoints/             # Saved model checkpoints
├── runs/                    # TensorBoard training logs
├── web_interface.py         # Gradio web app for route generation
└── *.md                     # Comprehensive documentation guides
```

## 📊 Dataset

**MoonBoard 2016** climbing problems dataset:
- **11,000+** climbing routes across all difficulty levels
- **Grades**: 6B, 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+, 8A, 8A+, 8B, 8B+
- **Hold positions**: (x, y) coordinates on 11×18 grid
- **Quality metrics**: Repeats, benchmark status, user ratings
- **Splits**: Train (80%), Validation (10%), Test (10%)

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_model.py
```

See [`INSTALL.md`](INSTALL.md) for detailed installation instructions.

### 2. Train the Model

#### CPU Training (Recommended for most users)
```bash
# Windows
train_cpu.bat

# Linux/Mac
bash train_cpu.sh
```

#### GPU Training
```bash
cd backend
python training/train_autoregressive.py --device cuda --batch_size 128
```

See [`CPU_TRAINING_GUIDE.md`](CPU_TRAINING_GUIDE.md) or [`QUICKSTART.md`](QUICKSTART.md) for detailed training instructions.

### 3. Generate Routes

#### Using Gradio Web Interface (Easiest)
```bash
# Windows
start_web_interface.bat

# Linux/Mac
python web_interface.py
```

Then open http://localhost:7860 in your browser.

#### Using Command Line
```bash
cd backend
python training/generate_paths.py --grade 7A --num_samples 5 --visualize
```

#### Using Python API
```python
from backend.models.tokenizer import ClimbPathTokenizer
from backend.models.climb_transformer import ClimbPathTransformerWithGeneration
import torch

# Load model
tokenizer = ClimbPathTokenizer()
model = ClimbPathTransformerWithGeneration(vocab_size=45)
checkpoint = torch.load('checkpoints/climb_path_cpu/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate route
grade_token = tokenizer.encode_grade('7A')
tokens = model.generate(grade_token=grade_token, tokenizer=tokenizer)
grade, holds = tokenizer.decode(tokens.cpu().numpy())
print(f"Generated {len(holds)} holds for grade {grade}")
```

### 4. Visualize Results

```bash
# Visualize training progress
tensorboard --logdir runs/climb_path_cpu

# Visualize generated paths
python visualize_path.py
```

See [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md) for more visualization options.

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [`INSTALL.md`](INSTALL.md) | Installation and setup instructions |
| [`QUICKSTART.md`](QUICKSTART.md) | Quick start guide for training and generation |
| [`CPU_TRAINING_GUIDE.md`](CPU_TRAINING_GUIDE.md) | CPU-optimized training guide |
| [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) | Windows-specific setup instructions |
| [`QUALITY_FILTERING_GUIDE.md`](QUALITY_FILTERING_GUIDE.md) | Training with quality-filtered data |
| [`REACHABILITY_LOSS_GUIDE.md`](REACHABILITY_LOSS_GUIDE.md) | Reachability constraint implementation |
| [`VERTICAL_PROGRESSION_TRAINING.md`](VERTICAL_PROGRESSION_TRAINING.md) | Vertical progression loss guide |
| [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md) | Visualization and evaluation tools |
| [`VALID_HOLDS_GUIDE.md`](VALID_HOLDS_GUIDE.md) | Valid hold constraints and filtering |

## 🏗️ Model Architecture

**Type**: Autoregressive Transformer (Decoder-only)

**Specifications**:
- Vocabulary: 45 tokens (BOS, 14 grades, 11 x-coords, 17 y-coords, EOS)
- Default size: ~2.5M parameters (d_model=256, 6 layers)
- Sequence format: `[BOS, grade, x1, y1, x2, y2, ..., EOS]`
- Max sequence length: 62 tokens (30 holds × 2 + 2)

**Key Components**:
1. **Tokenizer** - Encodes grades and coordinates into tokens
2. **Transformer** - Learns sequential patterns in climbing routes
3. **Logits Processors** - Enforces structural validity during generation
4. **Reachability Loss** - Penalizes unreachable hold distances
5. **Vertical Progression Loss** - Encourages upward movement

See [`docs/AUTOREGRESSIVE_MODEL.md`](docs/AUTOREGRESSIVE_MODEL.md) for architecture details.

## 🛠️ Tech Stack

**Machine Learning**:
- PyTorch 2.1+ (deep learning framework)
- Transformers (model utilities)
- TensorBoard (training visualization)

**Data Processing**:
- Pandas, NumPy (data manipulation)
- Scikit-learn (train/val/test splits)

**Web Interface**:
- Gradio (interactive UI)
- Matplotlib (route visualization)

**Optional**:
- FastAPI + Uvicorn (REST API)
- React + TailwindCSS (custom frontend)

## 🎓 Training Tips

1. **Start with CPU training** - Works well for initial experiments
2. **Use quality filtering** - Train on benchmark/high-repeat routes for better quality
3. **Monitor validation loss** - Stop when validation loss plateaus
4. **Adjust temperature** - Lower (0.5-0.7) for realistic, higher (1.2-1.5) for creative routes
5. **Enable reachability loss** - Improves physical plausibility of generated routes

## 📈 Results & Evaluation

Evaluate model performance:
```bash
# Evaluate on test set
python backend/training/evaluate_model.py

# Check for memorization
python backend/training/check_memorization.py

# Visualize generated vs real routes
python validate_and_visualize.py
```

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with different architectures
- Add new loss functions or constraints
- Improve the web interface
- Extend to other climbing wall types

## 📄 License

MIT

## 🙏 Acknowledgments

- **MoonBoard** for the climbing dataset
- **LegoACE** paper for autoregressive generation inspiration
- **Transformers library** for model utilities

---

**Ready to generate climbing routes?** Start with [`INSTALL.md`](INSTALL.md) → [`QUICKSTART.md`](QUICKSTART.md) → [`web_interface.py`](web_interface.py)
