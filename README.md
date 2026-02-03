# Climb Path Generator 🧗‍♂️

AI-powered MoonBoard climbing route generation system using fine-tuned DistilGPT-2. Generates realistic climbing paths of varying difficulty levels (6B to 8B+) with endpoint conditioning and quality filtering.

## 🎯 Features

- ✅ **Fine-tuned DistilGPT-2** - Lightweight pre-trained language model (82M parameters)
- ✅ **Endpoint Conditioning** - Specify grade, start hold, and end hold to generate intermediate holds
- ✅ **Grade-Conditioned Generation** - Generate routes for specific difficulty levels (6B+ to 7C+)
- ✅ **Quality Filtering** - Train on high-quality, balanced routes for better outputs
- ✅ **Interactive Web Interface** - Gradio-based UI for easy route generation and visualization
- ✅ **CPU & GPU Training** - Works on CPU (2-3 hours) or GPU (20 minutes)

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
- **6,610** quality-filtered training routes (balanced across grades)
- **Grades**: 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+ (rare grades removed)
- **Hold positions**: (x, y) coordinates on 11×18 grid
- **Quality metrics**: Filtered by repeats and quality scores
- **Format**: Grade + Start Hold + End Hold → Intermediate Holds

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

```bash
# Windows
train_distilgpt2.bat

# Linux/Mac
python backend/training/finetune_huggingface.py
```

**Training time**: ~2-3 hours on CPU, ~20 minutes on GPU

See [`QUICKSTART.md`](QUICKSTART.md) for detailed training instructions.

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
# Windows
generate_paths.bat

# Linux/Mac
python backend/training/generate_with_gpt2.py --grade 7A --start_hold 0,4 --end_hold 8,17 --num_samples 5
```

#### Using Python API
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model
model = GPT2LMHeadModel.from_pretrained('checkpoints/distilgpt2_climb/final')
tokenizer = GPT2Tokenizer.from_pretrained('checkpoints/distilgpt2_climb/final')
model.eval()

# Generate route
prompt = "GRADE: 7A | START: (0,4) | END: (8,17) | MID:"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=256, temperature=0.8)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 4. Visualize Results

```bash
# Visualize training progress
tensorboard --logdir checkpoints/distilgpt2_climb/logs

# Visualize generated paths
python visualize_path.py
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [`INSTALL.md`](INSTALL.md) | Installation and setup instructions |
| [`QUICKSTART.md`](QUICKSTART.md) | Quick start guide for training and generation |
| [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) | Windows-specific setup instructions |
| [`WORKFLOW.md`](WORKFLOW.md) | Complete workflow from data to deployment |
| [`ENDPOINT_CONDITIONING.md`](ENDPOINT_CONDITIONING.md) | Endpoint conditioning explanation |
| [`GRADE_BALANCE_GUIDE.md`](GRADE_BALANCE_GUIDE.md) | Grade balance and filtering guide |
| [`QUALITY_FILTERING_GUIDE.md`](QUALITY_FILTERING_GUIDE.md) | Training with quality-filtered data |

## 🏗️ Model Architecture

**Type**: Fine-tuned DistilGPT-2 (Causal Language Model)

**Specifications**:
- Base model: DistilGPT-2 (82M parameters, 6 layers, 768 hidden size)
- Vocabulary: ~50k tokens (GPT-2 tokenizer)
- Input format: `GRADE: 7A | START: (0,4) | END: (8,17) | MID:`
- Output format: `GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)`
- Max sequence length: 256 tokens

**Key Features**:
1. **Endpoint Conditioning** - Specify start and end holds, model generates intermediate
2. **Pre-trained Knowledge** - Leverages GPT-2's sequence understanding
3. **Fast Training** - 2-3 hours on CPU, 20 minutes on GPU
4. **Lightweight** - 82M parameters (33% smaller than GPT-2)
5. **Text-based** - Natural language format for climbing paths

## 🛠️ Tech Stack

**Machine Learning**:
- PyTorch 2.1+ (deep learning framework)
- HuggingFace Transformers (DistilGPT-2)
- Datasets (data loading)
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

1. **Use quality-filtered data** - Train on balanced, high-quality routes
2. **Monitor validation loss** - Should be < 2.0 after 10 epochs
3. **Adjust temperature** - Lower (0.6-0.8) for realistic, higher (1.0-1.2) for creative routes
4. **Specify endpoints** - Control start/end positions for better results
5. **Check grade balance** - Use `analyze_grade_distribution.py` to verify data balance

## 📈 Results & Evaluation

Evaluate model performance:
```bash
# Generate sample paths
generate_paths.bat

# Visualize paths
python visualize_path.py

# Check grade distribution
python analyze_grade_distribution.py
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
- **HuggingFace** for the Transformers library and DistilGPT-2
- **OpenAI** for the GPT-2 architecture

---

**Ready to generate climbing routes?** Start with [`INSTALL.md`](INSTALL.md) → [`QUICKSTART.md`](QUICKSTART.md) → [`web_interface.py`](web_interface.py)
