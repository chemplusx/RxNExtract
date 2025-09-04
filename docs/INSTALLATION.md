# Installation & Setup Guide

Complete installation instructions for RxNExtract across different platforms and use cases.

## 🚀 Quick Installation

### Recommended: PyPI Installation
```bash
pip install rxnextract
```

That's it! For most users, this one command provides everything needed to start extracting chemical reaction information.

## 📋 System Requirements

### Minimum Requirements
| Component | Specification |
|-----------|---------------|
| **Operating System** | Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10+ |
| **Python** | 3.8 or higher |
| **RAM** | 8GB |
| **Storage** | 20GB available space |
| **CPU** | 4 cores, x64 architecture |
| **Internet** | Required for model downloads |

### Recommended Requirements
| Component | Specification |
|-----------|---------------|
| **Python** | 3.9+ |
| **RAM** | 16GB+ |
| **GPU** | CUDA-compatible GPU with 12GB+ VRAM |
| **Storage** | 50GB+ SSD storage |
| **CPU** | 8+ cores |

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU** with CUDA Compute Capability 6.0+
- **CUDA** version 11.0+ 
- **cuDNN** version 8.0+
- **GPU Memory**: Minimum 4GB, Recommended 12GB+

*Note: All requirements are for inference only. Model fine-tuning requires significantly more resources.*

## 📦 Installation Methods

### Method 1: PyPI Installation (Recommended)

#### Basic Installation
```bash
pip install rxnextract
```

#### Installation Variants
```bash
# CPU-only version (smaller download)
pip install rxnextract[cpu]

# GPU-accelerated version
pip install rxnextract[gpu]

# Complete installation with all optional dependencies
pip install rxnextract[full]

# Development installation with testing tools
pip install rxnextract[dev]

# Documentation building tools
pip install rxnextract[docs]
```

#### Virtual Environment Setup (Recommended)
```bash
# Create virtual environment
python -m venv rxnextract-env

# Activate virtual environment
# Linux/macOS:
source rxnextract-env/bin/activate
# Windows:
rxnextract-env\Scripts\activate

# Install RxNExtract
pip install rxnextract[full]
```

### Method 2: Conda Installation

#### From Conda-Forge
```bash
# Basic installation
conda install -c conda-forge rxnextract

# With specific Python version
conda create -n rxnextract python=3.9
conda activate rxnextract
conda install -c conda-forge rxnextract

# GPU support
conda install -c conda-forge -c nvidia rxnextract pytorch-gpu
```

#### Using Environment File
Create `environment.yml`:
```yaml
name: rxnextract
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.9
  - rxnextract
  - pytorch-gpu
  - jupyter
  - matplotlib
  - seaborn
```

Install:
```bash
conda env create -f environment.yml
conda activate rxnextract
```

### Method 3: Docker Installation

#### Quick Start with Docker
```bash
# Pull latest stable image
docker pull chemplusx/rxnextract:latest

# Run interactive mode
docker run -it --gpus all chemplusx/rxnextract:latest

# Run with mounted data directory
docker run -v /path/to/your/data:/app/data \
  chemplusx/rxnextract:latest \
  --input /app/data/procedures.txt \
  --output /app/data/results.json
```

#### Docker Variants
```bash
# CPU-only version (smaller image)
docker pull chemplusx/rxnextract:latest-cpu

# GPU-accelerated version
docker pull chemplusx/rxnextract:latest-gpu

# Development version with additional tools
docker pull chemplusx/rxnextract:dev

# Specific version
docker pull chemplusx/rxnextract:v1.2.0
```

#### Docker Compose Setup
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  rxnextract:
    image: chemplusx/rxnextract:latest
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run:
```bash
docker-compose up
```

### Method 4: From Source (Development)

#### Clone and Install
```bash
# Clone repository
git clone https://github.com/chemplusx/RxNExtract.git
cd RxNExtract

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .
```

#### Development Setup with Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

## 🤖 Model Setup

### Automatic Model Download (Recommended)
Models are automatically downloaded on first use:
```python
from chemistry_llm import ChemistryReactionExtractor

# This will automatically download the default model
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")
```

### Pre-download Models
```bash
# Download default model
python -c "from chemistry_llm import ChemistryReactionExtractor; ChemistryReactionExtractor.download_default_model()"

# Download specific model
python -c "from chemistry_llm import ChemistryReactionExtractor; ChemistryReactionExtractor.from_pretrained('chemplusx/rxnextract-base')"
```

### HuggingFace Integration

#### Available Models
| Model | Description | Size | Performance |
|-------|-------------|------|-------------|
| `chemplusx/rxnextract-base` | Base fine-tuned model | 7B params | Good |
| `chemplusx/rxnextract-complete` | Complete framework model | 7B params | Best |
| `chemplusx/rxnextract-fast` | Optimized for speed | 1B params | Fast |

#### Manual Model Download
```bash
# Using git lfs
git lfs install
git clone https://huggingface.co/chemplusx/rxnextract-complete

# Using HuggingFace Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='chemplusx/rxnextract-complete', local_dir='./models/complete')
"
```

### Local Model Setup
If you have your own fine-tuned model:
```
your-model-path/
├── adapter_config.json
├── adapter_model.bin  # or adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.model  # if using SentencePiece
```

## ⚙️ Configuration

### Environment Variables
```bash
# Optional environment variables
export RXNEXTRACT_MODEL_PATH="/path/to/your/model"
export RXNEXTRACT_CACHE_DIR="./cache"
export RXNEXTRACT_DEVICE="cuda"  # or "cpu", "auto"
export RXNEXTRACT_LOG_LEVEL="INFO"
export HUGGINGFACE_HUB_CACHE="./hf_cache"
```

### Configuration File
Create `config/config.yaml`:
```yaml
model:
  default_model: "chemplusx/rxnextract-complete"
  device: "auto"  # auto, cpu, cuda
  max_length: 512
  temperature: 0.1
  top_p: 0.95
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "float16"

paths:
  model_cache: "./cache/models"
  output_dir: "./outputs"
  log_dir: "./logs"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: true

analysis:
  error_analysis:
    include_cot_failures: true
    categorize_by_complexity: true
  
  ablation_study:
    sample_size: 1000
    stratified_sampling: true
  
  statistical_analysis:
    significance_level: 0.05
    confidence_level: 0.95
    bootstrap_iterations: 1000

output:
  format: "json"  # json, xml, yaml
  include_raw: false
  include_confidence: true
  pretty_print: true
```

## 🔧 Hardware Optimization

### GPU Setup

#### NVIDIA GPU Optimization
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Install CUDA-specific PyTorch (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Optimization
```python
from chemistry_llm import ChemistryReactionExtractor

# Use 4-bit quantization for lower memory usage
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    load_in_4bit=True,
    device="cuda"
)

# Or use CPU if GPU memory is insufficient
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    device="cpu"
)
```

### CPU Optimization
```python
import torch

# Set number of CPU threads
torch.set_num_threads(8)

# Use CPU-optimized model loading
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    device="cpu",
    torch_dtype=torch.float32  # Use float32 for better CPU performance
)
```

## ✅ Verification

### Test Installation
```bash
# Basic functionality test
python -c "
from chemistry_llm import ChemistryReactionExtractor
print('✓ RxNExtract imported successfully')

# Test model loading
try:
    extractor = ChemistryReactionExtractor.from_pretrained('chemplusx/rxnextract-complete')
    print('✓ Model loaded successfully')
except Exception as e:
    print(f'✗ Model loading failed: {e}')
"
```

### Run Example Extraction
```python
from chemistry_llm import ChemistryReactionExtractor

# Test basic extraction
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")
procedure = "Add 5g NaCl to 100mL water and stir for 30 minutes."

try:
    result = extractor.analyze_procedure(procedure)
    print("✓ Extraction test passed")
    print(f"Found {len(result['extracted_data']['reactants'])} reactants")
except Exception as e:
    print(f"✗ Extraction test failed: {e}")
```

### Run Test Suite
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run basic tests
python -m pytest tests/test_basic.py -v

# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=chemistry_llm --cov-report=html
```

## 🚨 Troubleshooting

### Common Issues

#### Issue: CUDA out of memory
**Solution**:
```python
# Use 4-bit quantization
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    load_in_4bit=True
)

# Or use CPU
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    device="cpu"
)
```

#### Issue: Model download fails
**Solution**:
```bash
# Set HuggingFace cache directory
export HF_HOME="./hf_cache"

# Use manual download
huggingface-cli download chemplusx/rxnextract-complete

# Check internet connection and try again
ping huggingface.co
```

#### Issue: Import errors
**Solution**:
```bash
# Ensure all dependencies are installed
pip install --upgrade rxnextract[full]

# Check Python version
python --version  # Should be 3.8+

# Check installation
pip show rxnextract
```

#### Issue: Slow performance on CPU
**Solution**:
```python
import torch

# Optimize CPU performance
torch.set_num_threads(min(8, torch.get_num_threads()))

# Use smaller model for faster inference
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-fast"
)
```

### Platform-Specific Issues

#### Windows-Specific
```bash
# Use long path support
git config --global core.longpaths true

# Install Visual C++ redistributables if needed
# Download from Microsoft website

# Use Anaconda on Windows for easier dependency management
conda install -c conda-forge rxnextract
```

#### macOS-Specific
```bash
# Install Xcode command line tools
xcode-select --install

# Use Homebrew for dependencies
brew install python@3.9

# For Apple Silicon Macs, ensure compatible PyTorch
pip install torch torchvision torchaudio
```

#### Linux-Specific
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# For GPU support
sudo apt-get install nvidia-driver-470 nvidia-cuda-toolkit
```

### Getting Help

If you encounter issues not covered here:

1. **Check the FAQ**: [docs/FAQ.md](FAQ.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/chemplusx/RxNExtract/issues)
3. **Create a new issue**: Include your system info, Python version, and error messages
4. **Join discussions**: [GitHub Discussions](https://github.com/chemplusx/RxNExtract/discussions)
5. **Contact support**: support@rxnextract.org

## 🔄 Updating

### Update via PyPI
```bash
pip install --upgrade rxnextract
```

### Update via Conda
```bash
conda update -c conda-forge rxnextract
```

### Update Docker Images
```bash
docker pull chemplusx/rxnextract:latest
```

### Update from Source
```bash
cd RxNExtract
git pull origin main
pip install -e .
```

## 🔒 Security Considerations

- **Model files**: Downloaded models are cached locally and verified for integrity
- **Network access**: Required only for initial model download
- **Data privacy**: All processing is done locally; no data is sent to external servers
- **Dependencies**: All dependencies are from trusted sources (PyPI, Conda-Forge)

For enterprise deployments, consider:
- Using local model storage
- Network-isolated environments
- Custom Docker images with pre-downloaded models

---

**Next Steps**: After installation, see the [Usage Guide](USAGE.md) for detailed examples and API documentation.