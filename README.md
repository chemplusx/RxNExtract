# RxnExtract

A professional-grade system for extracting chemical reaction information from procedure texts using fine-tuned Large Language Models with optimized Chain-of-Thought prompting.

## 🚀 Features

- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **Chain-of-Thought Prompting**: Advanced prompting strategies for better extraction accuracy
- **Multiple Interfaces**: CLI, interactive mode, batch processing, and programmatic API
- **Memory Efficient**: 4-bit quantization support for deployment on various hardware
- **Robust Parsing**: Error-tolerant XML parsing with structured output
- **Professional Logging**: Comprehensive logging with configurable levels
- **Extensible Design**: Easy to customize prompts and add new extraction features

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU inference)

### Method 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/chemistry-llm-inference.git
cd chemistry-llm-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Method 2: Development Setup

```bash
# Clone and setup for development
git clone https://github.com/your-org/chemistry-llm-inference.git
cd chemistry-llm-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU Memory | 4GB | 12GB+ |
| Storage | 20GB | 50GB+ |
| CPU | 4 cores | 8+ cores |

## 🚀 Quick Start

### 1. Prepare Your Model

Ensure you have a fine-tuned model directory with the following structure:
```
your-model-path/
├── adapter_config.json
├── adapter_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### 2. Basic Usage

```python
from chemistry_llm import ChemistryReactionExtractor

# Initialize the extractor
extractor = ChemistryReactionExtractor(
    model_path="path/to/your/fine-tuned-model"
)

# Extract reaction information
procedure = """
Add 2.5 g of benzoic acid to 50 mL of ethanol. 
Heat the mixture to reflux for 4 hours.
Cool and filter to obtain the product.
"""

results = extractor.analyze_procedure(procedure)
print(results['extracted_data'])
```

### 3. Command Line Interface

```bash
# Interactive mode
chemistry-llm --model-path ./model --interactive

# Batch processing
chemistry-llm --model-path ./model --input procedures.txt --output results.json

# Single procedure
chemistry-llm --model-path ./model --procedure "Your procedure text here"
```

## 📖 Usage

### Interactive Mode

Start an interactive session for real-time procedure analysis:

```bash
python scripts/run_interactive.py --model-path ./your-model-path
```

Features:
- Real-time procedure input
- Formatted output display
- Error handling and recovery
- Session history

### Batch Processing

Process multiple procedures from a file:

```bash
python scripts/run_batch.py \
    --model-path ./your-model-path \
    --input-file procedures.txt \
    --output-file results.json \
    --batch-size 10
```

Input file format (one procedure per line):
```
Add 5g NaCl to 100mL water and stir for 30 minutes.
Reflux the mixture of benzene and AlCl3 for 2 hours at 80°C.
```

### Programmatic Usage

```python
from chemistry_llm import ChemistryReactionExtractor
from chemistry_llm.utils import setup_logging

# Setup logging
setup_logging(level="INFO")

# Initialize extractor with custom config
extractor = ChemistryReactionExtractor(
    model_path="./model",
    device="cuda",
    max_length=512,
    temperature=0.1
)

# Analyze multiple procedures
procedures = [
    "Mix 10g of compound A with 20mL solvent B...",
    "Heat the reaction mixture to 150°C for 3 hours..."
]

results = []
for procedure in procedures:
    result = extractor.analyze_procedure(procedure)
    results.append(result)

# Access structured data
for result in results:
    data = result['extracted_data']
    print(f"Reactants: {len(data['reactants'])}")
    print(f"Products: {len(data['products'])}")
```

## 🔧 Configuration

### config/config.yaml

```yaml
model:
  default_temperature: 0.1
  default_top_p: 0.95
  max_new_tokens: 512
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "float16"

prompts:
  use_cot: true
  cot_steps:
    - "Identify Reactants"
    - "Identify Reagents" 
    - "Identify Solvents"
    - "Identify Conditions"
    - "Identify Workup Steps"
    - "Identify Products"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

output:
  include_raw: false
  include_confidence: false
  xml_pretty_print: true
```

### Environment Variables

```bash
# Optional environment variables
export CHEMISTRY_LLM_MODEL_PATH="/path/to/model"
export CHEMISTRY_LLM_DEVICE="cuda"
export CHEMISTRY_LLM_LOG_LEVEL="INFO"
```

## 📚 API Reference

### ChemistryReactionExtractor

Main class for reaction extraction.

#### Methods

##### `__init__(model_path, base_model_name=None, device="auto", config=None)`

Initialize the extractor.

**Parameters:**
- `model_path` (str): Path to fine-tuned model directory
- `base_model_name` (str, optional): Base model name (auto-detected if None)
- `device` (str): Device for inference ("auto", "cpu", "cuda")
- `config` (dict, optional): Custom configuration

##### `analyze_procedure(procedure_text, return_raw=False)`

Analyze a chemical procedure text.

**Parameters:**
- `procedure_text` (str): The procedure to analyze
- `return_raw` (bool): Include raw model output

**Returns:**
- `dict`: Analysis results with extracted data

##### `extract_reaction(procedure_text, **kwargs)`

Low-level extraction method.

**Parameters:**
- `procedure_text` (str): Procedure text
- `**kwargs`: Generation parameters

**Returns:**
- `str`: Raw model output

### Utility Functions

#### `chemistry_llm.utils.xml_parser`

- `parse_reaction_xml(xml_text)`: Parse XML to structured data
- `validate_xml_structure(xml_text)`: Validate XML format

#### `chemistry_llm.utils.device_utils`

- `get_optimal_device()`: Auto-detect best available device
- `get_memory_info()`: Get system memory information

## 🎯 Examples

### Example 1: Basic Extraction

```python
from chemistry_llm import ChemistryReactionExtractor

extractor = ChemistryReactionExtractor("./model")

procedure = """
Dissolve 5.0 g of benzoic acid in 100 mL of hot water.
Add 10 mL of concentrated HCl and cool the solution.
Filter the precipitated product and wash with cold water.
Dry to obtain 4.2 g of product (84% yield).
"""

results = extractor.analyze_procedure(procedure)

# Access extracted components
data = results['extracted_data']
print("Reactants:", data['reactants'])
print("Reagents:", data['reagents'])
print("Products:", data['products'])
```

### Example 2: Custom Prompting

```python
from chemistry_llm.core import PromptBuilder

# Create custom prompt builder
prompt_builder = PromptBuilder(
    use_cot=True,
    custom_steps=[
        "Identify starting materials",
        "Identify reaction conditions", 
        "Identify final products"
    ]
)

extractor = ChemistryReactionExtractor(
    "./model", 
    prompt_builder=prompt_builder
)
```

### Example 3: Batch Processing with Progress

```python
from chemistry_llm import ChemistryReactionExtractor
from tqdm import tqdm
import json

extractor = ChemistryReactionExtractor("./model")

# Load procedures
with open("procedures.txt", "r") as f:
    procedures = [line.strip() for line in f if line.strip()]

# Process with progress bar
results = []
for procedure in tqdm(procedures, desc="Processing"):
    try:
        result = extractor.analyze_procedure(procedure)
        results.append(result)
    except Exception as e:
        results.append({"error": str(e), "procedure": procedure})

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_extractor.py

# Run with coverage
python -m pytest tests/ --cov=src/chemistry_llm --cov-report=html
```

### Test Structure

```
tests/
├── test_extractor.py          # Core extraction functionality
├── test_xml_parser.py         # XML parsing utilities
├── test_prompt_builder.py     # Prompt construction
├── test_integration.py        # End-to-end tests
└── fixtures/
    ├── sample_procedures.txt  # Test procedures
    └── expected_outputs.json  # Expected results
```

## 🛠️ Development

### Code Style

This project follows PEP 8 and uses:
- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Release Process

1. Update version in `setup.py` and `src/chemistry_llm/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag (`git tag v1.0.0`)
4. Push tag (`git push origin v1.0.0`)
5. GitHub Actions will automatically build and publish

## 📝 Changelog

### v1.0.0 (2025-05-21)
- Initial release
- Core extraction functionality
- Chain-of-Thought prompting
- XML parsing and validation
- CLI interface
- Comprehensive test suite

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Transformers library by Hugging Face
- PEFT library for efficient fine-tuning
- BitsAndBytes for quantization support

---

**Made with ❤️ for the Chemistry AI Community**
