# RxNExtract

A professional-grade system for extracting chemical reaction information from procedure texts using fine-tuned LLM with Dynamic prompting and self grounding.

[![PyPI version](https://badge.fury.io/py/rxnextract.svg)](https://badge.fury.io/py/rxnextract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/chemplusx/rxnextract)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/chemplusx/rxnextract)

## ✨ Key Features

- **Advanced AI**: Fine-tuned LLM with dynamic prompting and self-grounding
- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **Multiple Interfaces**: CLI, interactive mode, batch processing, and programmatic API
- **Memory Efficient**: 4-bit quantization support for deployment on various hardware
- **Comprehensive Analysis**: Error analysis, ablation studies, statistical testing, and uncertainty quantification
- **Easy Installation**: One-command installation via PyPI, or Docker

## 🚀 Quick Start

### 30-Second Demo

```bash
# Install
pip install rxnextract

# Use
python -c "
from chemistry_llm import ChemistryReactionExtractor
extractor = ChemistryReactionExtractor.from_pretrained('chemplusx/rxnextract')
procedure = 'Add 5g NaCl to 100mL water and stir for 30 minutes at room temperature.'
results = extractor.analyze_procedure(procedure)
print('Reactants:', results['extracted_data']['reactants'])
print('Conditions:', results['extracted_data']['conditions'])
"
```

### Try Without Installation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemplusx/RxNExtract/blob/master/examples/quickstart.ipynb)
[![Try on HuggingFace Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/chemplusx/rxnextract-demo)

## 📦 Installation Options

### Option 1: PyPI (Recommended)
```bash
pip install rxnextract                # Basic installation
pip install rxnextract[gpu]           # GPU support
pip install rxnextract[full]          # All features
```

### Option 2: Docker
```bash
docker pull chemplusx/rxnextract:latest
docker run -it --gpus all chemplusx/rxnextract:latest
```

### Option 3: From Source
```bash
git clone https://github.com/chemplusx/RxNExtract.git
cd RxNExtract
pip install -e .
```

## 🎯 Performance Highlights

Our complete framework achieves significant improvements over baseline methods:

| Metric | Baseline | RxNExtract | Improvement |
|--------|----------|------------|-------------|
| **Complete Reaction Accuracy** | 23.4% | **52.1%** | **+122.6%** |
| **Entity F1 Score** | 0.674 | **0.856** | **+27.0%** |
| **Role Classification Accuracy** | 68.2% | **85.9%** | **+25.9%** |
| **Condition F1 Score** | 0.421 | **0.689** | **+63.7%** |

**Error Reduction**: 47.8-55.2% across all major error categories
**Statistical Significance**: McNemar's χ² = 134.67 (p < 0.001), Cohen's d = 0.82

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Installation & Setup Guide](docs/INSTALLATION.md)** | Detailed installation instructions, system requirements, and configuration |
| **[Usage Guide & Examples](docs/USAGE.md)** | Comprehensive usage examples, API reference, and advanced features |
| **[Analysis & Evaluation](docs/ANALYSIS.md)** | Complete analysis framework, metrics, and research reproducibility |
| **[Changelog](docs/CHANGELOG.md)** | Version history and release notes |

## 🔬 Research Applications

Perfect for:
- **Chemical Literature Mining**: Extract structured reaction data from papers
- **Procedure Standardization**: Convert natural language to structured formats
- **Database Curation**: Automated reaction database construction
- **Educational Tools**: Teaching reaction analysis and extraction
- **Research Reproducibility**: Systematic evaluation of extraction methods

## 🤝 Community & Support

### Getting Help
- 📚 **Documentation**: [docs.rxnextract.org](https://docs.rxnextract.org)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/chemplusx/RxNExtract/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/chemplusx/RxNExtract/discussions)
- 📧 **Email**: support@rxnextract.org

### For Experimental Chemists
- 🎯 **One-click installations** via PyPI and docker
- 🐳 **Docker containers** for consistent environments
- 📖 **User-friendly tutorials** and examples
- 🎓 **Video tutorials** and webinars

### For Developers
- 🔧 **Extensive API documentation**
- 🧪 **Comprehensive test suite**
- 🏗️ **Modular architecture** for easy extension
- 📋 **Contributing guidelines** and code standards

## 🔑 Quick Examples

### Basic Usage
```python
from chemistry_llm import ChemistryReactionExtractor

# Initialize extractor
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")

# Analyze procedure
procedure = """
Dissolve 5.0 g of benzoic acid in 100 mL of hot water.
Add 10 mL of concentrated HCl and cool the solution.
Filter the precipitated product and wash with cold water.
"""

results = extractor.analyze_procedure(procedure)
print(results['extracted_data'])
```

### Command Line Interface
```bash
# Interactive mode
rxnextract --interactive

# Batch processing
rxnextract --input procedures.txt --output results.json

# Single procedure
rxnextract --procedure "Add 2g NaCl to 50mL water"
```

### Analysis & Research
```python
from chemistry_llm.analysis import ErrorAnalyzer, AblationStudy

# Error analysis
analyzer = ErrorAnalyzer()
error_results = analyzer.analyze_prediction_errors(predictions, ground_truth)

# Ablation study
ablation = AblationStudy(model_path="./model")
study_results = ablation.run_complete_study(test_data, ground_truth)
```

## 🏗️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.9+ |
| **RAM** | 8GB | 16GB+ |
| **GPU Memory** | 4GB | 12GB+ |
| **Storage** | 20GB | 50GB+ |
| **CPU** | 4 cores | 8+ cores |

*Note: Requirements are for inference only. Fine-tuning requires additional resources.*

## 📊 Data and Software Availability

**Code Repository**: All code used in this study is available under the MIT License at https://github.com/chemplusx/RxNExtract. The MIT License permits unrestricted use, modification, and distribution, making it suitable for both academic research and commercial applications.

**Package Distribution**:
- PyPI: `pip install rxnextract` 
- Docker Hub: `docker pull chemplusx/rxnextract:latest`

**Datasets**: Training and evaluation datasets are available at [Zenodo DOI: 10.5281/zenodo.XXXXXX](link-to-zenodo)

**Reproducibility**: Complete analysis scripts and configuration files are provided to reproduce all results presented in the paper.

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing use cases, your help makes RxNExtract better.

### Quick Contributing Guide
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python -m pytest`)
5. Submit a pull request

See our [Contributing Guidelines](CONTRIBUTING.md) for detailed instructions.

## 📄 License & Citation

**License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete terms.

**Citation**: If you use RxNExtract in your research, please cite our paper:
```bibtex
@article{rxnextract2025,
  title={RxNExtract: A Professional-Grade System for Chemical Reaction Extraction using Fine-tuned LLMs},
  author={[Your Authors]},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]}
}
```

## 🔗 Links

- **Homepage**: https://github.com/chemplusx/RxNExtract
- **Documentation**: https://docs.rxnextract.org
- **PyPI Package**: https://pypi.org/project/rxnextract/
- **Docker Images**: https://hub.docker.com/r/chemplusx/rxnextract
- **HuggingFace Space**: https://huggingface.co/spaces/chemplusx/rxnextract
- **Paper**: [Link to published paper]

---
