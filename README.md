# RxnExtract

A professional-grade system for extracting chemical reaction information from procedure texts using fine-tuned LLM with Dynamic prompting and self grounding.

## 🚀 Features

- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **Dynamic Prompting**: Advanced dynamic prompt selection for better extraction accuracy
- **Multiple Interfaces**: CLI, interactive mode, batch processing, and programmatic API
- **Memory Efficient**: 4-bit quantization support for deployment on various hardware
- **Robust Parsing**: Error-tolerant XML parsing with structured output
- **Professional Logging**: Comprehensive logging with configurable levels
- **Extensible Design**: Easy to customize prompts and add new extraction features
- **Comprehensive Analysis Suite**: Error analysis, ablation studies, statistical testing, and uncertainty quantification

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Analysis and Evaluation](#analysis-and-evaluation)
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
git clone https://github.com/chemplusx/RxNExtract.git
cd RxNExtract

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Method 2: Development Setup

```bash
# Clone and setup for development
git clone https://github.com/your-org/RxNExtract.git
cd RxNExtract

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

```
Please note: The above requirements are for inferencing and not fine-tuning the LLM
```

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

## 📊 Analysis and Evaluation

The framework includes comprehensive analysis modules for research-grade evaluation:

### Error Analysis

Analyze extraction errors systematically across different categories:

```python
from chemistry_llm.analysis import ErrorAnalyzer

# Initialize error analyzer
error_analyzer = ErrorAnalyzer()

# Analyze prediction errors
error_results = error_analyzer.analyze_prediction_errors(
    predictions=model_predictions,
    ground_truth=ground_truth_data,
    method_name="baseline"
)

# Compare methods
method_results = {
    'baseline': baseline_error_results,
    'improved': improved_error_results
}

error_comparisons = error_analyzer.compare_methods(method_results)

# Generate error report
report = error_analyzer.generate_error_report(error_results, "error_analysis.txt")
print(report)
```

**Error Categories Analyzed:**
- **Entity Recognition**: Missing entities, false positives, incorrect entity types
- **Role Classification**: Reactant/product confusion, catalyst misidentification, solvent misclassification
- **Condition Extraction**: Missing temperature/time/catalyst, incomplete procedures
- **CoT Reasoning**: Implicit condition interpretation, generic entity handling, multi-step confusion

### Ablation Study

Systematic component-level performance analysis:

```python
from chemistry_llm.analysis import AblationStudy

# Initialize ablation study
ablation = AblationStudy(model_path="./model")

# Run complete ablation study
study_results = ablation.run_complete_study(
    test_data=test_procedures,
    ground_truth=ground_truth,
    sample_size=1000,
    stratified=True  # Stratify by reaction complexity
)

# Generate comprehensive report
report = ablation.generate_ablation_report(study_results, "ablation_report.txt")

# Export results to CSV
df = ablation.export_results_to_csv(study_results, "ablation_results.csv")
```

**Ablation Configurations:**
- Direct Extraction (baseline)
- Structured Output
- Meta Prompt
- Chain-of-Thought
- CoT + Reflection
- Self-Grounding
- Complete Framework
- Iterative Refinement

**Metrics Calculated:**
- Complete Reaction Accuracy (CRA)
- Entity-level F1
- Role Classification Accuracy (RCA)
- Condition Extraction F1
- Inference Time
- Performance by complexity level

### Uncertainty Quantification

Confidence calibration and uncertainty analysis:

```python
from chemistry_llm.analysis import UncertaintyQuantifier

# Initialize uncertainty quantifier
uncertainty = UncertaintyQuantifier()

# Calculate calibration metrics
calibration_metrics = uncertainty.calculate_calibration_metrics(
    confidences=model_confidences,
    accuracies=binary_accuracies
)

print(f"Expected Calibration Error: {calibration_metrics.ece:.4f}")
print(f"Brier Score: {calibration_metrics.brier_score:.4f}")

# Perform temperature scaling
calibrated_probs, optimal_temp = uncertainty.perform_temperature_scaling(
    validation_logits=val_logits,
    validation_labels=val_labels,
    test_logits=test_logits
)

# Analyze confidence-stratified performance
confidence_analysis = uncertainty.analyze_confidence_stratified_performance(
    confidences=model_confidences,
    accuracies=binary_accuracies,
    n_strata=5
)

# Generate reliability diagram
fig = uncertainty.generate_reliability_diagram(
    confidences=model_confidences,
    accuracies=binary_accuracies,
    save_path="reliability_diagram.png"
)
```

**Uncertainty Features:**
- Expected Calibration Error (ECE)
- Brier Score decomposition
- Temperature scaling
- Platt scaling
- Isotonic regression
- Confidence-stratified analysis
- Reliability diagrams

### Statistical Analysis

Comprehensive statistical testing and significance analysis:

```python
from chemistry_llm.analysis import StatisticalAnalyzer

# Initialize statistical analyzer
stats_analyzer = StatisticalAnalyzer()

# Pairwise method comparison
comparison = stats_analyzer.perform_pairwise_comparison(
    method1_results=baseline_results,
    method2_results=improved_results,
    method1_name="Baseline",
    method2_name="Complete Framework",
    test_type="paired_t"
)

print(f"p-value: {comparison['p_value']:.6f}")
print(f"Effect size (Cohen's d): {comparison['effect_size']:.3f}")
print(f"Significant: {comparison['significant']}")

# McNemar's test for classification comparison
mcnemar_result = stats_analyzer.perform_mcnemar_test(
    method1_correct=baseline_correct,
    method2_correct=improved_correct,
    method1_name="Baseline",
    method2_name="Improved"
)

# ANOVA with post-hoc tests
groups = {
    'Method A': results_a,
    'Method B': results_b,
    'Method C': results_c
}

anova_results = stats_analyzer.perform_anova(groups, post_hoc=True)

# Baseline reproducibility analysis
reproducibility = stats_analyzer.calculate_baseline_reproducibility(
    literature_results={'ChemRxnBERT': 0.789, 'GPT-3.5': 0.641},
    reproduced_results={'ChemRxnBERT': [0.782, 0.785, 0.779], 'GPT-3.5': [0.634, 0.637, 0.631]}
)

# Generate statistical report
report = stats_analyzer.generate_statistical_report(
    {
        'pairwise_comparisons': {'baseline_vs_improved': comparison},
        'mcnemar_tests': {'classification_comparison': mcnemar_result},
        'anova': anova_results,
        'reproducibility': reproducibility
    },
    output_file="statistical_analysis.txt"
)
```

**Statistical Tests Available:**
- Paired t-test
- Wilcoxon signed-rank test
- Mann-Whitney U test
- McNemar's test
- One-way ANOVA with post-hoc
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, eta-squared)

### Metrics Calculator

Comprehensive performance metrics calculation:

```python
from chemistry_llm.analysis import MetricsCalculator

# Initialize metrics calculator
metrics_calc = MetricsCalculator()

# Calculate comprehensive metrics
metrics = metrics_calc.calculate_comprehensive_metrics(
    predictions=model_predictions,
    ground_truth=ground_truth_data
)

print(f"Complete Reaction Accuracy: {metrics['complete_reaction_accuracy']:.3f}")
print(f"Entity F1: {metrics['entity_f1']:.3f}")
print(f"Role Classification Accuracy: {metrics['role_classification_accuracy']:.3f}")

# Performance by complexity
complexity_labels = ['simple', 'moderate', 'complex'] * (len(predictions) // 3)
complexity_metrics = metrics_calc.analyze_performance_by_complexity(
    predictions=model_predictions,
    ground_truth=ground_truth_data,
    complexity_labels=complexity_labels
)

# Calculate error reduction
error_reduction = metrics_calc.calculate_error_reduction(
    baseline_metrics=baseline_metrics,
    improved_metrics=improved_metrics
)

# Export metrics summary
metrics_calc.export_metrics_summary(metrics, "metrics_summary.json")
```

### Running Complete Analysis Pipeline

Example of running the complete analysis pipeline:

```python
from chemistry_llm.analysis import (
    ErrorAnalyzer, AblationStudy, UncertaintyQuantifier, 
    StatisticalAnalyzer, MetricsCalculator
)

def run_complete_analysis(model_path, test_data, ground_truth):
    """Run complete analysis pipeline"""
    
    # 1. Error Analysis
    print("Running error analysis...")
    error_analyzer = ErrorAnalyzer()
    error_results = error_analyzer.analyze_prediction_errors(
        predictions, ground_truth, "complete_framework"
    )
    
    # 2. Ablation Study
    print("Running ablation study...")
    ablation = AblationStudy(model_path)
    ablation_results = ablation.run_complete_study(
        test_data, ground_truth, sample_size=1000, stratified=True
    )
    
    # 3. Statistical Analysis
    print("Running statistical analysis...")
    stats_analyzer = StatisticalAnalyzer()
    
    # Compare ablation methods
    for method1, method2 in [('baseline', 'complete_framework'), 
                           ('chain_of_thought', 'complete_framework')]:
        if method1 in ablation_results and method2 in ablation_results:
            comparison = stats_analyzer.perform_pairwise_comparison(
                [ablation_results[method1].cra], 
                [ablation_results[method2].cra],
                method1, method2
            )
            statistical_results[f"{method1}_vs_{method2}"] = comparison
    
    # 4. Uncertainty Quantification
    print("Running uncertainty quantification...")
    uncertainty = UncertaintyQuantifier()
    
    if hasattr(predictions[0], 'confidence'):
        confidences = [p.confidence for p in predictions]
        accuracies = [1.0 if is_correct(p, t) else 0.0 
                     for p, t in zip(predictions, ground_truth)]
        
        uncertainty_results = uncertainty.analyze_prediction_uncertainty(
            predictions, ground_truth
        )
    
    # 5. Generate Reports
    print("Generating reports...")
    
    # Error analysis report
    error_analyzer.generate_error_report(error_results, "error_analysis_report.txt")
    
    # Ablation study report
    ablation.generate_ablation_report(ablation_results, "ablation_study_report.txt")
    
    # Statistical analysis report
    stats_analyzer.generate_statistical_report(
        {'pairwise_comparisons': statistical_results},
        "statistical_analysis_report.txt"
    )
    
    if 'uncertainty_results' in locals():
        uncertainty.generate_uncertainty_report(
            uncertainty_results, "uncertainty_analysis_report.txt"
        )
    
    print("Analysis complete! Check generated report files.")
    
    return {
        'error_analysis': error_results,
        'ablation_study': ablation_results,
        'statistical_analysis': statistical_results,
        'uncertainty_analysis': uncertainty_results if 'uncertainty_results' in locals() else None
    }

# Run the complete analysis
results = run_complete_analysis(
    model_path="./your-model-path",
    test_data=your_test_data,
    ground_truth=your_ground_truth
)
```

### Command Line Analysis Scripts

```bash
# Run error analysis
python scripts/run_error_analysis.py \
    --predictions model_predictions.json \
    --ground-truth ground_truth.json \
    --method-name "Complete Framework" \
    --output-dir ./analysis_output \
    --cot-analysis \
    --raw-outputs raw_model_outputs.json

# Run ablation study  
python scripts/run_ablation_study.py \
    --model-path ./model \
    --test-data test_procedures.json \
    --ground-truth ground_truth.json \
    --output-dir ./ablation_output \
    --sample-size 1000 \
    --stratified \
    --dynamic-prompt-analysis

# Run statistical analysis
python scripts/run_statistical_analysis.py \
    --results-files baseline_results.json framework_results.json \
    --method-names "Baseline" "Complete Framework" \
    --output-dir ./stats_output \
    --metric cra \
    --literature-results literature_baselines.json

# Run uncertainty analysis
python scripts/run_uncertainty_analysis.py \
    --predictions predictions_with_confidence.json \
    --ground-truth ground_truth.json \
    --output-dir ./uncertainty_output \
    --validation-data validation_data.json \
    --generate-plots

# Run complete pipeline
python scripts/run_complete_analysis.py \
    --config analysis_config.yaml \
    --output-dir ./complete_analysis_output
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

# Analysis configuration
analysis:
  error_analysis:
    include_cot_failures: true
    categorize_by_complexity: true
  
  ablation_study:
    sample_size: 1000
    stratified_sampling: true
    include_dynamic_prompt_analysis: true
  
  statistical_analysis:
    significance_level: 0.05
    confidence_level: 0.95
    bootstrap_iterations: 1000
  
  uncertainty_quantification:
    calibration_methods: ["temperature_scaling", "platt_scaling", "isotonic_regression"]
    confidence_threshold: 0.8
    generate_plots: true

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

### Analysis Module APIs

#### ErrorAnalyzer

```python
# Initialize
error_analyzer = ErrorAnalyzer(config)

# Analyze errors
error_results = error_analyzer.analyze_prediction_errors(
    predictions=predictions,
    ground_truth=ground_truth,
    method_name="method_name"
)

# Compare methods
comparisons = error_analyzer.compare_methods(method_results)

# CoT failure analysis
cot_failures = error_analyzer.analyze_cot_failures(
    predictions=predictions,
    ground_truth=ground_truth,
    raw_outputs=raw_outputs
)

# Generate report
report = error_analyzer.generate_error_report(error_results, "error_report.txt")
```

#### AblationStudy

```python
# Initialize
ablation = AblationStudy(model_path="./model", config=config)

# Run complete study
study_results = ablation.run_complete_study(
    test_data=test_data,
    ground_truth=ground_truth,
    sample_size=1000,
    stratified=True
)

# Dynamic prompt analysis
dynamic_results = ablation.analyze_dynamic_prompt_components(
    test_sample=test_sample,
    truth_sample=truth_sample
)

# Generate reports
report = ablation.generate_ablation_report(study_results, "ablation_report.txt")
df = ablation.export_results_to_csv(study_results, "results.csv")
```

#### UncertaintyQuantifier

```python
# Initialize
uncertainty = UncertaintyQuantifier(config)

# Calibration metrics
calibration = uncertainty.calculate_calibration_metrics(
    confidences=confidences,
    accuracies=accuracies
)

# Temperature scaling
calibrated_probs, temp = uncertainty.perform_temperature_scaling(
    validation_logits=val_logits,
    validation_labels=val_labels,
    test_logits=test_logits
)

# Confidence analysis
confidence_analysis = uncertainty.analyze_prediction_uncertainty(
    predictions=predictions,
    ground_truth=ground_truth,
    confidence_threshold=0.8
)

# Generate reliability diagram
fig = uncertainty.generate_reliability_diagram(
    confidences=confidences,
    accuracies=accuracies,
    save_path="reliability.png"
)
```

#### StatisticalAnalyzer

```python
# Initialize
stats = StatisticalAnalyzer(config)

# Pairwise comparison
comparison = stats.perform_pairwise_comparison(
    method1_results=results1,
    method2_results=results2,
    method1_name="Method 1",
    method2_name="Method 2",
    test_type="paired_t"
)

# McNemar's test
mcnemar = stats.perform_mcnemar_test(
    method1_correct=correct1,
    method2_correct=correct2
)

# ANOVA
anova = stats.perform_anova(groups=group_dict, post_hoc=True)

# Reproducibility analysis
reproducibility = stats.calculate_baseline_reproducibility(
    literature_results=lit_results,
    reproduced_results=repro_results
)
```

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

### Example 2: Research Paper Reproduction

```python
"""
Reproduce the statistical analysis from the research paper
"""

from chemistry_llm.analysis import StatisticalAnalyzer, ErrorAnalyzer

def reproduce_paper_analysis():
    # Error reduction analysis (Table 4 in paper)
    error_analyzer = ErrorAnalyzer()
    
    # Load baseline, CoT+Prompt, and hybrid results
    baseline_results = load_results("baseline_predictions.json")
    cot_prompt_results = load_results("cot_prompt_predictions.json") 
    hybrid_results = load_results("hybrid_predictions.json")
    ground_truth = load_results("ground_truth.json")
    
    # Analyze each method
    methods = {
        'baseline': baseline_results,
        'cot_prompt': cot_prompt_results,
        'hybrid': hybrid_results
    }
    
    method_analyses = {}
    for method_name, results in methods.items():
        analysis = error_analyzer.analyze_prediction_errors(
            results, ground_truth, method_name
        )
        method_analyses[method_name] = analysis
    
    # Calculate error reductions
    comparisons = error_analyzer.compare_methods(method_analyses)
    
    # Print Table 4 style results
    print("Error Type                    | Baseline | CoT+Prompt | Hybrid | Reduction")
    print("-" * 75)
    
    for comparison in comparisons:
        if 'entity_recognition' in comparison.error_type.lower():
            print(f"{comparison.error_type:<30} | {comparison.baseline_rate:6.1f}% | "
                  f"{comparison.cot_prompt_rate:6.1f}% | {comparison.hybrid_rate:5.1f}% | "
                  f"{comparison.error_reduction:5.1f}%")
    
    # Statistical significance testing (Table 8 in paper)
    stats_analyzer = StatisticalAnalyzer()
    
    # Extract CRA scores for statistical testing
    baseline_cra = [r.get('cra', 0) for r in baseline_results]
    hybrid_cra = [r.get('cra', 0) for r in hybrid_results]
    
    # McNemar's test
    baseline_correct = [is_completely_correct(p, t) for p, t in zip(baseline_results, ground_truth)]
    hybrid_correct = [is_completely_correct(p, t) for p, t in zip(hybrid_results, ground_truth)]
    
    mcnemar_result = stats_analyzer.perform_mcnemar_test(
        baseline_correct, hybrid_correct, "Baseline", "Complete Framework"
    )
    
    print(f"\nMcNemar's χ² = {mcnemar_result['statistic']:.2f}")
    print(f"p-value = {mcnemar_result['p_value']:.6f}")
    print(f"Effect Size = {calculate_cohens_d(baseline_cra, hybrid_cra):.2f}")

reproduce_paper_analysis()
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

## 📊 Analysis Output Files

The analysis modules generate various output files:

### Error Analysis
- `error_analysis_results.json`: Detailed error categorization
- `error_analysis_report.txt`: Human-readable error report  
- `cot_failure_analysis.json`: Chain-of-Thought failure patterns
- `method_comparison.json`: Error rate comparisons between methods

### Ablation Study
- `ablation_study_results.json`: Complete ablation results
- `ablation_study_report.txt`: Formatted ablation report
- `ablation_results.csv`: Results in CSV format for analysis
- `dynamic_prompt_analysis.json`: Dynamic prompt component analysis

### Statistical Analysis
- `statistical_analysis_results.json`: All statistical test results
- `statistical_analysis_report.txt`: Statistical significance report
- `statistical_results.csv`: Statistical results in CSV format

### Uncertainty Analysis  
- `uncertainty_analysis_results.json`: Calibration and confidence analysis
- `uncertainty_analysis_report.txt`: Uncertainty quantification report
- `reliability_diagram.png`: Reliability diagram visualization
- `calibration_comparison.json`: Comparison of calibration methods

### Metrics Calculation
- `comprehensive_metrics.json`: All calculated performance metrics
- `complexity_analysis.json`: Performance by reaction complexity
- `metrics_summary.json`: Summary statistics

## 🧪 Testing

Run the test suite including analysis modules:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/chemistry_llm --cov-report=html

# Test analysis pipeline
python -m pytest tests/test_analysis_pipeline.py -v
```

### Test Structure

```
tests/
├── test_extractor.py              # Core extraction functionality
├── test_xml_parser.py             # XML parsing utilities
├── test_prompt_builder.py         # Prompt construction
├── test_integration.py            # End-to-end tests
└── fixtures/
    ├── sample_procedures.txt       # Test procedures
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
6. Run analysis module tests (`python -m pytest tests/analysis/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Release Process

1. Update version in `setup.py` and `src/chemistry_llm/__init__.py`
2. Update `CHANGELOG.md`
3. Run complete test suite including analysis modules
4. Generate analysis reports for validation
5. Create a git tag (`git tag v1.2.0`)
6. Push tag (`git push origin v1.2.0`)
7. GitHub Actions will automatically build and publish

## 📝 Changelog

### v1.2.0 (2025-08-21)
- **NEW**: Comprehensive analysis suite
- **NEW**: Error analysis with categorization and CoT failure analysis
- **NEW**: Ablation study framework with component analysis
- **NEW**: Statistical significance testing (t-tests, ANOVA, McNemar's)
- **NEW**: Uncertainty quantification and confidence calibration
- **NEW**: Metrics calculator with complexity-stratified analysis
- **NEW**: Command-line analysis scripts
- **NEW**: Complete analysis pipeline
- **IMPROVED**: Enhanced documentation with analysis examples
- **IMPROVED**: Additional test coverage for analysis modules

### v1.0.0 (2025-05-21)
- Initial release
- Core extraction functionality
- Chain-of-Thought prompting
- XML parsing and validation
- CLI interface
- Comprehensive test suite

## 📁 Repository Structure

```
RxnExtract/
├── README.md                           # This file
├── setup.py                           # Package setup
├── requirements.txt                   # Dependencies
├── config/                           # Configuration files
│   └── config.yaml                   # Main configuration
├── src/
│   └── chemistry_llm/
│       ├── __init__.py
│       ├── core/                     # Core extraction modules
│       │   ├── __init__.py
│       │   ├── extractor.py          # Main extraction engine
│       │   ├── model_loader.py       # Model loading utilities
│       │   └── prompt_builder.py     # Prompt construction
│       ├── analysis/                 # Analysis and evaluation modules
│       │   ├── __init__.py
│       │   ├── error_analysis.py     # Error categorization and analysis
│       │   ├── ablation_analysis.py  # Component ablation analysis
│       │   ├── metrics.py            # Comprehensive metrics
│       │   ├── statistical_analysis.py # Statistical testing
│       │   └── ucq_module.py         # Confidence calibration
│       ├── utils/                    # Utility modules
│       │   ├── __init__.py
│       │   ├── xml_parser.py         # XML parsing utilities
│       │   ├── logger.py             # Logging configuration
│       │   └── device_utils.py       # Hardware utilities
│       └── cli/                      # Command-line interface
│           ├── __init__.py
│           └── interface.py               # CLI entry point
├── scripts/                          # Analysis scripts
│   ├── run_error_analysis.py         # Error analysis script
│   ├── run_example.py                # Run Example extraction
│   ├── run_interactive.py            # Interactive mode
│   └── run_batch.py                  # Batch processing
├── tests/                            # Test suite
    ├── test_extractor.py             # Core extraction tests
    ├── test_xml_parser.py            # XML parsing tests
    └── fixtures/                     # Test data
        ├── sample_procedures.txt     # Sample procedures
```

## 🔍 Key Analysis Features

### 📊 Error Analysis Capabilities
- **Entity Recognition Errors**: Missing entities (52.4% reduction), false positives (54.8% reduction)
- **Role Classification Errors**: Reactant/product confusion (55.2% reduction), catalyst misidentification (51.5% reduction)
- **Condition Extraction Errors**: Missing temperature (49.1% reduction), incomplete procedures (50.8% reduction)
- **CoT Reasoning Failures**: Systematic analysis of Chain-of-Thought failure modes

### 🔬 Ablation Study Framework
- **8 Ablation Configurations**: From direct extraction to complete framework
- **Complexity Stratification**: Simple (40%), moderate (35%), complex (25%) reactions
- **Performance Metrics**: CRA, Entity F1, RCA, Condition F1, inference time
- **Component Contributions**: Individual and synergistic effects

### 📈 Statistical Analysis Suite
- **Significance Testing**: Paired t-tests, Wilcoxon, Mann-Whitney, McNemar's
- **Effect Size Calculation**: Cohen's d, eta-squared for practical significance
- **Confidence Intervals**: Bootstrap and parametric methods
- **Reproducibility Analysis**: Literature baseline validation

### 🎯 Uncertainty Quantification
- **Calibration Metrics**: ECE (57.1% reduction with temperature scaling), Brier Score
- **Calibration Methods**: Temperature scaling, Platt scaling, isotonic regression
- **Confidence Stratification**: High (≥0.8), medium (0.5-0.8), low (<0.5) confidence analysis
- **Reliability Diagrams**: Visual calibration assessment

## 🚀 Performance Highlights

Based on the research analysis, the complete framework achieves:

| Metric | Baseline | Complete Framework | Improvement |
|--------|----------|-------------------|-------------|
| Complete Reaction Accuracy | 23.4% | 52.1% | +122.6% |
| Entity F1 | 0.674 | 0.856 | +27.0% |
| Role Classification Accuracy | 68.2% | 85.9% | +25.9% |
| Condition F1 | 0.421 | 0.689 | +63.7% |

### Error Reduction Summary
- **Entity Recognition**: 47.8-55.2% error reduction
- **Role Classification**: 51.5-55.2% error reduction  
- **Condition Extraction**: 47.8-50.8% error reduction

### Statistical Significance
- **McNemar's χ²**: 134.67 (p < 0.001)
- **Effect Size**: Cohen's d = 0.82 (large effect)
- **95% CI**: [0.489, 0.535] for Complete Reaction Accuracy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
