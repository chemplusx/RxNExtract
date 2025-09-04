# Usage Guide & Examples

Comprehensive guide to using RxNExtract for chemical reaction extraction, from basic usage to advanced analysis.

## 🚀 Quick Start Examples

### Basic Python Usage
```python
from chemistry_llm import ChemistryReactionExtractor

# Initialize extractor with pre-trained model
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")

# Analyze a chemical procedure
procedure = """
Add 2.5 g of benzoic acid to 50 mL of ethanol. 
Heat the mixture to reflux for 4 hours.
Cool and filter to obtain the product.
"""

# Extract reaction information
results = extractor.analyze_procedure(procedure)

# Access structured data
data = results['extracted_data']
print("Reactants:", data['reactants'])
print("Products:", data['products'])
print("Conditions:", data['conditions'])
print("Reagents:", data['reagents'])
```

### Command Line Interface
```bash
# Interactive mode for real-time analysis
rxnextract --interactive

# Analyze single procedure
rxnextract --procedure "Add 5g NaCl to 100mL water and stir for 30 minutes"

# Batch processing from file
rxnextract --input procedures.txt --output results.json

# With custom model
rxnextract --model-path ./custom-model --input data.txt --output results.json
```

## 📖 Programmatic API

### ChemistryReactionExtractor Class

#### Initialization Options
```python
from chemistry_llm import ChemistryReactionExtractor

# Option 1: Load from HuggingFace Hub (recommended)
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    device="cuda",  # "cpu", "cuda", "auto"
    load_in_4bit=True,  # Memory optimization
    temperature=0.1,
    max_length=512
)

# Option 2: Load from local model path
extractor = ChemistryReactionExtractor(
    model_path="./path/to/local/model",
    device="auto",
    config={
        "temperature": 0.1,
        "top_p": 0.95,
        "max_new_tokens": 512
    }
)

# Option 3: Advanced configuration
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    torch_dtype="float16",
    device_map="auto",  # Automatic device assignment
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16"
    }
)
```

#### Core Methods

##### `analyze_procedure(procedure_text, return_raw=False, **kwargs)`
Main method for extracting reaction information.

```python
# Basic usage
results = extractor.analyze_procedure(procedure_text)

# With raw output included
results = extractor.analyze_procedure(
    procedure_text, 
    return_raw=True,
    temperature=0.05  # Override default temperature
)

# Return structure
{
    'extracted_data': {
        'reactants': [{'name': 'benzoic acid', 'amount': '2.5 g', ...}],
        'products': [{'name': 'product', 'yield': '84%', ...}],
        'reagents': [{'name': 'HCl', 'amount': '10 mL', ...}],
        'solvents': [{'name': 'ethanol', 'amount': '50 mL', ...}],
        'conditions': {
            'temperature': 'reflux',
            'time': '4 hours',
            'atmosphere': 'ambient'
        },
        'workup': ['Cool', 'filter', 'wash with cold water']
    },
    'confidence': 0.89,
    'processing_time': 2.3,
    'raw_output': '...'  # If return_raw=True
}
```

##### `extract_reaction(procedure_text, **generation_params)`
Low-level extraction method for advanced users.

```python
raw_output = extractor.extract_reaction(
    procedure_text,
    temperature=0.1,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.95
)
```

##### `batch_analyze(procedures_list, batch_size=8, show_progress=True)`
Efficient batch processing with progress tracking.

```python
procedures = [
    "Mix compound A with solvent B...",
    "Heat the reaction mixture to 150°C...",
    "Add catalyst C slowly while stirring..."
]

# Batch processing
results = extractor.batch_analyze(
    procedures,
    batch_size=4,
    show_progress=True
)

# Results is a list of analysis results
for i, result in enumerate(results):
    print(f"Procedure {i+1}: {len(result['extracted_data']['reactants'])} reactants")
```

## 🖥️ Command Line Interface

### Interactive Mode
Start an interactive session for real-time analysis:

```bash
rxnextract --interactive
```

Features:
- Real-time procedure input and analysis
- Formatted output display with syntax highlighting
- Session history and recall
- Error handling and recovery
- Multi-line input support

Interactive commands:
```
> analyze: Add 5g NaCl to water
> history: Show previous analyses
> save results.json: Save current session
> load previous.json: Load previous session
> config: Show current configuration
> help: Show available commands
> quit: Exit interactive mode
```

### Batch Processing
Process multiple procedures from various input formats:

```bash
# Basic batch processing
rxnextract --input procedures.txt --output results.json

# With progress bar and custom batch size
rxnextract --input procedures.txt --output results.json --batch-size 16 --progress

# Specify model and device
rxnextract --model chemplusx/rxnextract-complete --device cuda --input data.txt

# With confidence filtering
rxnextract --input procedures.txt --output results.json --min-confidence 0.8

# Custom output format
rxnextract --input procedures.txt --output results.xml --format xml
```

#### Input File Formats

**Text file** (`.txt`) - One procedure per line:
```
Add 5g NaCl to 100mL water and stir for 30 minutes.
Reflux the mixture of benzene and AlCl3 for 2 hours at 80°C.
Cool the solution to room temperature and filter the precipitate.
```

**JSON file** (`.json`) - Structured input:
```json
[
    {
        "id": "proc_001",
        "procedure": "Add 5g NaCl to 100mL water and stir for 30 minutes.",
        "metadata": {"source": "paper_1", "page": 15}
    },
    {
        "id": "proc_002", 
        "procedure": "Reflux the mixture of benzene and AlCl3 for 2 hours at 80°C.",
        "metadata": {"source": "paper_2", "page": 8}
    }
]
```

**CSV file** (`.csv`) - Tabular format:
```csv
id,procedure,source,page
proc_001,"Add 5g NaCl to 100mL water and stir for 30 minutes.",paper_1,15
proc_002,"Reflux the mixture of benzene and AlCl3 for 2 hours at 80°C.",paper_2,8
```

### Advanced CLI Options
```bash
# Full list of CLI options
rxnextract --help

# Memory optimization
rxnextract --input data.txt --load-in-4bit --max-memory 8GB

# Custom configuration
rxnextract --config custom_config.yaml --input procedures.txt

# Parallel processing
rxnextract --input data.txt --workers 4 --output results.json

# Verbose logging
rxnextract --input data.txt --log-level DEBUG --log-file extraction.log
```

## 📊 Advanced Usage Examples

### Custom Configuration
```python
from chemistry_llm import ChemistryReactionExtractor
from chemistry_llm.utils import setup_logging

# Setup custom logging
setup_logging(level="INFO", log_file="extraction.log")

# Custom configuration
config = {
    "model": {
        "temperature": 0.05,
        "top_p": 0.9,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.1
    },
    "prompts": {
        "use_cot": True,
        "cot_steps": [
            "Identify all chemical compounds",
            "Determine reaction roles",
            "Extract reaction conditions",
            "Identify products and yields"
        ]
    },
    "output": {
        "include_confidence": True,
        "include_reasoning": True,
        "xml_pretty_print": True
    }
}

# Initialize with custom config
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    config=config
)
```

### Multi-Document Processing
```python
import json
from pathlib import Path
from chemistry_llm import ChemistryReactionExtractor

extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")

def process_documents(input_dir, output_dir):
    """Process all text files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for txt_file in input_path.glob("*.txt"):
        print(f"Processing {txt_file.name}...")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            procedures = [line.strip() for line in f if line.strip()]
        
        # Batch process
        results = extractor.batch_analyze(procedures, batch_size=8)
        
        # Save results
        output_file = output_path / f"{txt_file.stem}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} results to {output_file}")

# Process all documents
process_documents("./input_docs", "./output_results")
```

### Integration with Scientific Libraries
```python
import pandas as pd
from chemistry_llm import ChemistryReactionExtractor
from rdkit import Chem
from rdkit.Chem import Draw

# Initialize extractor
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")

def analyze_and_visualize(procedures_df):
    """Analyze procedures and create molecular visualizations"""
    results = []
    
    for idx, row in procedures_df.iterrows():
        # Extract reaction information
        result = extractor.analyze_procedure(row['procedure'])
        
        # Process reactants for visualization
        reactant_smiles = []
        for reactant in result['extracted_data']['reactants']:
            if 'smiles' in reactant:
                reactant_smiles.append(reactant['smiles'])
        
        # Create molecular visualization
        if reactant_smiles:
            mols = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles]
            img = Draw.MolsToGridImage(mols, molsPerRow=3)
            img.save(f"reaction_{idx}_molecules.png")
        
        # Store results
        results.append({
            'procedure_id': row['id'],
            'reactants_count': len(result['extracted_data']['reactants']),
            'products_count': len(result['extracted_data']['products']),
            'confidence': result['confidence'],
            'extraction_result': result
        })
    
    return pd.DataFrame(results)

# Example usage
procedures_df = pd.read_csv("procedures.csv")
analysis_results = analyze_and_visualize(procedures_df)
analysis_results.to_csv("analysis_summary.csv", index=False)
```

### Error Handling and Robustness
```python
from chemistry_llm import ChemistryReactionExtractor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_extraction(procedures, max_retries=3):
    """Robust extraction with error handling and retries"""
    extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")
    results = []
    
    for i, procedure in enumerate(procedures):
        success = False
        retries = 0
        
        while not success and retries < max_retries:
            try:
                result = extractor.analyze_procedure(procedure)
                
                # Validate result
                if result['confidence'] < 0.5:
                    logger.warning(f"Low confidence ({result['confidence']:.2f}) for procedure {i}")
                
                results.append({
                    'procedure_index': i,
                    'success': True,
                    'result': result,
                    'retries_used': retries
                })
                success = True
                
            except Exception as e:
                retries += 1
                logger.error(f"Extraction failed for procedure {i} (attempt {retries}): {e}")
                
                if retries >= max_retries:
                    results.append({
                        'procedure_index': i,
                        'success': False,
                        'error': str(e),
                        'retries_used': retries
                    })
    
    return results
```

## 🔍 Analysis & Evaluation Framework

### Error Analysis
Systematic analysis of extraction errors across different categories:

```python
from chemistry_llm.analysis import ErrorAnalyzer

# Initialize error analyzer
error_analyzer = ErrorAnalyzer()

# Load your predictions and ground truth
predictions = load_predictions("model_predictions.json")
ground_truth = load_ground_truth("ground_truth.json")

# Comprehensive error analysis
error_results = error_analyzer.analyze_prediction_errors(
    predictions=predictions,
    ground_truth=ground_truth,
    method_name="RxNExtract-Complete"
)

# Analyze specific error categories
entity_errors = error_analyzer.analyze_entity_errors(predictions, ground_truth)
role_errors = error_analyzer.analyze_role_classification_errors(predictions, ground_truth)
condition_errors = error_analyzer.analyze_condition_extraction_errors(predictions, ground_truth)

# Chain-of-Thought failure analysis
cot_failures = error_analyzer.analyze_cot_failures(
    predictions=predictions,
    ground_truth=ground_truth,
    raw_outputs=raw_model_outputs
)

# Generate comprehensive error report
report = error_analyzer.generate_error_report(
    error_results, 
    output_file="comprehensive_error_analysis.txt"
)

print("Error Analysis Summary:")
print(f"Total errors analyzed: {error_results.total_errors}")
print(f"Entity recognition errors: {error_results.entity_errors}")
print(f"Role classification errors: {error_results.role_errors}")
print(f"Condition extraction errors: {error_results.condition_errors}")
```

### Ablation Studies
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
    stratified=True,  # Stratify by reaction complexity
    random_state=42
)

# Analyze dynamic prompt components
dynamic_analysis = ablation.analyze_dynamic_prompt_components(
    test_sample=test_procedures[:100],
    truth_sample=ground_truth[:100]
)

# Component contribution analysis
component_contributions = ablation.analyze_component_contributions(
    study_results
)

# Generate ablation report
report = ablation.generate_ablation_report(
    study_results, 
    output_file="ablation_study_report.txt"
)

# Export results for further analysis
df = ablation.export_results_to_csv(study_results, "ablation_results.csv")

print("Ablation Study Results:")
for config, metrics in study_results.items():
    print(f"{config}: CRA = {metrics.cra:.3f}, F1 = {metrics.entity_f1:.3f}")
```

#### Available Ablation Configurations
- **Direct Extraction**: Basic extraction without enhancements
- **Structured Output**: XML-structured output format
- **Meta Prompt**: Enhanced prompt engineering
- **Chain-of-Thought**: Step-by-step reasoning
- **CoT + Reflection**: Chain-of-thought with self-reflection
- **Self-Grounding**: Entity validation and correction
- **Complete Framework**: All components combined
- **Iterative Refinement**: Multi-pass extraction

### Statistical Analysis
Comprehensive statistical testing and significance analysis:

```python
from chemistry_llm.analysis import StatisticalAnalyzer

# Initialize statistical analyzer
stats_analyzer = StatisticalAnalyzer()

# Load results from different methods
baseline_results = load_results("baseline_predictions.json")
improved_results = load_results("rxnextract_predictions.json")

# Pairwise method comparison
comparison = stats_analyzer.perform_pairwise_comparison(
    method1_results=[r['cra'] for r in baseline_results],
    method2_results=[r['cra'] for r in improved_results],
    method1_name="Baseline",
    method2_name="RxNExtract",
    test_type="paired_t"  # or "wilcoxon", "mann_whitney"
)

print(f"Statistical Comparison Results:")
print(f"p-value: {comparison['p_value']:.6f}")
print(f"Effect size (Cohen's d): {comparison['effect_size']:.3f}")
print(f"Statistically significant: {comparison['significant']}")

# McNemar's test for classification performance
baseline_correct = [is_correct(pred, truth) for pred, truth in zip(baseline_results, ground_truth)]
improved_correct = [is_correct(pred, truth) for pred, truth in zip(improved_results, ground_truth)]

mcnemar_result = stats_analyzer.perform_mcnemar_test(
    method1_correct=baseline_correct,
    method2_correct=improved_correct,
    method1_name="Baseline",
    method2_name="RxNExtract"
)

# ANOVA for multiple method comparison
methods_data = {
    'Baseline': [r['cra'] for r in baseline_results],
    'CoT-Only': [r['cra'] for r in cot_results],
    'RxNExtract': [r['cra'] for r in improved_results]
}

anova_results = stats_analyzer.perform_anova(
    groups=methods_data,
    post_hoc=True  # Include post-hoc pairwise comparisons
)

# Generate statistical report
stats_report = stats_analyzer.generate_statistical_report(
    {
        'pairwise_comparisons': {'baseline_vs_rxnextract': comparison},
        'mcnemar_tests': {'classification_performance': mcnemar_result},
        'anova_results': anova_results
    },
    output_file="statistical_analysis_report.txt"
)
```

### Uncertainty Quantification
Confidence calibration and uncertainty analysis:

```python
from chemistry_llm.analysis import UncertaintyQuantifier

# Initialize uncertainty quantifier
uncertainty = UncertaintyQuantifier()

# Extract confidence scores from predictions
confidences = [pred['confidence'] for pred in predictions]
accuracies = [1.0 if is_correct(pred, truth) else 0.0 
              for pred, truth in zip(predictions, ground_truth)]

# Calculate calibration metrics
calibration_metrics = uncertainty.calculate_calibration_metrics(
    confidences=confidences,
    accuracies=accuracies,
    n_bins=10
)

print(f"Calibration Metrics:")
print(f"Expected Calibration Error (ECE): {calibration_metrics.ece:.4f}")
print(f"Brier Score: {calibration_metrics.brier_score:.4f}")
print(f"Reliability: {calibration_metrics.reliability:.4f}")

# Perform temperature scaling for calibration
calibrated_probs, optimal_temperature = uncertainty.perform_temperature_scaling(
    validation_logits=validation_logits,
    validation_labels=validation_labels,
    test_logits=test_logits
)

print(f"Optimal temperature: {optimal_temperature:.3f}")

# Confidence-stratified performance analysis
confidence_analysis = uncertainty.analyze_confidence_stratified_performance(
    confidences=confidences,
    accuracies=accuracies,
    n_strata=5
)

# Generate reliability diagram
reliability_fig = uncertainty.generate_reliability_diagram(
    confidences=confidences,
    accuracies=accuracies,
    save_path="reliability_diagram.png"
)

# Comprehensive uncertainty analysis
uncertainty_results = uncertainty.analyze_prediction_uncertainty(
    predictions=predictions,
    ground_truth=ground_truth,
    confidence_threshold=0.8
)
```

### Metrics Calculator
Comprehensive performance metrics calculation:

```python
from chemistry_llm.analysis import MetricsCalculator

# Initialize metrics calculator
metrics_calc = MetricsCalculator()

# Calculate comprehensive metrics
comprehensive_metrics = metrics_calc.calculate_comprehensive_metrics(
    predictions=predictions,
    ground_truth=ground_truth
)

print("Performance Metrics:")
print(f"Complete Reaction Accuracy (CRA): {comprehensive_metrics['cra']:.3f}")
print(f"Entity F1 Score: {comprehensive_metrics['entity_f1']:.3f}")
print(f"Role Classification Accuracy: {comprehensive_metrics['rca']:.3f}")
print(f"Condition Extraction F1: {comprehensive_metrics['condition_f1']:.3f}")

# Performance by reaction complexity
complexity_labels = assign_complexity_labels(ground_truth)  # Your complexity assignment logic
complexity_metrics = metrics_calc.analyze_performance_by_complexity(
    predictions=predictions,
    ground_truth=ground_truth,
    complexity_labels=complexity_labels
)

# Error reduction analysis
baseline_metrics = calculate_baseline_metrics(baseline_predictions, ground_truth)
error_reduction = metrics_calc.calculate_error_reduction(
    baseline_metrics=baseline_metrics,
    improved_metrics=comprehensive_metrics
)

print(f"Error Reduction:")
print(f"Entity Recognition: {error_reduction['entity_recognition']:.1f}%")
print(f"Role Classification: {error_reduction['role_classification']:.1f}%")
print(f"Condition Extraction: {error_reduction['condition_extraction']:.1f}%")

# Export metrics summary
metrics_calc.export_metrics_summary(
    comprehensive_metrics, 
    "comprehensive_metrics_summary.json"
)
```

## 🔧 Utility Functions and Helpers

### XML Parsing and Validation
```python
from chemistry_llm.utils.xml_parser import parse_reaction_xml, validate_xml_structure

# Parse extracted XML to structured data
xml_output = """
<reaction>
    <reactants>
        <reactant name="benzoic acid" amount="2.5 g"/>
    </reactants>
    <products>
        <product name="product" yield="84%"/>
    </products>
    <conditions>
        <temperature>reflux</temperature>
        <time>4 hours</time>
    </conditions>
</reaction>
"""

# Parse XML
structured_data = parse_reaction_xml(xml_output)

# Validate XML structure
is_valid, errors = validate_xml_structure(xml_output)
if not is_valid:
    print("XML validation errors:", errors)
```

### Device and Memory Management
```python
from chemistry_llm.utils.device_utils import get_optimal_device, get_memory_info

# Automatic device selection
device = get_optimal_device()
print(f"Selected device: {device}")

# Memory information
memory_info = get_memory_info()
print(f"Available GPU memory: {memory_info['gpu_memory_available']:.1f} GB")
print(f"Available system RAM: {memory_info['system_memory_available']:.1f} GB")

# Optimize model loading based on available memory
if memory_info['gpu_memory_available'] > 12:
    load_in_4bit = False
    torch_dtype = "float16"
elif memory_info['gpu_memory_available'] > 6:
    load_in_4bit = True
    torch_dtype = "float16"
else:
    # Use CPU
    device = "cpu"
    torch_dtype = "float32"
```

### Custom Prompt Engineering
```python
from chemistry_llm.core.prompt_builder import PromptBuilder

# Create custom prompt builder
prompt_builder = PromptBuilder(
    use_cot=True,
    cot_steps=[
        "Identify all chemical compounds mentioned",
        "Determine the role of each compound",
        "Extract reaction conditions",
        "Identify products and yields",
        "Structure the output in XML format"
    ]
)

# Build custom prompt
custom_prompt = prompt_builder.build_extraction_prompt(
    procedure_text="Your procedure here",
    include_examples=True,
    format_type="xml"
)

# Use with extractor
result = extractor.extract_reaction(
    procedure_text,
    custom_prompt=custom_prompt
)
```

## 📊 Output Formats and Post-Processing

### Structured Output Formats
```python
# JSON output (default)
results = extractor.analyze_procedure(procedure, output_format="json")

# XML output
results = extractor.analyze_procedure(procedure, output_format="xml")

# YAML output
results = extractor.analyze_procedure(procedure, output_format="yaml")

# Custom structured output
results = extractor.analyze_procedure(
    procedure,
    output_format="structured",
    include_confidence=True,
    include_reasoning=True
)
```

### Post-Processing and Validation
```python
from chemistry_llm.utils.post_processing import validate_extraction, normalize_entities

def post_process_results(results):
    """Post-process extraction results for quality assurance"""
    
    # Validate extraction completeness
    validation_results = validate_extraction(results['extracted_data'])
    
    # Normalize chemical entities
    normalized_data = normalize_entities(results['extracted_data'])
    
    # Check for common extraction errors
    if validation_results['missing_reactants']:
        print("Warning: No reactants found")
    
    if validation_results['missing_products']:
        print("Warning: No products found")
    
    if results['confidence'] < 0.7:
        print(f"Warning: Low confidence score ({results['confidence']:.2f})")
    
    return {
        **results,
        'extracted_data': normalized_data,
        'validation': validation_results
    }

# Apply post-processing
processed_results = post_process_results(raw_results)
```

## 🔗 Integration Examples

### Jupyter Notebook Integration
```python
# Display results with rich formatting in Jupyter
from IPython.display import display, HTML, JSON
import pandas as pd

def display_extraction_results(results):
    """Display extraction results in Jupyter with rich formatting"""
    
    # Display confidence and timing
    print(f"Confidence: {results['confidence']:.2f}")
    print(f"Processing time: {results['processing_time']:.1f}s")
    
    # Display extracted entities as DataFrame
    entities_data = []
    for reactant in results['extracted_data']['reactants']:
        entities_data.append({
            'Type': 'Reactant',
            'Name': reactant['name'],
            'Amount': reactant.get('amount', 'N/A'),
            'Role': 'Reactant'
        })
    
    for product in results['extracted_data']['products']:
        entities_data.append({
            'Type': 'Product',
            'Name': product['name'],
            'Amount': product.get('yield', 'N/A'),
            'Role': 'Product'
        })
    
    df = pd.DataFrame(entities_data)
    display(df)
    
    # Display conditions
    if results['extracted_data']['conditions']:
        print("\nReaction Conditions:")
        display(JSON(results['extracted_data']['conditions']))

# Use in Jupyter
results = extractor.analyze_procedure(procedure)
display_extraction_results(results)
```

### Web API Integration
```python
from flask import Flask, request, jsonify
from chemistry_llm import ChemistryReactionExtractor

app = Flask(__name__)
extractor = ChemistryReactionExtractor.from_pretrained("chemplusx/rxnextract-complete")

@app.route('/extract', methods=['POST'])
def extract_reaction():
    """Web API endpoint for reaction extraction"""
    try:
        data = request.get_json()
        procedure = data.get('procedure', '')
        
        if not procedure:
            return jsonify({'error': 'No procedure provided'}), 400
        
        # Extract reaction information
        results = extractor.analyze_procedure(procedure)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_extract', methods=['POST'])
def batch_extract():
    """Batch extraction endpoint"""
    try:
        data = request.get_json()
        procedures = data.get('procedures', [])
        
        if not procedures:
            return jsonify({'error': 'No procedures provided'}), 400
        
        # Batch processing
        results = extractor.batch_analyze(procedures)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 🚀 Performance Optimization

### Memory Optimization
```python
# Optimize for low-memory environments
extractor = ChemistryReactionExtractor.from_pretrained(
    "chemplusx/rxnextract-complete",
    load_in_4bit=True,
    device="cuda",
    max_memory={0: "6GB"},  # Limit GPU 0 to 6GB
    offload_folder="./offload"
)

# Clear cache periodically
import torch
torch.cuda.empty_cache()
```

### Batch Processing Optimization
```python
def optimized_batch_processing(procedures, batch_size=8):
    """Optimized batch processing with memory management"""
    
    extractor = ChemistryReactionExtractor.from_pretrained(
        "chemplusx/rxnextract-complete",
        load_in_4bit=True
    )
    
    results = []
    for i in range(0, len(procedures), batch_size):
        batch = procedures[i:i+batch_size]
        
        # Process batch
        batch_results = extractor.batch_analyze(batch)
        results.extend(batch_results)
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Progress update
        print(f"Processed {min(i+batch_size, len(procedures))}/{len(procedures)} procedures")
    
    return results
```

---

**Next Steps**: 
- For comprehensive analysis capabilities, see [Analysis & Evaluation Guide](ANALYSIS.md)
- For version history and updates, see [Changelog](CHANGELOG.md)
- For contributing to the project, see [Contributing Guidelines](../CONTRIBUTING.md)