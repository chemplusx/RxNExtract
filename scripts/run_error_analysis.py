"""
Command-line scripts for running analysis modules
"""

# scripts/run_error_analysis.py
"""
Run comprehensive error analysis on model predictions
"""

import argparse
import json
import sys
from pathlib import Path

from chemistry_llm.analysis import ErrorAnalyzer
from chemistry_llm.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run error analysis on model predictions")
    
    parser.add_argument("--predictions", required=True, 
                       help="JSON file containing model predictions")
    parser.add_argument("--ground-truth", required=True,
                       help="JSON file containing ground truth data")
    parser.add_argument("--method-name", default="model",
                       help="Name of the method being analyzed")
    parser.add_argument("--output-dir", default="./analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--compare-methods", nargs="+",
                       help="Additional prediction files to compare against")
    parser.add_argument("--cot-analysis", action="store_true",
                       help="Perform Chain-of-Thought failure analysis")
    parser.add_argument("--raw-outputs",
                       help="JSON file containing raw model outputs for CoT analysis")
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading prediction and ground truth data...")
    
    try:
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
            
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)
    
    # Initialize error analyzer
    error_analyzer = ErrorAnalyzer()
    
    # Run error analysis
    logger.info(f"Running error analysis for {args.method_name}...")
    
    error_results = error_analyzer.analyze_prediction_errors(
        predictions=predictions,
        ground_truth=ground_truth,
        method_name=args.method_name
    )
    
    # Save detailed results
    results_file = output_dir / f"{args.method_name}_error_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(error_results, f, indent=2)
    
    logger.info(f"Error analysis results saved to {results_file}")
    
    # Generate and save report
    report = error_analyzer.generate_error_report(error_results)
    report_file = output_dir / f"{args.method_name}_error_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Error analysis report saved to {report_file}")
    
    # Method comparison if requested
    if args.compare_methods:
        logger.info("Running method comparisons...")
        
        method_results = {args.method_name: error_results}
        
        for i, compare_file in enumerate(args.compare_methods):
            try:
                with open(compare_file, 'r') as f:
                    compare_predictions = json.load(f)
                
                compare_name = f"method_{i+1}"
                compare_results = error_analyzer.analyze_prediction_errors(
                    predictions=compare_predictions,
                    ground_truth=ground_truth,
                    method_name=compare_name
                )
                
                method_results[compare_name] = compare_results
                
            except Exception as e:
                logger.warning(f"Failed to load comparison method {compare_file}: {str(e)}")
        
        # Generate comparison
        comparisons = error_analyzer.compare_methods(method_results)
        
        comparison_file = output_dir / "method_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump([comp.to_dict() for comp in comparisons], f, indent=2)
        
        logger.info(f"Method comparison saved to {comparison_file}")
    
    # CoT failure analysis if requested
    if args.cot_analysis and args.raw_outputs:
        logger.info("Running Chain-of-Thought failure analysis...")
        
        try:
            with open(args.raw_outputs, 'r') as f:
                raw_outputs = json.load(f)
            
            cot_results = error_analyzer.analyze_cot_failures(
                predictions=predictions,
                ground_truth=ground_truth,
                raw_outputs=raw_outputs
            )
            
            cot_file = output_dir / f"{args.method_name}_cot_analysis.json"
            with open(cot_file, 'w') as f:
                json.dump(cot_results, f, indent=2)
            
            logger.info(f"CoT failure analysis saved to {cot_file}")
            
        except Exception as e:
            logger.error(f"Failed to run CoT analysis: {str(e)}")
    
    logger.info("Error analysis complete!")


if __name__ == "__main__":
    main()


# scripts/run_ablation_study.py
"""
Run comprehensive ablation study
"""

import argparse
import json
import sys
from pathlib import Path

from chemistry_llm.analysis import AblationStudy
from chemistry_llm.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    
    parser.add_argument("--model-path", required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--test-data", required=True,
                       help="JSON file containing test procedures")
    parser.add_argument("--ground-truth", required=True,
                       help="JSON file containing ground truth data")
    parser.add_argument("--output-dir", default="./ablation_output",
                       help="Output directory for ablation results")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--stratified", action="store_true",
                       help="Use stratified sampling by reaction complexity")
    parser.add_argument("--config",
                       help="Path to configuration file")
    parser.add_argument("--dynamic-prompt-analysis", action="store_true",
                       help="Include dynamic prompt component analysis")
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading test data and ground truth...")
    
    try:
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
            
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {str(e)}")
    
    # Initialize ablation study
    ablation = AblationStudy(
        model_path=args.model_path,
        config=config
    )
    
    # Run ablation study
    logger.info("Running ablation study...")
    
    study_results = ablation.run_complete_study(
        test_data=test_data,
        ground_truth=ground_truth,
        sample_size=args.sample_size,
        stratified=args.stratified
    )
    
    # Save results
    results_file = output_dir / "ablation_study_results.json"
    
    # Convert results to serializable format
    serializable_results = {}
    for key, value in study_results.items():
        if key == 'ablation_results':
            serializable_results[key] = {
                k: v.to_dict() if v else None 
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Ablation study results saved to {results_file}")
    
    # Generate report
    report = ablation.generate_ablation_report(study_results)
    report_file = output_dir / "ablation_study_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Ablation study report saved to {report_file}")
    
    # Export to CSV
    csv_file = output_dir / "ablation_results.csv"
    df = ablation.export_results_to_csv(study_results, csv_file)
    logger.info(f"Ablation results exported to {csv_file}")
    
    # Dynamic prompt analysis if requested
    if args.dynamic_prompt_analysis:
        logger.info("Running dynamic prompt component analysis...")
        
        try:
            dynamic_results = ablation.analyze_dynamic_prompt_components(
                test_sample=test_data[:100],  # Use subset for speed
                truth_sample=ground_truth[:100]
            )
            
            dynamic_file = output_dir / "dynamic_prompt_analysis.json"
            with open(dynamic_file, 'w') as f:
                json.dump(dynamic_results, f, indent=2)
            
            logger.info(f"Dynamic prompt analysis saved to {dynamic_file}")
            
        except Exception as e:
            logger.error(f"Failed to run dynamic prompt analysis: {str(e)}")
    
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()


# scripts/run_statistical_analysis.py
"""
Run comprehensive statistical analysis
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from chemistry_llm.analysis import StatisticalAnalyzer
from chemistry_llm.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run statistical analysis")
    
    parser.add_argument("--results-files", nargs="+", required=True,
                       help="JSON files containing method results")
    parser.add_argument("--method-names", nargs="+", required=True,
                       help="Names of the methods (must match order of results files)")
    parser.add_argument("--output-dir", default="./statistical_output",
                       help="Output directory for statistical results")
    parser.add_argument("--metric", default="cra",
                       help="Metric to analyze (cra, entity_f1, etc.)")
    parser.add_argument("--significance-level", type=float, default=0.05,
                       help="Significance level for tests")
    parser.add_argument("--literature-results",
                       help="JSON file with literature baseline results")
    parser.add_argument("--cv-results",
                       help="JSON file with cross-validation results")
    
    args = parser.parse_args()
    
    if len(args.results_files) != len(args.method_names):
        logger.error("Number of result files must match number of method names")
        sys.exit(1)
    
    # Setup logging and output directory
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info("Loading method results...")
    
    method_results = {}
    
    for results_file, method_name in zip(args.results_files, args.method_names):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract the specified metric
            if isinstance(data, list):
                # List of individual results
                metric_values = [item.get(args.metric, 0.0) for item in data if item]
            elif isinstance(data, dict) and 'ablation_results' in data:
                # Ablation study results
                metric_values = []
                for config_name, result in data['ablation_results'].items():
                    if result and args.metric in result:
                        metric_values.append(result[args.metric])
            else:
                # Single result or other format
                metric_values = [data.get(args.metric, 0.0)]
            
            method_results[method_name] = metric_values
            logger.info(f"Loaded {len(metric_values)} results for {method_name}")
            
        except Exception as e:
            logger.error(f"Failed to load results from {results_file}: {str(e)}")
            sys.exit(1)
    
    # Initialize statistical analyzer
    stats_analyzer = StatisticalAnalyzer(
        config={'significance_level': args.significance_level}
    )
    
    # Pairwise comparisons
    logger.info("Running pairwise statistical comparisons...")
    
    pairwise_results = {}
    method_names = list(method_results.keys())
    
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method1, method2 = method_names[i], method_names[j]
            
            comparison = stats_analyzer.perform_pairwise_comparison(
                method1_results=method_results[method1],
                method2_results=method_results[method2],
                method1_name=method1,
                method2_name=method2,
                test_type="paired_t"
            )
            
            pairwise_results[f"{method1}_vs_{method2}"] = comparison
    
    # ANOVA if more than 2 methods
    anova_results = None
    if len(method_names) > 2:
        logger.info("Running ANOVA...")
        
        anova_results = stats_analyzer.perform_anova(
            groups=method_results,
            post_hoc=True
        )
    
    # McNemar tests (if we have binary correctness data)
    mcnemar_results = {}
    # This would require additional data processing to convert continuous metrics to binary correctness
    
    # Baseline reproducibility analysis
    reproducibility_results = None
    if args.literature_results:
        logger.info("Running baseline reproducibility analysis...")
        
        try:
            with open(args.literature_results, 'r') as f:
                literature_data = json.load(f)
            
            reproducibility_results = stats_analyzer.calculate_baseline_reproducibility(
                literature_results=literature_data,
                reproduced_results=method_results
            )
            
        except Exception as e:
            logger.warning(f"Failed to run reproducibility analysis: {str(e)}")
    
    # Cross-validation analysis
    cv_results = None
    if args.cv_results:
        logger.info("Running cross-validation analysis...")
        
        try:
            with open(args.cv_results, 'r') as f:
                cv_data = json.load(f)
            
            # Extract CV results for each method
            cv_method_results = []
            cv_method_names = []
            
            for method_name in method_names:
                if method_name in cv_data:
                    cv_method_results.append(cv_data[method_name])
                    cv_method_names.append(method_name)
            
            if cv_method_results:
                cv_results = stats_analyzer.perform_cross_validation_analysis(
                    cv_results=cv_method_results,
                    method_names=cv_method_names
                )
            
        except Exception as e:
            logger.warning(f"Failed to run CV analysis: {str(e)}")
    
    # Compile all results
    all_results = {
        'pairwise_comparisons': pairwise_results,
        'anova': anova_results,
        'mcnemar_tests': mcnemar_results,
        'reproducibility': reproducibility_results,
        'cross_validation': cv_results
    }
    
    # Save detailed results
    results_file = output_dir / "statistical_analysis_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Statistical analysis results saved to {results_file}")
    
    # Generate report
    report = stats_analyzer.generate_statistical_report(all_results)
    report_file = output_dir / "statistical_analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Statistical analysis report saved to {report_file}")
    
    # Export to DataFrame
    df = stats_analyzer.export_results_to_dataframe(all_results)
    csv_file = output_dir / "statistical_results.csv"
    df.to_csv(csv_file, index=False)
    
    logger.info(f"Statistical results exported to {csv_file}")
    logger.info("Statistical analysis complete!")


if __name__ == "__main__":
    main()


# scripts/run_uncertainty_analysis.py
"""
Run uncertainty quantification and confidence calibration analysis
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from chemistry_llm.analysis import UncertaintyQuantifier
from chemistry_llm.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run uncertainty quantification analysis")
    
    parser.add_argument("--predictions", required=True,
                       help="JSON file containing predictions with confidence scores")
    parser.add_argument("--ground-truth", required=True,
                       help="JSON file containing ground truth data")
    parser.add_argument("--output-dir", default="./uncertainty_output",
                       help="Output directory for uncertainty analysis results")
    parser.add_argument("--validation-data",
                       help="JSON file containing validation data for calibration")
    parser.add_argument("--calibration-methods", nargs="+", 
                       default=["temperature_scaling", "platt_scaling"],
                       help="Calibration methods to compare")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                       help="Threshold for high confidence classification")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate reliability diagrams and other plots")
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading prediction and ground truth data...")
    
    try:
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
            
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)
    
    # Extract confidence scores and calculate accuracies
    confidences = []
    accuracies = []
    
    for pred, truth in zip(predictions, ground_truth):
        if pred and 'confidence' in pred:
            confidences.append(pred['confidence'])
            
            # Calculate accuracy (simplified - you may need to adjust this)
            # This assumes you have a function to calculate sample-level accuracy
            accuracy = calculate_sample_accuracy(pred, truth)
            accuracies.append(accuracy)
    
    if not confidences:
        logger.error("No confidence scores found in predictions")
        sys.exit(1)
    
    logger.info(f"Found {len(confidences)} predictions with confidence scores")
    
    # Initialize uncertainty quantifier
    uncertainty = UncertaintyQuantifier()
    
    # Basic calibration metrics
    logger.info("Calculating calibration metrics...")
    
    calibration_metrics = uncertainty.calculate_calibration_metrics(
        confidences=confidences,
        accuracies=accuracies
    )
    
    logger.info(f"Expected Calibration Error: {calibration_metrics.ece:.4f}")
    logger.info(f"Brier Score: {calibration_metrics.brier_score:.4f}")
    
    # Confidence-stratified performance analysis
    logger.info("Analyzing confidence-stratified performance...")
    
    uncertainty_results = uncertainty.analyze_prediction_uncertainty(
        predictions=predictions,
        ground_truth=ground_truth,
        confidence_threshold=args.confidence_threshold
    )
    
    # Calibration method comparison (if validation data provided)
    calibration_comparison = None
    if args.validation_data:
        logger.info("Comparing calibration methods...")
        
        try:
            with open(args.validation_data, 'r') as f:
                validation_data = json.load(f)
            
            # Extract validation scores and labels
            val_scores = []
            val_labels = []
            test_scores = np.array(confidences)
            test_labels = np.array(accuracies)
            
            for val_pred, val_truth in zip(validation_data['predictions'], 
                                         validation_data['ground_truth']):
                if val_pred and 'confidence' in val_pred:
                    val_scores.append(val_pred['confidence'])
                    val_labels.append(calculate_sample_accuracy(val_pred, val_truth))
            
            val_scores = np.array(val_scores)
            val_labels = np.array(val_labels)
            
            calibration_comparison = uncertainty.compare_calibration_methods(
                validation_scores=val_scores,
                validation_labels=val_labels,
                test_scores=test_scores,
                test_labels=test_labels
            )
            
        except Exception as e:
            logger.warning(f"Failed to run calibration comparison: {str(e)}")
    
    # Confidence intervals
    logger.info("Calculating confidence intervals...")
    
    confidence_intervals = uncertainty.calculate_confidence_intervals(
        performance_scores=accuracies,
        confidence_level=0.95
    )
    
    # Compile results
    all_results = {
        'calibration_metrics': calibration_metrics.to_dict(),
        'uncertainty_analysis': uncertainty_results,
        'calibration_comparison': {
            method: metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
            for method, metrics in (calibration_comparison or {}).items()
        },
        'confidence_intervals': confidence_intervals,
        'summary': {
            'total_samples': len(predictions),
            'samples_with_confidence': len(confidences),
            'mean_confidence': np.mean(confidences),
            'mean_accuracy': np.mean(accuracies)
        }
    }
    
    # Save results
    results_file = output_dir / "uncertainty_analysis_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Uncertainty analysis results saved to {results_file}")
    
    # Generate report
    report = uncertainty.generate_uncertainty_report(uncertainty_results)
    report_file = output_dir / "uncertainty_analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Uncertainty analysis report saved to {report_file}")
    
    # Generate plots if requested
    if args.generate_plots:
        logger.info("Generating reliability diagram...")
        
        try:
            fig = uncertainty.generate_reliability_diagram(
                confidences=confidences,
                accuracies=accuracies,
                save_path=output_dir / "reliability_diagram.png"
            )
            
            logger.info("Reliability diagram saved")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {str(e)}")
    
    logger.info("Uncertainty analysis complete!")


def calculate_sample_accuracy(prediction, ground_truth):
    """
    Calculate accuracy for a single sample
    This is a simplified version - you may need to adjust based on your data format
    """
    # Extract entities from prediction and ground truth
    pred_entities = set()
    true_entities = set()
    
    for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
        for component in prediction.get(component_type, []):
            if isinstance(component, dict) and 'name' in component:
                pred_entities.add(component['name'].lower().strip())
        
        for component in ground_truth.get(component_type, []):
            if isinstance(component, dict) and 'name' in component:
                true_entities.add(component['name'].lower().strip())
    
    if len(true_entities) == 0:
        return 1.0 if len(pred_entities) == 0 else 0.0
    
    # Calculate F1 as accuracy proxy
    intersection = len(pred_entities & true_entities)
    
    if len(pred_entities) == 0:
        precision = 0.0
    else:
        precision = intersection / len(pred_entities)
    
    recall = intersection / len(true_entities)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


if __name__ == "__main__":
    main()
