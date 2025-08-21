"""
Ablation study module for chemistry LLM inference
Provides systematic component-level performance analysis
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.extractor import ChemistryReactionExtractor
from ..core.prompt_builder import PromptBuilder
from ..utils.logger import get_logger
from ..utils.xml_parser import parse_reaction_xml
from .metrics import MetricsCalculator
from .error_analysis import ErrorAnalyzer

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """Container for ablation study results"""
    method_name: str
    cra: float  # Complete Reaction Accuracy
    entity_f1: float
    rca: float  # Role Classification Accuracy
    condition_f1: float
    inference_time: float
    complexity_performance: Dict[str, float]
    component_contributions: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReactionComplexityClassifier:
    """
    Classifies chemical reactions by complexity level
    """
    
    def __init__(self):
        self.complexity_criteria = {
            'simple': {
                'max_steps': 1,
                'max_components': 5,
                'has_catalyst': False,
                'multi_phase': False
            },
            'moderate': {
                'max_steps': 2,
                'max_components': 8,
                'has_catalyst': True,
                'multi_phase': True
            },
            'complex': {
                'max_steps': float('inf'),
                'max_components': float('inf'),
                'has_catalyst': True,
                'multi_phase': True
            }
        }
    
    def classify_reaction(self, reaction_data: Dict[str, Any]) -> str:
        """
        Classify reaction complexity
        
        Args:
            reaction_data: Parsed reaction data
            
        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'
        """
        # Count total components
        total_components = 0
        for component_type in ['reactants', 'reagents', 'solvents', 'products']:
            total_components += len(reaction_data.get(component_type, []))
        
        # Count steps (workups)
        num_steps = len(reaction_data.get('workups', []))
        
        # Check for catalyst
        has_catalyst = len(reaction_data.get('catalysts', [])) > 0
        
        # Check for multi-phase (presence of multiple solvents or specific workup types)
        multi_phase = (len(reaction_data.get('solvents', [])) > 1 or
                      any('extraction' in w.get('type', '').lower() 
                          for w in reaction_data.get('workups', [])))
        
        # Classify based on criteria
        if (num_steps <= 1 and total_components <= 5 and 
            not has_catalyst and not multi_phase):
            return 'simple'
        elif (num_steps <= 2 and total_components <= 8):
            return 'moderate'
        else:
            return 'complex'


class AblationStudy:
    """
    Comprehensive ablation study for reaction extraction system
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ablation study
        
        Args:
            model_path: Path to the fine-tuned model
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.config = config or {}
        
        # Initialize components
        self.complexity_classifier = ReactionComplexityClassifier()
        self.metrics_calculator = MetricsCalculator()
        self.error_analyzer = ErrorAnalyzer()
        
        # Define ablation configurations
        self.ablation_configs = {
            'direct_extraction': {
                'use_cot': False,
                'use_structured_output': False,
                'use_meta_prompt': False,
                'use_self_grounding': False,
                'use_dynamic_prompt': False
            },
            'structured_output': {
                'use_cot': False,
                'use_structured_output': True,
                'use_meta_prompt': False,
                'use_self_grounding': False,
                'use_dynamic_prompt': False
            },
            'meta_prompt': {
                'use_cot': False,
                'use_structured_output': True,
                'use_meta_prompt': True,
                'use_self_grounding': False,
                'use_dynamic_prompt': False
            },
            'chain_of_thought': {
                'use_cot': True,
                'use_structured_output': True,
                'use_meta_prompt': False,
                'use_self_grounding': False,
                'use_dynamic_prompt': False
            },
            'cot_reflection': {
                'use_cot': True,
                'use_structured_output': True,
                'use_meta_prompt': False,
                'use_self_grounding': False,
                'use_dynamic_prompt': False,
                'use_reflection': True
            },
            'self_grounding': {
                'use_cot': False,
                'use_structured_output': True,
                'use_meta_prompt': False,
                'use_self_grounding': True,
                'use_dynamic_prompt': False
            },
            'complete_framework': {
                'use_cot': True,
                'use_structured_output': True,
                'use_meta_prompt': True,
                'use_self_grounding': True,
                'use_dynamic_prompt': True
            },
            'iterative_refinement': {
                'use_cot': True,
                'use_structured_output': True,
                'use_meta_prompt': True,
                'use_self_grounding': True,
                'use_dynamic_prompt': True,
                'use_iterative_refinement': True
            }
        }
        
        logger.info(f"Ablation study initialized with {len(self.ablation_configs)} configurations")
    
    def run_complete_study(self, 
                          test_data: List[Dict[str, Any]],
                          ground_truth: List[Dict[str, Any]],
                          sample_size: int = 1000,
                          stratified: bool = True) -> Dict[str, Any]:
        """
        Run complete ablation study
        
        Args:
            test_data: List of test procedures
            ground_truth: Corresponding ground truth data
            sample_size: Number of samples to evaluate
            stratified: Whether to use stratified sampling by complexity
            
        Returns:
            Complete ablation study results
        """
        logger.info(f"Starting complete ablation study with {sample_size} samples")
        
        # Prepare stratified sample if requested
        if stratified:
            test_sample, truth_sample = self._prepare_stratified_sample(
                test_data, ground_truth, sample_size
            )
        else:
            indices = np.random.choice(len(test_data), sample_size, replace=False)
            test_sample = [test_data[i] for i in indices]
            truth_sample = [ground_truth[i] for i in indices]
        
        # Run ablation for each configuration
        results = {}
        for config_name, config in self.ablation_configs.items():
            logger.info(f"Running ablation: {config_name}")
            
            try:
                result = self._run_single_ablation(
                    config_name, config, test_sample, truth_sample
                )
                results[config_name] = result
                
                logger.info(f"Completed {config_name}: CRA={result.cra:.3f}, "
                           f"Entity F1={result.entity_f1:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to run ablation {config_name}: {str(e)}")
                results[config_name] = None
        
        # Calculate component contributions
        component_analysis = self._analyze_component_contributions(results)
        
        # Analyze complexity-stratified performance
        complexity_analysis = self._analyze_complexity_performance(
            test_sample, truth_sample, results
        )
        
        return {
            'ablation_results': results,
            'component_analysis': component_analysis,
            'complexity_analysis': complexity_analysis,
            'study_metadata': {
                'total_samples': sample_size,
                'stratified_sampling': stratified,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _prepare_stratified_sample(self, 
                                  test_data: List[Dict],
                                  ground_truth: List[Dict],
                                  sample_size: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare stratified sample by reaction complexity
        
        Args:
            test_data: Full test dataset
            ground_truth: Full ground truth dataset
            sample_size: Desired sample size
            
        Returns:
            Stratified test and ground truth samples
        """
        logger.info("Preparing stratified sample by reaction complexity")
        
        # Classify all reactions by complexity
        complexity_indices = {'simple': [], 'moderate': [], 'complex': []}
        
        for i, truth in enumerate(ground_truth):
            complexity = self.complexity_classifier.classify_reaction(truth)
            complexity_indices[complexity].append(i)
        
        # Target distribution: 40% simple, 35% moderate, 25% complex
        target_distribution = {'simple': 0.40, 'moderate': 0.35, 'complex': 0.25}
        
        # Sample from each complexity level
        sampled_indices = []
        for complexity, target_ratio in target_distribution.items():
            available_indices = complexity_indices[complexity]
            target_count = int(sample_size * target_ratio)
            
            if len(available_indices) >= target_count:
                selected = np.random.choice(available_indices, target_count, replace=False)
            else:
                # Use all available if not enough samples
                selected = available_indices
                logger.warning(f"Only {len(available_indices)} {complexity} reactions available, "
                             f"target was {target_count}")
            
            sampled_indices.extend(selected)
        
        # Create stratified samples
        test_sample = [test_data[i] for i in sampled_indices]
        truth_sample = [ground_truth[i] for i in sampled_indices]
        
        logger.info(f"Created stratified sample with {len(sampled_indices)} reactions")
        return test_sample, truth_sample
    
    def _run_single_ablation(self, 
                            config_name: str,
                            config: Dict[str, Any],
                            test_sample: List[Dict],
                            truth_sample: List[Dict]) -> AblationResult:
        """
        Run single ablation configuration
        
        Args:
            config_name: Name of the configuration
            config: Configuration parameters
            test_sample: Test data sample
            truth_sample: Ground truth sample
            
        Returns:
            Ablation results
        """
        # Create extractor with specific configuration
        extractor_config = self.config.copy()
        extractor_config.update(config)
        
        extractor = ChemistryReactionExtractor(
            model_path=self.model_path,
            config=extractor_config
        )
        
        # Run predictions
        predictions = []
        inference_times = []
        
        for i, sample in enumerate(test_sample):
            procedure_text = sample.get('procedure_text', '')
            
            start_time = time.time()
            try:
                result = extractor.analyze_procedure(procedure_text, return_raw=False)
                prediction = result.get('extracted_data', {})
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {str(e)}")
                predictions.append({})
            
            inference_times.append(time.time() - start_time)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            predictions, truth_sample
        )
        
        # Analyze performance by complexity
        complexity_performance = self._calculate_complexity_performance(
            predictions, truth_sample, test_sample
        )
        
        # Calculate average inference time
        avg_inference_time = np.mean(inference_times)
        
        return AblationResult(
            method_name=config_name,
            cra=metrics['complete_reaction_accuracy'],
            entity_f1=metrics['entity_f1'],
            rca=metrics['role_classification_accuracy'],
            condition_f1=metrics['condition_f1'],
            inference_time=avg_inference_time,
            complexity_performance=complexity_performance,
            component_contributions={}  # Will be filled later
        )
    
    def _calculate_complexity_performance(self, 
                                        predictions: List[Dict],
                                        truth_sample: List[Dict],
                                        test_sample: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance metrics by reaction complexity
        
        Args:
            predictions: Model predictions
            truth_sample: Ground truth data
            test_sample: Test data (for classification)
            
        Returns:
            Performance by complexity level
        """
        complexity_groups = {'simple': [], 'moderate': [], 'complex': []}
        
        # Group by complexity
        for i, (pred, truth, test) in enumerate(zip(predictions, truth_sample, test_sample)):
            complexity = self.complexity_classifier.classify_reaction(truth)
            complexity_groups[complexity].append((pred, truth))
        
        # Calculate CRA for each complexity level
        complexity_performance = {}
        for complexity, group in complexity_groups.items():
            if group:
                group_predictions = [item[0] for item in group]
                group_truth = [item[1] for item in group]
                
                group_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                    group_predictions, group_truth
                )
                complexity_performance[complexity] = group_metrics['complete_reaction_accuracy']
            else:
                complexity_performance[complexity] = 0.0
        
        return complexity_performance
    
    def _analyze_component_contributions(self, 
                                       results: Dict[str, AblationResult]) -> Dict[str, Any]:
        """
        Analyze individual component contributions
        
        Args:
            results: Ablation results for all configurations
            
        Returns:
            Component contribution analysis
        """
        if not results or 'direct_extraction' not in results:
            return {}
        
        baseline_cra = results['direct_extraction'].cra
        contributions = {}
        
        # Calculate incremental contributions
        component_improvements = {
            'structured_output': results.get('structured_output', {}).cra - baseline_cra,
            'meta_prompt': results.get('meta_prompt', {}).cra - results.get('structured_output', {}).cra,
            'chain_of_thought': results.get('chain_of_thought', {}).cra - baseline_cra,
            'self_grounding': results.get('self_grounding', {}).cra - baseline_cra,
            'complete_framework': results.get('complete_framework', {}).cra - baseline_cra
        }
        
        # Filter out None values and calculate relative improvements
        for component, improvement in component_improvements.items():
            if improvement is not None:
                contributions[component] = {
                    'absolute_improvement': improvement,
                    'relative_improvement': (improvement / baseline_cra) * 100 if baseline_cra > 0 else 0
                }
        
        return contributions
    
    def _analyze_complexity_performance(self, 
                                      test_sample: List[Dict],
                                      truth_sample: List[Dict],
                                      results: Dict[str, AblationResult]) -> Dict[str, Any]:
        """
        Analyze performance across complexity levels
        
        Args:
            test_sample: Test data
            truth_sample: Ground truth data
            results: Ablation results
            
        Returns:
            Complexity-stratified analysis
        """
        complexity_analysis = {}
        
        for method_name, result in results.items():
            if result is not None:
                complexity_analysis[method_name] = result.complexity_performance
        
        return complexity_analysis
    
    def analyze_dynamic_prompt_components(self, 
                                        test_sample: List[Dict],
                                        truth_sample: List[Dict]) -> Dict[str, Any]:
        """
        Analyze individual dynamic prompt selection components
        
        Args:
            test_sample: Test data sample
            truth_sample: Ground truth sample
            
        Returns:
            Dynamic prompt component analysis
        """
        logger.info("Analyzing dynamic prompt selection components")
        
        # Define component configurations
        component_configs = {
            'reaction_type_only': {'use_reaction_type': True, 'use_complexity': False, 
                                 'use_entity_density': False, 'use_linguistic': False},
            'complexity_only': {'use_reaction_type': False, 'use_complexity': True,
                               'use_entity_density': False, 'use_linguistic': False},
            'entity_density_only': {'use_reaction_type': False, 'use_complexity': False,
                                   'use_entity_density': True, 'use_linguistic': False},
            'linguistic_only': {'use_reaction_type': False, 'use_complexity': False,
                               'use_entity_density': False, 'use_linguistic': True},
            'full_dynamic': {'use_reaction_type': True, 'use_complexity': True,
                           'use_entity_density': True, 'use_linguistic': True}
        }
        
        component_results = {}
        baseline_performance = None
        
        for component_name, component_config in component_configs.items():
            # Create specialized extractor
            extractor_config = self.config.copy()
            extractor_config.update({
                'use_cot': True,
                'use_dynamic_prompt': True,
                'dynamic_prompt_config': component_config
            })
            
            extractor = ChemistryReactionExtractor(
                model_path=self.model_path,
                config=extractor_config
            )
            
            # Run evaluation
            predictions = []
            for sample in test_sample:
                try:
                    result = extractor.analyze_procedure(
                        sample.get('procedure_text', ''), 
                        return_raw=False
                    )
                    predictions.append(result.get('extracted_data', {}))
                except Exception as e:
                    logger.warning(f"Failed to process sample: {str(e)}")
                    predictions.append({})
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                predictions, truth_sample
            )
            
            component_results[component_name] = {
                'cra': metrics['complete_reaction_accuracy'],
                'entity_f1': metrics['entity_f1'],
                'improvement_over_baseline': 0.0  # Will be calculated
            }
            
            if component_name == 'full_dynamic':
                baseline_performance = metrics['complete_reaction_accuracy']
        
        # Calculate improvements over baseline
        if baseline_performance:
            for component_name in component_results:
                improvement = component_results[component_name]['cra'] - baseline_performance
                component_results[component_name]['improvement_over_baseline'] = improvement
        
        return component_results
    
    def generate_ablation_report(self, 
                               study_results: Dict[str, Any],
                               output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive ablation study report
        
        Args:
            study_results: Complete study results
            output_file: Optional output file path
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "CHEMISTRY REACTION EXTRACTION - ABLATION STUDY REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Samples: {study_results['study_metadata']['total_samples']}",
            f"Stratified Sampling: {study_results['study_metadata']['stratified_sampling']}",
            ""
        ])
        
        # Performance Summary Table
        report_lines.extend([
            "PERFORMANCE SUMMARY",
            "-" * 50
        ])
        
        ablation_results = study_results['ablation_results']
        
        # Create results table
        header = f"{'Method':<25} {'CRA (%)':<8} {'Entity F1':<9} {'RCA (%)':<8} {'Cond F1':<8} {'Time (s)':<8}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for method_name, result in ablation_results.items():
            if result:
                line = (f"{method_name:<25} "
                       f"{result.cra*100:<8.1f} "
                       f"{result.entity_f1:<9.3f} "
                       f"{result.rca*100:<8.1f} "
                       f"{result.condition_f1:<8.3f} "
                       f"{result.inference_time:<8.2f}")
                report_lines.append(line)
        
        report_lines.append("")
        
        # Component Contributions
        if 'component_analysis' in study_results:
            report_lines.extend([
                "COMPONENT CONTRIBUTIONS",
                "-" * 30
            ])
            
            for component, contribution in study_results['component_analysis'].items():
                improvement = contribution.get('absolute_improvement', 0)
                relative = contribution.get('relative_improvement', 0)
                report_lines.append(f"{component:<25}: +{improvement:.3f} ({relative:+.1f}%)")
            
            report_lines.append("")
        
        # Complexity Analysis
        if 'complexity_analysis' in study_results:
            report_lines.extend([
                "PERFORMANCE BY REACTION COMPLEXITY",
                "-" * 40
            ])
            
            complexity_header = f"{'Method':<25} {'Simple':<8} {'Moderate':<10} {'Complex':<8}"
            report_lines.append(complexity_header)
            report_lines.append("-" * len(complexity_header))
            
            for method_name, complexity_perf in study_results['complexity_analysis'].items():
                if complexity_perf:
                    line = (f"{method_name:<25} "
                           f"{complexity_perf.get('simple', 0)*100:<8.1f} "
                           f"{complexity_perf.get('moderate', 0)*100:<10.1f} "
                           f"{complexity_perf.get('complex', 0)*100:<8.1f}")
                    report_lines.append(line)
        
        # Generate report string
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Ablation report saved to {output_file}")
        
        return report_text
    
    def export_results_to_csv(self, 
                             study_results: Dict[str, Any],
                             output_file: str):
        """
        Export ablation results to CSV format
        
        Args:
            study_results: Complete study results
            output_file: Output CSV file path
        """
        # Prepare data for CSV
        csv_data = []
        
        for method_name, result in study_results['ablation_results'].items():
            if result:
                row = {
                    'Method': method_name,
                    'CRA': result.cra,
                    'Entity_F1': result.entity_f1,
                    'RCA': result.rca,
                    'Condition_F1': result.condition_f1,
                    'Inference_Time': result.inference_time,
                    'Simple_Reactions': result.complexity_performance.get('simple', 0),
                    'Moderate_Reactions': result.complexity_performance.get('moderate', 0),
                    'Complex_Reactions': result.complexity_performance.get('complex', 0)
                }
                csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Ablation results exported to {output_file}")
        
        return df
