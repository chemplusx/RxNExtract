"""
Comprehensive metrics calculation for chemistry LLM evaluation
Provides statistical analysis, confidence calibration, and performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import calibration_curve
from scipy import stats
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculation for reaction extraction evaluation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def calculate_comprehensive_metrics(self, 
                                      predictions: List[Dict[str, Any]], 
                                      ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            predictions: List of predicted reaction data
            ground_truth: List of ground truth reaction data
            
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info(f"Calculating metrics for {len(predictions)} predictions")
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        metrics = {}
        
        # Entity-level metrics
        entity_metrics = self._calculate_entity_metrics(predictions, ground_truth)
        metrics.update(entity_metrics)
        
        # Role classification metrics
        role_metrics = self._calculate_role_classification_metrics(predictions, ground_truth)
        metrics.update(role_metrics)
        
        # Condition extraction metrics
        condition_metrics = self._calculate_condition_metrics(predictions, ground_truth)
        metrics.update(condition_metrics)
        
        # Complete reaction accuracy
        cra = self._calculate_complete_reaction_accuracy(predictions, ground_truth)
        metrics['complete_reaction_accuracy'] = cra
        
        # Confidence-based metrics (if confidence scores available)
        if self._has_confidence_scores(predictions):
            confidence_metrics = self._calculate_confidence_metrics(predictions, ground_truth)
            metrics.update(confidence_metrics)
        
        return metrics
    
    def _calculate_entity_metrics(self, 
                                 predictions: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate entity-level precision, recall, and F1"""
        
        all_pred_entities = []
        all_true_entities = []
        
        # Extract binary labels for all entities
        for pred, truth in zip(predictions, ground_truth):
            pred_entities = self._extract_entity_set(pred)
            true_entities = self._extract_entity_set(truth)
            
            # Create unified entity vocabulary
            all_entities = pred_entities | true_entities
            
            # Convert to binary vectors
            pred_vector = [1 if entity in pred_entities else 0 for entity in all_entities]
            true_vector = [1 if entity in true_entities else 0 for entity in all_entities]
            
            all_pred_entities.extend(pred_vector)
            all_true_entities.extend(true_vector)
        
        # Calculate metrics
        if len(all_pred_entities) == 0:
            return {'entity_precision': 0.0, 'entity_recall': 0.0, 'entity_f1': 0.0}
        
        precision = precision_score(all_true_entities, all_pred_entities, zero_division=0)
        recall = recall_score(all_true_entities, all_pred_entities, zero_division=0)
        f1 = f1_score(all_true_entities, all_pred_entities, zero_division=0)
        
        return {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1
        }
    
    def _calculate_role_classification_metrics(self, 
                                             predictions: List[Dict], 
                                             ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate role classification accuracy"""
        
        correct_roles = 0
        total_entities = 0
        
        for pred, truth in zip(predictions, ground_truth):
            # Build entity-to-role mappings
            pred_roles = self._build_entity_role_map(pred)
            true_roles = self._build_entity_role_map(truth)
            
            # Count correct role assignments for entities present in both
            common_entities = set(pred_roles.keys()) & set(true_roles.keys())
            
            for entity in common_entities:
                total_entities += 1
                if pred_roles[entity] == true_roles[entity]:
                    correct_roles += 1
        
        rca = correct_roles / total_entities if total_entities > 0 else 0.0
        
        return {'role_classification_accuracy': rca}
    
    def _calculate_condition_metrics(self, 
                                   predictions: List[Dict], 
                                   ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate condition extraction metrics"""
        
        all_pred_conditions = []
        all_true_conditions = []
        
        for pred, truth in zip(predictions, ground_truth):
            pred_conditions = self._extract_conditions(pred)
            true_conditions = self._extract_conditions(truth)
            
            # Create unified condition vocabulary
            all_conditions = pred_conditions | true_conditions
            
            # Convert to binary vectors
            pred_vector = [1 if cond in pred_conditions else 0 for cond in all_conditions]
            true_vector = [1 if cond in true_conditions else 0 for cond in all_conditions]
            
            all_pred_conditions.extend(pred_vector)
            all_true_conditions.extend(true_vector)
        
        if len(all_pred_conditions) == 0:
            return {'condition_precision': 0.0, 'condition_recall': 0.0, 'condition_f1': 0.0}
        
        precision = precision_score(all_true_conditions, all_pred_conditions, zero_division=0)
        recall = recall_score(all_true_conditions, all_pred_conditions, zero_division=0)
        f1 = f1_score(all_true_conditions, all_pred_conditions, zero_division=0)
        
        return {
            'condition_precision': precision,
            'condition_recall': recall,
            'condition_f1': f1
        }
    
    def _calculate_complete_reaction_accuracy(self, 
                                            predictions: List[Dict], 
                                            ground_truth: List[Dict]) -> float:
        """
        Calculate Complete Reaction Accuracy (CRA)
        A reaction is considered correct if all components are correctly identified and classified
        """
        correct_reactions = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if self._is_reaction_completely_correct(pred, truth):
                correct_reactions += 1
        
        return correct_reactions / len(predictions) if predictions else 0.0
    
    def _is_reaction_completely_correct(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Check if a reaction is completely correctly extracted"""
        
        # Check entity completeness and correctness
        pred_entities = self._extract_entity_set(prediction)
        true_entities = self._extract_entity_set(ground_truth)
        
        # Require high overlap (>= 80% recall and precision)
        if len(true_entities) == 0:
            entity_correct = len(pred_entities) == 0
        else:
            recall = len(pred_entities & true_entities) / len(true_entities)
            precision = len(pred_entities & true_entities) / len(pred_entities) if pred_entities else 0
            entity_correct = recall >= 0.8 and precision >= 0.8
        
        # Check role classification for common entities
        pred_roles = self._build_entity_role_map(prediction)
        true_roles = self._build_entity_role_map(ground_truth)
        
        common_entities = set(pred_roles.keys()) & set(true_roles.keys())
        role_correct = True
        
        for entity in common_entities:
            if pred_roles[entity] != true_roles[entity]:
                role_correct = False
                break
        
        # Check essential conditions (temperature, time, catalyst if present)
        condition_correct = self._check_essential_conditions(prediction, ground_truth)
        
        return entity_correct and role_correct and condition_correct
    
    def _check_essential_conditions(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Check if essential reaction conditions are correctly extracted"""
        
        essential_conditions = ['temperature', 'time', 'catalyst']
        
        for condition_type in essential_conditions:
            true_has = self._has_condition_type(ground_truth, condition_type)
            pred_has = self._has_condition_type(prediction, condition_type)
            
            # If ground truth has this condition, prediction must have it too
            if true_has and not pred_has:
                return False
        
        return True
    
    def _has_condition_type(self, reaction_data: Dict, condition_type: str) -> bool:
        """Check if reaction data contains specific condition type"""
        
        if condition_type == 'catalyst':
            return len(reaction_data.get('catalysts', [])) > 0
        
        # Check in conditions
        for condition in reaction_data.get('conditions', []):
            if condition_type.lower() in condition.get('type', '').lower():
                return True
        
        # Check in workups
        for workup in reaction_data.get('workups', []):
            if condition_type == 'temperature' and workup.get('temperature'):
                return True
            elif condition_type == 'time' and workup.get('duration'):
                return True
        
        return False
    
    def _extract_entity_set(self, reaction_data: Dict) -> set:
        """Extract all entities from reaction data as a set"""
        entities = set()
        
        for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
            for component in reaction_data.get(component_type, []):
                if isinstance(component, dict) and 'name' in component:
                    entities.add(component['name'].lower().strip())
                elif isinstance(component, str):
                    entities.add(component.lower().strip())
        
        return entities
    
    def _build_entity_role_map(self, reaction_data: Dict) -> Dict[str, str]:
        """Build mapping from entity name to role"""
        entity_roles = {}
        
        for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
            for component in reaction_data.get(component_type, []):
                name = component.get('name', '') if isinstance(component, dict) else component
                if name:
                    entity_roles[name.lower().strip()] = component_type
        
        return entity_roles
    
    def _extract_conditions(self, reaction_data: Dict) -> set:
        """Extract condition identifiers from reaction data"""
        conditions = set()
        
        # Extract from conditions array
        for condition in reaction_data.get('conditions', []):
            cond_type = condition.get('type', '').lower()
            cond_value = condition.get('value', '')
            if cond_type and cond_value:
                conditions.add(f"{cond_type}:{cond_value}")
        
        # Extract from workups
        for workup in reaction_data.get('workups', []):
            if workup.get('temperature'):
                conditions.add(f"temperature:{workup['temperature']}")
            if workup.get('duration'):
                conditions.add(f"time:{workup['duration']}")
        
        return conditions
    
    def _has_confidence_scores(self, predictions: List[Dict]) -> bool:
        """Check if predictions contain confidence scores"""
        return any('confidence' in pred for pred in predictions if pred)
    
    def _calculate_confidence_metrics(self, 
                                    predictions: List[Dict], 
                                    ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate confidence calibration metrics"""
        
        confidences = []
        accuracies = []
        
        for pred, truth in zip(predictions, ground_truth):
            if 'confidence' in pred:
                confidences.append(pred['confidence'])
                accuracies.append(1.0 if self._is_reaction_completely_correct(pred, truth) else 0.0)
        
        if not confidences:
            return {}
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, accuracies)
        
        # Calculate Brier Score
        brier_score = np.mean([(conf - acc)**2 for conf, acc in zip(confidences, accuracies)])
        
        # Calculate reliability coefficient
        reliability = np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 1 else 0.0
        
        return {
            'expected_calibration_error': ece,
            'brier_score': brier_score,
            'reliability_coefficient': reliability
        }
    
    def _calculate_ece(self, confidences: List[float], accuracies: List[float], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Select samples in this bin
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                bin_confidences = [conf for conf, in_b in zip(confidences, in_bin) if in_b]
                bin_accuracies = [acc for acc, in_b in zip(accuracies, in_bin) if in_b]
                
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
    
    def calculate_statistical_significance(self, 
                                         method1_results: List[float],
                                         method2_results: List[float],
                                         test_type: str = 'paired_t') -> Dict[str, Any]:
        """
        Calculate statistical significance between two methods
        
        Args:
            method1_results: Results from first method
            method2_results: Results from second method
            test_type: Type of statistical test ('paired_t', 'wilcoxon', 'mcnemar')
            
        Returns:
            Statistical test results
        """
        if len(method1_results) != len(method2_results):
            raise ValueError("Result lists must have same length")
        
        if test_type == 'paired_t':
            statistic, p_value = stats.ttest_rel(method1_results, method2_results)
            test_name = "Paired t-test"
            
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(method1_results, method2_results)
            test_name = "Wilcoxon signed-rank test"
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        diff = np.array(method1_results) - np.array(method2_results)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
    
    def calculate_mcnemar_test(self, 
                              method1_predictions: List[bool],
                              method2_predictions: List[bool],
                              ground_truth: List[bool]) -> Dict[str, Any]:
        """
        Calculate McNemar's test for comparing two methods
        
        Args:
            method1_predictions: Binary predictions from method 1
            method2_predictions: Binary predictions from method 2
            ground_truth: Ground truth binary labels
            
        Returns:
            McNemar test results
        """
        # Create contingency table
        method1_correct = [pred == truth for pred, truth in zip(method1_predictions, ground_truth)]
        method2_correct = [pred == truth for pred, truth in zip(method2_predictions, ground_truth)]
        
        # Count disagreements
        method1_right_method2_wrong = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if m1 and not m2)
        method1_wrong_method2_right = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if not m1 and m2)
        
        # McNemar's chi-square statistic
        if (method1_right_method2_wrong + method1_wrong_method2_right) > 0:
            chi_square = (method1_right_method2_wrong - method1_wrong_method2_right)**2 / (method1_right_method2_wrong + method1_wrong_method2_right)
            p_value = 1 - stats.chi2.cdf(chi_square, 1)
        else:
            chi_square = 0.0
            p_value = 1.0
        
        return {
            'test_name': "McNemar's test",
            'chi_square': chi_square,
            'p_value': p_value,
            'method1_right_method2_wrong': method1_right_method2_wrong,
            'method1_wrong_method2_right': method1_wrong_method2_right,
            'significant': p_value < 0.05
        }
    
    def calculate_confidence_intervals(self, 
                                     results: List[float],
                                     confidence_level: float = 0.95,
                                     method: str = 'bootstrap') -> Dict[str, float]:
        """
        Calculate confidence intervals for results
        
        Args:
            results: List of result values
            confidence_level: Confidence level (default 0.95)
            method: Method to use ('bootstrap', 'normal')
            
        Returns:
            Confidence interval bounds
        """
        if method == 'bootstrap':
            return self._bootstrap_confidence_interval(results, confidence_level)
        elif method == 'normal':
            return self._normal_confidence_interval(results, confidence_level)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _bootstrap_confidence_interval(self, 
                                     results: List[float],
                                     confidence_level: float,
                                     n_bootstrap: int = 1000) -> Dict[str, float]:
        """Calculate bootstrap confidence interval"""
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(results, size=len(results), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': np.mean(results),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    
    def _normal_confidence_interval(self, 
                                  results: List[float],
                                  confidence_level: float) -> Dict[str, float]:
        """Calculate normal-based confidence interval"""
        
        mean = np.mean(results)
        std_error = stats.sem(results)
        
        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha/2, len(results) - 1)
        
        margin_error = critical_value * std_error
        
        return {
            'mean': mean,
            'lower_bound': mean - margin_error,
            'upper_bound': mean + margin_error,
            'confidence_level': confidence_level
        }
    
    def analyze_performance_by_complexity(self, 
                                        predictions: List[Dict],
                                        ground_truth: List[Dict],
                                        complexity_labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance stratified by reaction complexity
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth data
            complexity_labels: Complexity labels for each sample
            
        Returns:
            Performance metrics by complexity level
        """
        complexity_groups = defaultdict(list)
        
        # Group samples by complexity
        for pred, truth, complexity in zip(predictions, ground_truth, complexity_labels):
            complexity_groups[complexity].append((pred, truth))
        
        # Calculate metrics for each complexity level
        complexity_metrics = {}
        
        for complexity, group in complexity_groups.items():
            if group:
                group_predictions = [item[0] for item in group]
                group_truth = [item[1] for item in group]
                
                metrics = self.calculate_comprehensive_metrics(group_predictions, group_truth)
                complexity_metrics[complexity] = {
                    'sample_count': len(group),
                    'cra': metrics['complete_reaction_accuracy'],
                    'entity_f1': metrics['entity_f1'],
                    'role_accuracy': metrics['role_classification_accuracy'],
                    'condition_f1': metrics['condition_f1']
                }
        
        return complexity_metrics
    
    def calculate_error_reduction(self, 
                                baseline_metrics: Dict[str, float],
                                improved_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate error reduction percentages
        
        Args:
            baseline_metrics: Baseline method metrics
            improved_metrics: Improved method metrics
            
        Returns:
            Error reduction percentages for each metric
        """
        error_reductions = {}
        
        for metric_name in baseline_metrics:
            if metric_name in improved_metrics:
                baseline_error = 1 - baseline_metrics[metric_name]
                improved_error = 1 - improved_metrics[metric_name]
                
                if baseline_error > 0:
                    reduction = ((baseline_error - improved_error) / baseline_error) * 100
                    error_reductions[metric_name] = reduction
                else:
                    error_reductions[metric_name] = 0.0
        
        return error_reductions
    
    def export_metrics_summary(self, 
                              metrics_dict: Dict[str, Any],
                              output_file: str):
        """
        Export metrics summary to JSON file
        
        Args:
            metrics_dict: Dictionary of calculated metrics
            output_file: Output file path
        """
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = self._convert_for_json(metrics_dict)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics summary exported to {output_file}")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
