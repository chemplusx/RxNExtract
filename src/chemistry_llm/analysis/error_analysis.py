"""
Error analysis module for chemistry LLM inference
Provides comprehensive error categorization and analysis capabilities
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.xml_parser import parse_reaction_xml

logger = get_logger(__name__)


@dataclass
class ErrorMetrics:
    """Container for error analysis metrics"""
    error_type: str
    baseline_rate: float
    cot_prompt_rate: float
    hybrid_rate: float
    error_reduction: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.error_type,
            'baseline_rate': self.baseline_rate,
            'cot_prompt_rate': self.cot_prompt_rate,
            'hybrid_rate': self.hybrid_rate,
            'error_reduction': self.error_reduction
        }


class ErrorAnalyzer:
    """
    Comprehensive error analysis for chemical reaction extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.error_categories = {
            'entity_recognition': [
                'missing_entities',
                'false_positives',
                'incorrect_entity_type'
            ],
            'role_classification': [
                'reactant_product_confusion',
                'catalyst_misidentification',
                'solvent_misclassification',
                'reagent_role_error'
            ],
            'condition_extraction': [
                'missing_temperature',
                'missing_time',
                'missing_catalyst',
                'incomplete_procedures',
                'incorrect_stoichiometry'
            ],
            'cot_reasoning': [
                'implicit_condition_interpretation',
                'generic_entity_handling',
                'multi_step_confusion',
                'nomenclature_ambiguity',
                'complex_stoichiometry_errors'
            ]
        }
    
    def analyze_prediction_errors(self, 
                                 predictions: List[Dict[str, Any]], 
                                 ground_truth: List[Dict[str, Any]],
                                 method_name: str = "baseline") -> Dict[str, Any]:
        """
        Analyze errors in predictions compared to ground truth
        
        Args:
            predictions: List of predicted reaction data
            ground_truth: List of ground truth reaction data
            method_name: Name of the method being analyzed
            
        Returns:
            Comprehensive error analysis results
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        logger.info(f"Analyzing errors for {len(predictions)} predictions using {method_name}")
        
        error_counts = defaultdict(int)
        total_samples = len(predictions)
        detailed_errors = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            sample_errors = self._analyze_single_prediction(pred, truth, i)
            detailed_errors.append(sample_errors)
            
            # Count error types
            for error_type in sample_errors['errors']:
                error_counts[error_type] += 1
        
        # Calculate error rates
        error_rates = {
            error_type: count / total_samples * 100 
            for error_type, count in error_counts.items()
        }
        
        # Categorize errors
        categorized_errors = self._categorize_errors(error_rates)
        
        return {
            'method_name': method_name,
            'total_samples': total_samples,
            'error_rates': error_rates,
            'categorized_errors': categorized_errors,
            'detailed_errors': detailed_errors,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_single_prediction(self, 
                                  prediction: Dict[str, Any], 
                                  ground_truth: Dict[str, Any],
                                  sample_id: int) -> Dict[str, Any]:
        """
        Analyze errors in a single prediction
        
        Args:
            prediction: Predicted reaction data
            ground_truth: Ground truth reaction data
            sample_id: Sample identifier
            
        Returns:
            Error analysis for single prediction
        """
        errors = []
        error_details = {}
        
        # Entity recognition errors
        entity_errors = self._check_entity_errors(prediction, ground_truth)
        errors.extend(entity_errors['error_types'])
        error_details['entity_errors'] = entity_errors
        
        # Role classification errors
        role_errors = self._check_role_errors(prediction, ground_truth)
        errors.extend(role_errors['error_types'])
        error_details['role_errors'] = role_errors
        
        # Condition extraction errors
        condition_errors = self._check_condition_errors(prediction, ground_truth)
        errors.extend(condition_errors['error_types'])
        error_details['condition_errors'] = condition_errors
        
        return {
            'sample_id': sample_id,
            'errors': errors,
            'error_details': error_details,
            'total_errors': len(errors)
        }
    
    def _check_entity_errors(self, prediction: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """Check for entity recognition errors"""
        errors = []
        details = {}
        
        # Extract entity sets
        pred_entities = self._extract_entities(prediction)
        true_entities = self._extract_entities(ground_truth)
        
        # Missing entities
        missing = true_entities - pred_entities
        if missing:
            errors.append('missing_entities')
            details['missing_entities'] = list(missing)
        
        # False positives
        false_positives = pred_entities - true_entities
        if false_positives:
            errors.append('false_positives')
            details['false_positives'] = list(false_positives)
        
        # Entity type errors
        type_errors = self._check_entity_type_errors(prediction, ground_truth)
        if type_errors:
            errors.append('incorrect_entity_type')
            details['type_errors'] = type_errors
        
        return {
            'error_types': errors,
            'details': details,
            'missing_count': len(missing),
            'false_positive_count': len(false_positives)
        }
    
    def _check_role_errors(self, prediction: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """Check for role classification errors"""
        errors = []
        details = {}
        
        # Check reactant/product confusion
        reactant_confusion = self._check_reactant_product_confusion(prediction, ground_truth)
        if reactant_confusion:
            errors.append('reactant_product_confusion')
            details['reactant_product_confusion'] = reactant_confusion
        
        # Check catalyst misidentification
        catalyst_errors = self._check_catalyst_errors(prediction, ground_truth)
        if catalyst_errors:
            errors.append('catalyst_misidentification')
            details['catalyst_errors'] = catalyst_errors
        
        # Check solvent misclassification
        solvent_errors = self._check_solvent_errors(prediction, ground_truth)
        if solvent_errors:
            errors.append('solvent_misclassification')
            details['solvent_errors'] = solvent_errors
        
        return {
            'error_types': errors,
            'details': details
        }
    
    def _check_condition_errors(self, prediction: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """Check for condition extraction errors"""
        errors = []
        details = {}
        
        # Temperature extraction
        if not self._has_temperature(prediction) and self._has_temperature(ground_truth):
            errors.append('missing_temperature')
        
        # Time/duration extraction
        if not self._has_time(prediction) and self._has_time(ground_truth):
            errors.append('missing_time')
        
        # Catalyst extraction
        if not self._has_catalyst(prediction) and self._has_catalyst(ground_truth):
            errors.append('missing_catalyst')
        
        # Procedure completeness
        if self._is_incomplete_procedure(prediction, ground_truth):
            errors.append('incomplete_procedures')
        
        return {
            'error_types': errors,
            'details': details
        }
    
    def _extract_entities(self, reaction_data: Dict) -> set:
        """Extract all entities from reaction data"""
        entities = set()
        
        for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
            if component_type in reaction_data:
                for component in reaction_data[component_type]:
                    if isinstance(component, dict) and 'name' in component:
                        entities.add(component['name'].lower().strip())
                    elif isinstance(component, str):
                        entities.add(component.lower().strip())
        
        return entities
    
    def _check_entity_type_errors(self, prediction: Dict, ground_truth: Dict) -> List[Dict]:
        """Check for entity type classification errors"""
        type_errors = []
        
        # Build entity type maps
        pred_types = self._build_entity_type_map(prediction)
        true_types = self._build_entity_type_map(ground_truth)
        
        # Find type mismatches
        for entity in pred_types:
            if entity in true_types and pred_types[entity] != true_types[entity]:
                type_errors.append({
                    'entity': entity,
                    'predicted_type': pred_types[entity],
                    'true_type': true_types[entity]
                })
        
        return type_errors
    
    def _build_entity_type_map(self, reaction_data: Dict) -> Dict[str, str]:
        """Build mapping from entity name to type"""
        entity_types = {}
        
        for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
            if component_type in reaction_data:
                for component in reaction_data[component_type]:
                    name = component.get('name', '') if isinstance(component, dict) else component
                    if name:
                        entity_types[name.lower().strip()] = component_type
        
        return entity_types
    
    def _check_reactant_product_confusion(self, prediction: Dict, ground_truth: Dict) -> List[str]:
        """Check for reactant/product role confusion"""
        confused_entities = []
        
        pred_reactants = {r.get('name', '').lower() for r in prediction.get('reactants', [])}
        pred_products = {p.get('name', '').lower() for p in prediction.get('products', [])}
        
        true_reactants = {r.get('name', '').lower() for r in ground_truth.get('reactants', [])}
        true_products = {p.get('name', '').lower() for p in ground_truth.get('products', [])}
        
        # Reactants classified as products
        confused_entities.extend(pred_products & true_reactants)
        
        # Products classified as reactants
        confused_entities.extend(pred_reactants & true_products)
        
        return confused_entities
    
    def _check_catalyst_errors(self, prediction: Dict, ground_truth: Dict) -> List[str]:
        """Check for catalyst misidentification"""
        pred_catalysts = {c.get('name', '').lower() for c in prediction.get('catalysts', [])}
        true_catalysts = {c.get('name', '').lower() for c in ground_truth.get('catalysts', [])}
        
        # Non-catalysts classified as catalysts
        all_true_non_catalysts = self._extract_entities(ground_truth) - true_catalysts
        misidentified = list(pred_catalysts & all_true_non_catalysts)
        
        return misidentified
    
    def _check_solvent_errors(self, prediction: Dict, ground_truth: Dict) -> List[str]:
        """Check for solvent misclassification"""
        pred_solvents = {s.get('name', '').lower() for s in prediction.get('solvents', [])}
        true_solvents = {s.get('name', '').lower() for s in ground_truth.get('solvents', [])}
        
        # Non-solvents classified as solvents
        all_true_non_solvents = self._extract_entities(ground_truth) - true_solvents
        misclassified = list(pred_solvents & all_true_non_solvents)
        
        return misclassified
    
    def _has_temperature(self, reaction_data: Dict) -> bool:
        """Check if reaction data contains temperature information"""
        # Check in conditions
        for condition in reaction_data.get('conditions', []):
            if 'temperature' in condition.get('type', '').lower():
                return True
        
        # Check in workups
        for workup in reaction_data.get('workups', []):
            if workup.get('temperature'):
                return True
        
        return False
    
    def _has_time(self, reaction_data: Dict) -> bool:
        """Check if reaction data contains time/duration information"""
        # Check in conditions
        for condition in reaction_data.get('conditions', []):
            if 'time' in condition.get('type', '').lower() or condition.get('duration'):
                return True
        
        # Check in workups
        for workup in reaction_data.get('workups', []):
            if workup.get('duration'):
                return True
        
        return False
    
    def _has_catalyst(self, reaction_data: Dict) -> bool:
        """Check if reaction data contains catalyst information"""
        return len(reaction_data.get('catalysts', [])) > 0
    
    def _is_incomplete_procedure(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Check if procedure extraction is incomplete"""
        # Compare number of steps/workups
        pred_workups = len(prediction.get('workups', []))
        true_workups = len(ground_truth.get('workups', []))
        
        # Consider incomplete if missing >50% of workup steps
        return pred_workups < (true_workups * 0.5)
    
    def _categorize_errors(self, error_rates: Dict[str, float]) -> Dict[str, Dict]:
        """Categorize errors by type"""
        categorized = {}
        
        for category, error_types in self.error_categories.items():
            category_errors = {}
            for error_type in error_types:
                if error_type in error_rates:
                    category_errors[error_type] = error_rates[error_type]
            
            if category_errors:
                categorized[category] = category_errors
        
        return categorized
    
    def compare_methods(self, 
                       method_results: Dict[str, Dict]) -> List[ErrorMetrics]:
        """
        Compare error rates across different methods
        
        Args:
            method_results: Dictionary mapping method names to their error analysis results
            
        Returns:
            List of ErrorMetrics comparing methods
        """
        if len(method_results) < 2:
            raise ValueError("Need at least 2 methods to compare")
        
        # Assume first method is baseline
        method_names = list(method_results.keys())
        baseline_name = method_names[0]
        baseline_rates = method_results[baseline_name]['error_rates']
        
        comparisons = []
        
        for method_name in method_names[1:]:
            method_rates = method_results[method_name]['error_rates']
            
            for error_type in baseline_rates:
                if error_type in method_rates:
                    baseline_rate = baseline_rates[error_type]
                    method_rate = method_rates[error_type]
                    
                    # Calculate error reduction percentage
                    if baseline_rate > 0:
                        error_reduction = ((baseline_rate - method_rate) / baseline_rate) * 100
                    else:
                        error_reduction = 0.0
                    
                    comparisons.append(ErrorMetrics(
                        error_type=error_type,
                        baseline_rate=baseline_rate,
                        cot_prompt_rate=method_rate,  # Generic field name
                        hybrid_rate=method_rate,     # Will be updated for 3-way comparison
                        error_reduction=error_reduction
                    ))
        
        return comparisons
    
    def analyze_cot_failures(self, 
                            predictions: List[Dict], 
                            ground_truth: List[Dict],
                            raw_outputs: List[str]) -> Dict[str, Any]:
        """
        Analyze Chain-of-Thought reasoning failures
        
        Args:
            predictions: Extracted predictions
            ground_truth: Ground truth data
            raw_outputs: Raw model outputs for analysis
            
        Returns:
            CoT failure analysis
        """
        logger.info("Analyzing Chain-of-Thought reasoning failures")
        
        cot_failure_patterns = {
            'implicit_condition_interpretation': 0,
            'generic_entity_handling': 0,
            'multi_step_confusion': 0,
            'nomenclature_ambiguity': 0,
            'complex_stoichiometry_errors': 0
        }
        
        total_failures = 0
        detailed_failures = []
        
        for i, (pred, truth, raw_output) in enumerate(zip(predictions, ground_truth, raw_outputs)):
            is_failure = self._is_extraction_failure(pred, truth)
            
            if is_failure:
                total_failures += 1
                failure_modes = self._analyze_cot_failure_modes(pred, truth, raw_output)
                detailed_failures.append({
                    'sample_id': i,
                    'failure_modes': failure_modes,
                    'raw_output': raw_output[:500]  # Truncate for storage
                })
                
                # Count failure patterns
                for mode in failure_modes:
                    if mode in cot_failure_patterns:
                        cot_failure_patterns[mode] += 1
        
        # Calculate frequencies and CRA impact
        failure_frequencies = {
            mode: (count / total_failures * 100) if total_failures > 0 else 0
            for mode, count in cot_failure_patterns.items()
        }
        
        # Estimate CRA impact (simplified calculation)
        cra_impact = self._estimate_cra_impact(cot_failure_patterns, total_failures)
        
        return {
            'total_samples': len(predictions),
            'total_failures': total_failures,
            'failure_rate': (total_failures / len(predictions)) * 100,
            'failure_frequencies': failure_frequencies,
            'cra_impact': cra_impact,
            'detailed_failures': detailed_failures
        }
    
    def _is_extraction_failure(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Determine if extraction is considered a failure"""
        # Simple failure criteria - can be made more sophisticated
        pred_entities = self._extract_entities(prediction)
        true_entities = self._extract_entities(ground_truth)
        
        # Failure if <50% of entities correctly extracted
        if len(true_entities) == 0:
            return len(pred_entities) > 0  # False positives
        
        recall = len(pred_entities & true_entities) / len(true_entities)
        return recall < 0.5
    
    def _analyze_cot_failure_modes(self, 
                                  prediction: Dict, 
                                  ground_truth: Dict, 
                                  raw_output: str) -> List[str]:
        """Analyze specific CoT failure modes"""
        failure_modes = []
        
        # Check for implicit condition interpretation
        if self._has_implicit_condition_failure(raw_output, ground_truth):
            failure_modes.append('implicit_condition_interpretation')
        
        # Check for generic entity handling
        if self._has_generic_entity_failure(raw_output, prediction):
            failure_modes.append('generic_entity_handling')
        
        # Check for multi-step confusion
        if self._has_multi_step_confusion(raw_output, ground_truth):
            failure_modes.append('multi_step_confusion')
        
        # Check for nomenclature ambiguity
        if self._has_nomenclature_ambiguity(raw_output, prediction):
            failure_modes.append('nomenclature_ambiguity')
        
        # Check for stoichiometry errors
        if self._has_stoichiometry_errors(prediction, ground_truth):
            failure_modes.append('complex_stoichiometry_errors')
        
        return failure_modes
    
    def _has_implicit_condition_failure(self, raw_output: str, ground_truth: Dict) -> bool:
        """Check for implicit condition interpretation failures"""
        # Look for mentions of conditions in reasoning but missing from extraction
        condition_keywords = ['temperature', 'heat', 'cool', 'pressure', 'reflux']
        
        has_reasoning_conditions = any(keyword in raw_output.lower() for keyword in condition_keywords)
        has_extracted_conditions = len(ground_truth.get('conditions', [])) > 0
        
        return has_reasoning_conditions and not has_extracted_conditions
    
    def _has_generic_entity_failure(self, raw_output: str, prediction: Dict) -> bool:
        """Check for generic entity handling failures"""
        generic_terms = ['compound', 'substance', 'material', 'chemical', 'reagent']
        
        # Check if generic terms appear in reasoning
        has_generic_reasoning = any(term in raw_output.lower() for term in generic_terms)
        
        # Check if extraction contains specific entities
        entities = self._extract_entities(prediction)
        has_specific_entities = any(len(entity) > 5 for entity in entities)  # Rough heuristic
        
        return has_generic_reasoning and not has_specific_entities
    
    def _has_multi_step_confusion(self, raw_output: str, ground_truth: Dict) -> bool:
        """Check for multi-step procedure confusion"""
        step_indicators = ['step', 'then', 'next', 'subsequently', 'followed by']
        
        has_multi_step_reasoning = sum(1 for indicator in step_indicators 
                                     if indicator in raw_output.lower()) >= 2
        
        # Should have multiple workup steps for multi-step procedures
        expected_workups = len(ground_truth.get('workups', []))
        
        return has_multi_step_reasoning and expected_workups >= 3
    
    def _has_nomenclature_ambiguity(self, raw_output: str, prediction: Dict) -> bool:
        """Check for nomenclature ambiguity issues"""
        # Look for uncertainty language in reasoning
        uncertainty_phrases = ['unclear', 'ambiguous', 'might be', 'could be', 'uncertain']
        
        has_uncertainty = any(phrase in raw_output.lower() for phrase in uncertainty_phrases)
        
        return has_uncertainty
    
    def _has_stoichiometry_errors(self, prediction: Dict, ground_truth: Dict) -> bool:
        """Check for stoichiometry errors"""
        # Compare amounts/stoichiometry information
        pred_amounts = self._extract_amounts(prediction)
        true_amounts = self._extract_amounts(ground_truth)
        
        # Simple check - significant mismatch in number of quantified components
        amount_difference = abs(len(pred_amounts) - len(true_amounts))
        return amount_difference >= 2
    
    def _extract_amounts(self, reaction_data: Dict) -> List[str]:
        """Extract amount information from reaction data"""
        amounts = []
        
        for component_type in ['reactants', 'reagents', 'solvents', 'products']:
            for component in reaction_data.get(component_type, []):
                if isinstance(component, dict) and component.get('amount'):
                    amounts.append(component['amount'])
        
        return amounts
    
    def _estimate_cra_impact(self, failure_patterns: Dict, total_failures: int) -> Dict[str, float]:
        """Estimate impact on Chemical Reaction Accuracy (CRA)"""
        # Simplified impact estimation based on research data
        impact_weights = {
            'implicit_condition_interpretation': -6.7,
            'generic_entity_handling': -8.4,
            'multi_step_confusion': -12.1,
            'nomenclature_ambiguity': -5.8,
            'complex_stoichiometry_errors': -4.2
        }
        
        cra_impact = {}
        for pattern, count in failure_patterns.items():
            if pattern in impact_weights and total_failures > 0:
                frequency = count / total_failures
                impact = frequency * impact_weights[pattern]
                cra_impact[pattern] = impact
        
        return cra_impact
    
    def generate_error_report(self, 
                             analysis_results: Dict[str, Any],
                             output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive error analysis report
        
        Args:
            analysis_results: Results from error analysis
            output_file: Optional output file path
            
        Returns:
            Formatted error report
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "CHEMISTRY REACTION EXTRACTION - ERROR ANALYSIS REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Method: {analysis_results.get('method_name', 'Unknown')}",
            f"Total Samples: {analysis_results.get('total_samples', 0)}",
            ""
        ])
        
        # Error rates summary
        report_lines.append("ERROR RATES SUMMARY")
        report_lines.append("-" * 40)
        
        error_rates = analysis_results.get('error_rates', {})
        for error_type, rate in sorted(error_rates.items()):
            report_lines.append(f"{error_type:<30}: {rate:6.2f}%")
        
        report_lines.append("")
        
        # Categorized errors
        categorized = analysis_results.get('categorized_errors', {})
        for category, errors in categorized.items():
            report_lines.append(f"{category.upper().replace('_', ' ')}")
            report_lines.append("-" * len(category))
            for error_type, rate in errors.items():
                report_lines.append(f"  {error_type:<28}: {rate:6.2f}%")
            report_lines.append("")
        
        # Generate report string
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Error report saved to {output_file}")
        
        return report_text
