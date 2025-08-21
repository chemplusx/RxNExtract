"""
Uncertainty quantification and confidence calibration for chemistry LLM
Provides temperature scaling, confidence analysis, and reliability metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration assessment metrics"""
    ece: float  # Expected Calibration Error
    brier_score: float
    reliability: float
    resolution: float
    optimal_temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'ece': self.ece,
            'brier_score': self.brier_score,
            'reliability': self.reliability,
            'resolution': self.resolution,
            'optimal_temperature': self.optimal_temperature
        }


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaling
        
        Args:
            initial_temperature: Initial temperature value
        """
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Input logits
            
        Returns:
            Temperature-scaled probabilities
        """
        return torch.softmax(logits / self.temperature, dim=-1)
    
    def fit(self, 
            logits: torch.Tensor, 
            labels: torch.Tensor,
            max_iter: int = 1000,
            lr: float = 0.01) -> float:
        """
        Fit optimal temperature using validation data
        
        Args:
            logits: Validation logits
            labels: Validation labels
            max_iter: Maximum optimization iterations
            lr: Learning rate
            
        Returns:
            Optimal temperature value
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        optimal_temp = self.temperature.item()
        logger.info(f"Optimal temperature: {optimal_temp:.3f}")
        
        return optimal_temp


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for reaction extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize uncertainty quantifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.temperature_scaler = None
        self.calibration_method = self.config.get('calibration_method', 'temperature_scaling')
        
    def calculate_calibration_metrics(self, 
                                    confidences: List[float],
                                    accuracies: List[float],
                                    n_bins: int = 10) -> CalibrationMetrics:
        """
        Calculate comprehensive calibration metrics
        
        Args:
            confidences: Predicted confidence scores
            accuracies: Binary accuracy labels
            n_bins: Number of bins for ECE calculation
            
        Returns:
            Calibration metrics
        """
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have same length")
        
        # Expected Calibration Error
        ece = self._calculate_ece(confidences, accuracies, n_bins)
        
        # Brier Score
        brier_score = np.mean([(conf - acc)**2 for conf, acc in zip(confidences, accuracies)])
        
        # Reliability and Resolution (decomposition of Brier Score)
        reliability, resolution = self._calculate_reliability_resolution(
            confidences, accuracies, n_bins
        )
        
        return CalibrationMetrics(
            ece=ece,
            brier_score=brier_score,
            reliability=reliability,
            resolution=resolution
        )
    
    def _calculate_ece(self, 
                      confidences: List[float], 
                      accuracies: List[float],
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Identify samples in this bin
            in_bin = [(conf > bin_lower) and (conf <= bin_upper) for conf in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                bin_confidences = [conf for conf, in_b in zip(confidences, in_bin) if in_b]
                bin_accuracies = [acc for acc, in_b in zip(accuracies, in_bin) if in_b]
                
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
    
    def _calculate_reliability_resolution(self, 
                                        confidences: List[float],
                                        accuracies: List[float],
                                        n_bins: int = 10) -> Tuple[float, float]:
        """Calculate reliability and resolution components of Brier Score"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        overall_accuracy = np.mean(accuracies)
        reliability = 0
        resolution = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf > bin_lower) and (conf <= bin_upper) for conf in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                bin_confidences = [conf for conf, in_b in zip(confidences, in_bin) if in_b]
                bin_accuracies = [acc for acc, in_b in zip(accuracies, in_bin) if in_b]
                
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                
                # Reliability: weighted average of squared differences
                reliability += prop_in_bin * (avg_confidence - avg_accuracy)**2
                
                # Resolution: weighted variance of bin accuracies
                resolution += prop_in_bin * (avg_accuracy - overall_accuracy)**2
        
        return reliability, resolution
    
    def perform_temperature_scaling(self, 
                                  validation_logits: np.ndarray,
                                  validation_labels: np.ndarray,
                                  test_logits: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform temperature scaling calibration
        
        Args:
            validation_logits: Validation set logits for finding optimal temperature
            validation_labels: Validation set labels
            test_logits: Test set logits to be calibrated
            
        Returns:
            Calibrated probabilities and optimal temperature
        """
        logger.info("Performing temperature scaling calibration")
        
        # Convert to tensors
        val_logits_tensor = torch.tensor(validation_logits, dtype=torch.float32)
        val_labels_tensor = torch.tensor(validation_labels, dtype=torch.long)
        test_logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
        
        # Initialize and fit temperature scaler
        self.temperature_scaler = TemperatureScaling()
        optimal_temp = self.temperature_scaler.fit(val_logits_tensor, val_labels_tensor)
        
        # Apply to test set
        with torch.no_grad():
            calibrated_probs = self.temperature_scaler(test_logits_tensor)
        
        return calibrated_probs.numpy(), optimal_temp
    
    def analyze_confidence_stratified_performance(self, 
                                                confidences: List[float],
                                                accuracies: List[float],
                                                n_strata: int = 5) -> Dict[str, Any]:
        """
        Analyze performance across confidence strata
        
        Args:
            confidences: Predicted confidence scores
            accuracies: Binary accuracy labels
            n_strata: Number of confidence strata
            
        Returns:
            Stratified performance analysis
        """
        logger.info(f"Analyzing confidence-stratified performance with {n_strata} strata")
        
        # Create confidence bins
        confidence_percentiles = np.linspace(0, 100, n_strata + 1)
        confidence_thresholds = np.percentile(confidences, confidence_percentiles)
        
        strata_analysis = {}
        
        for i in range(n_strata):
            lower_thresh = confidence_thresholds[i]
            upper_thresh = confidence_thresholds[i + 1]
            
            # Select samples in this stratum
            in_stratum = [(conf >= lower_thresh) and (conf <= upper_thresh) 
                         for conf in confidences]
            
            if any(in_stratum):
                stratum_confidences = [conf for conf, in_s in zip(confidences, in_stratum) if in_s]
                stratum_accuracies = [acc for acc, in_s in zip(accuracies, in_stratum) if in_s]
                
                stratum_stats = {
                    'confidence_range': [lower_thresh, upper_thresh],
                    'sample_count': len(stratum_confidences),
                    'coverage': len(stratum_confidences) / len(confidences),
                    'mean_confidence': np.mean(stratum_confidences),
                    'mean_accuracy': np.mean(stratum_accuracies),
                    'calibration_error': abs(np.mean(stratum_confidences) - np.mean(stratum_accuracies))
                }
                
                strata_analysis[f'stratum_{i+1}'] = stratum_stats
        
        return strata_analysis
    
    def calculate_confidence_intervals(self, 
                                     performance_scores: List[float],
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for performance metrics
        
        Args:
            performance_scores: List of performance scores
            confidence_level: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Confidence interval bounds
        """
        logger.info(f"Calculating {confidence_level*100}% confidence intervals")
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(performance_scores, 
                                              size=len(performance_scores), 
                                              replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': np.mean(performance_scores),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'margin_of_error': (upper_bound - lower_bound) / 2
        }
    
    def generate_reliability_diagram(self, 
                                   confidences: List[float],
                                   accuracies: List[float],
                                   n_bins: int = 10,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate reliability diagram for calibration visualization
        
        Args:
            confidences: Predicted confidence scores
            accuracies: Binary accuracy labels
            n_bins: Number of bins
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Generating reliability diagram")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=n_bins
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot reliability diagram
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Model (ECE = {self._calculate_ece(confidences, accuracies, n_bins):.3f})")
        ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
        
        # Add histogram of confidence scores
        ax2 = ax.twinx()
        ax2.hist(confidences, bins=n_bins, alpha=0.3, color='gray', density=True)
        ax2.set_ylabel("Density of confidence scores")
        
        # Formatting
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Reliability Diagram")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig
    
    def perform_platt_scaling(self, 
                            validation_scores: np.ndarray,
                            validation_labels: np.ndarray,
                            test_scores: np.ndarray) -> np.ndarray:
        """
        Perform Platt scaling (sigmoid calibration)
        
        Args:
            validation_scores: Validation set scores for fitting
            validation_labels: Validation set labels
            test_scores: Test set scores to be calibrated
            
        Returns:
            Calibrated probabilities
        """
        logger.info("Performing Platt scaling calibration")
        
        # Fit logistic regression on validation data
        lr = LogisticRegression()
        lr.fit(validation_scores.reshape(-1, 1), validation_labels)
        
        # Apply to test data
        calibrated_probs = lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        
        return calibrated_probs
    
    def perform_isotonic_regression(self, 
                                  validation_scores: np.ndarray,
                                  validation_labels: np.ndarray,
                                  test_scores: np.ndarray) -> np.ndarray:
        """
        Perform isotonic regression calibration
        
        Args:
            validation_scores: Validation set scores for fitting
            validation_labels: Validation set labels
            test_scores: Test set scores to be calibrated
            
        Returns:
            Calibrated probabilities
        """
        logger.info("Performing isotonic regression calibration")
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(validation_scores, validation_labels)
        
        # Apply to test data
        calibrated_probs = iso_reg.predict(test_scores)
        
        return calibrated_probs
    
    def analyze_prediction_uncertainty(self, 
                                     predictions: List[Dict[str, Any]],
                                     ground_truth: List[Dict[str, Any]],
                                     confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze prediction uncertainty and identify high/low confidence samples
        
        Args:
            predictions: Model predictions with confidence scores
            ground_truth: Ground truth data
            confidence_threshold: Threshold for high confidence classification
            
        Returns:
            Uncertainty analysis results
        """
        logger.info("Analyzing prediction uncertainty")
        
        if not all('confidence' in pred for pred in predictions if pred):
            logger.warning("Not all predictions contain confidence scores")
            return {}
        
        high_confidence_indices = []
        medium_confidence_indices = []
        low_confidence_indices = []
        
        confidences = []
        accuracies = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred and 'confidence' in pred:
                conf = pred['confidence']
                confidences.append(conf)
                
                # Calculate accuracy for this sample
                accuracy = self._calculate_sample_accuracy(pred, truth)
                accuracies.append(accuracy)
                
                # Categorize by confidence
                if conf >= confidence_threshold:
                    high_confidence_indices.append(i)
                elif conf >= 0.5:
                    medium_confidence_indices.append(i)
                else:
                    low_confidence_indices.append(i)
        
        # Calculate statistics for each confidence group
        uncertainty_analysis = {
            'total_samples': len(predictions),
            'high_confidence': {
                'threshold': confidence_threshold,
                'count': len(high_confidence_indices),
                'coverage': len(high_confidence_indices) / len(predictions),
                'accuracy': np.mean([accuracies[i] for i in high_confidence_indices]) if high_confidence_indices else 0.0
            },
            'medium_confidence': {
                'threshold_range': [0.5, confidence_threshold],
                'count': len(medium_confidence_indices),
                'coverage': len(medium_confidence_indices) / len(predictions),
                'accuracy': np.mean([accuracies[i] for i in medium_confidence_indices]) if medium_confidence_indices else 0.0
            },
            'low_confidence': {
                'threshold': 0.5,
                'count': len(low_confidence_indices),
                'coverage': len(low_confidence_indices) / len(predictions),
                'accuracy': np.mean([accuracies[i] for i in low_confidence_indices]) if low_confidence_indices else 0.0
            },
            'overall_calibration': self.calculate_calibration_metrics(confidences, accuracies)
        }
        
        return uncertainty_analysis
    
    def _calculate_sample_accuracy(self, prediction: Dict, ground_truth: Dict) -> float:
        """Calculate accuracy for a single sample"""
        
        # Extract entities
        pred_entities = self._extract_entities(prediction)
        true_entities = self._extract_entities(ground_truth)
        
        if len(true_entities) == 0:
            return 1.0 if len(pred_entities) == 0 else 0.0
        
        # Calculate F1 score as accuracy proxy
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
    
    def _extract_entities(self, reaction_data: Dict) -> set:
        """Extract entities from reaction data"""
        entities = set()
        
        for component_type in ['reactants', 'reagents', 'solvents', 'catalysts', 'products']:
            for component in reaction_data.get(component_type, []):
                if isinstance(component, dict) and 'name' in component:
                    entities.add(component['name'].lower().strip())
                elif isinstance(component, str):
                    entities.add(component.lower().strip())
        
        return entities
    
    def generate_uncertainty_report(self, 
                                  uncertainty_results: Dict[str, Any],
                                  output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive uncertainty quantification report
        
        Args:
            uncertainty_results: Results from uncertainty analysis
            output_file: Optional output file path
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "UNCERTAINTY QUANTIFICATION AND CONFIDENCE CALIBRATION REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Samples: {uncertainty_results.get('total_samples', 0)}",
            ""
        ])
        
        # Calibration Metrics
        if 'overall_calibration' in uncertainty_results:
            calibration = uncertainty_results['overall_calibration']
            report_lines.extend([
                "CALIBRATION METRICS",
                "-" * 30,
                f"Expected Calibration Error (ECE): {calibration.ece:.4f}",
                f"Brier Score:                     {calibration.brier_score:.4f}",
                f"Reliability:                     {calibration.reliability:.4f}",
                f"Resolution:                      {calibration.resolution:.4f}",
                ""
            ])
        
        # Confidence-Stratified Performance
        report_lines.extend([
            "CONFIDENCE-STRATIFIED PERFORMANCE",
            "-" * 40
        ])
        
        for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            if confidence_level in uncertainty_results:
                stats = uncertainty_results[confidence_level]
                level_name = confidence_level.replace('_', ' ').title()
                
                report_lines.extend([
                    f"{level_name}:",
                    f"  Count:      {stats['count']}",
                    f"  Coverage:   {stats['coverage']*100:.1f}%",
                    f"  Accuracy:   {stats['accuracy']*100:.1f}%",
                    ""
                ])
        
        # Generate report string
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Uncertainty report saved to {output_file}")
        
        return report_text
    
    def compare_calibration_methods(self, 
                                  validation_scores: np.ndarray,
                                  validation_labels: np.ndarray,
                                  test_scores: np.ndarray,
                                  test_labels: np.ndarray) -> Dict[str, CalibrationMetrics]:
        """
        Compare different calibration methods
        
        Args:
            validation_scores: Validation scores for fitting calibration
            validation_labels: Validation labels
            test_scores: Test scores to be calibrated
            test_labels: Test labels for evaluation
            
        Returns:
            Comparison of calibration methods
        """
        logger.info("Comparing calibration methods")
        
        methods_results = {}
        
        # Original (uncalibrated)
        original_confidences = test_scores.tolist()
        original_accuracies = test_labels.tolist()
        methods_results['uncalibrated'] = self.calculate_calibration_metrics(
            original_confidences, original_accuracies
        )
        
        # Temperature scaling
        try:
            # Convert to logits (assuming scores are probabilities)
            val_logits = np.log(validation_scores / (1 - validation_scores + 1e-8))
            test_logits = np.log(test_scores / (1 - test_scores + 1e-8))
            
            calibrated_probs, optimal_temp = self.perform_temperature_scaling(
                val_logits.reshape(-1, 1), validation_labels, test_logits.reshape(-1, 1)
            )
            
            temp_scaled_metrics = self.calculate_calibration_metrics(
                calibrated_probs[:, 1].tolist(), test_labels.tolist()
            )
            temp_scaled_metrics.optimal_temperature = optimal_temp
            methods_results['temperature_scaling'] = temp_scaled_metrics
            
        except Exception as e:
            logger.warning(f"Temperature scaling failed: {str(e)}")
        
        # Platt scaling
        try:
            platt_probs = self.perform_platt_scaling(
                validation_scores, validation_labels, test_scores
            )
            methods_results['platt_scaling'] = self.calculate_calibration_metrics(
                platt_probs.tolist(), test_labels.tolist()
            )
        except Exception as e:
            logger.warning(f"Platt scaling failed: {str(e)}")
        
        # Isotonic regression
        try:
            isotonic_probs = self.perform_isotonic_regression(
                validation_scores, validation_labels, test_scores
            )
            methods_results['isotonic_regression'] = self.calculate_calibration_metrics(
                isotonic_probs.tolist(), test_labels.tolist()
            )
        except Exception as e:
            logger.warning(f"Isotonic regression failed: {str(e)}")
        
        return methods_results
    
    def calculate_prediction_intervals(self, 
                                     predictions: List[float],
                                     prediction_errors: List[float],
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate prediction intervals for continuous outputs
        
        Args:
            predictions: Point predictions
            prediction_errors: Prediction errors
            confidence_level: Confidence level for intervals
            
        Returns:
            Prediction interval statistics
        """
        logger.info(f"Calculating {confidence_level*100}% prediction intervals")
        
        # Estimate prediction error distribution
        error_std = np.std(prediction_errors)
        error_mean = np.mean(prediction_errors)
        
        # Calculate intervals (assuming normal distribution)
        from scipy import stats
        alpha = 1 - confidence_level
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        margin_of_error = critical_value * error_std
        
        # Calculate coverage (percentage of true values within intervals)
        lower_bounds = np.array(predictions) - margin_of_error
        upper_bounds = np.array(predictions) + margin_of_error
        
        # For evaluation, would need true values
        # coverage = np.mean((true_values >= lower_bounds) & (true_values <= upper_bounds))
        
        return {
            'confidence_level': confidence_level,
            'error_mean': error_mean,
            'error_std': error_std,
            'margin_of_error': margin_of_error,
            'interval_width': 2 * margin_of_error,
            'prediction_intervals': list(zip(lower_bounds, upper_bounds))
        }
