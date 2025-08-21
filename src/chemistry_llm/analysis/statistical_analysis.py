"""
Statistical analysis module for chemistry LLM evaluation
Provides comprehensive statistical testing and significance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for reaction extraction evaluation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.significance_level = self.config.get('significance_level', 0.05)
        
    def perform_pairwise_comparison(self, 
                                  method1_results: List[float],
                                  method2_results: List[float],
                                  method1_name: str = "Method 1",
                                  method2_name: str = "Method 2",
                                  test_type: str = "paired_t") -> Dict[str, Any]:
        """
        Perform pairwise statistical comparison between two methods
        
        Args:
            method1_results: Results from first method
            method2_results: Results from second method
            method1_name: Name of first method
            method2_name: Name of second method
            test_type: Type of statistical test
            
        Returns:
            Statistical comparison results
        """
        logger.info(f"Performing pairwise comparison: {method1_name} vs {method2_name}")
        
        if len(method1_results) != len(method2_results):
            raise ValueError("Result lists must have same length")
        
        # Choose appropriate test
        if test_type == "paired_t":
            statistic, p_value = stats.ttest_rel(method1_results, method2_results)
            test_name = "Paired t-test"
            
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(method1_results, method2_results)
            test_name = "Wilcoxon signed-rank test"
            
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(method1_results, method2_results, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(method1_results, method2_results)
        
        # Calculate confidence intervals for the difference
        diff_ci = self._calculate_difference_ci(method1_results, method2_results)
        
        return {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': effect_size,
            'significant': p_value < self.significance_level,
            'difference_ci': diff_ci,
            'method1_mean': np.mean(method1_results),
            'method2_mean': np.mean(method2_results),
            'method1_std': np.std(method1_results),
            'method2_std': np.std(method2_results)
        }
    
    def perform_mcnemar_test(self, 
                           method1_correct: List[bool],
                           method2_correct: List[bool],
                           method1_name: str = "Method 1",
                           method2_name: str = "Method 2") -> Dict[str, Any]:
        """
        Perform McNemar's test for comparing classification methods
        
        Args:
            method1_correct: Binary correctness for method 1
            method2_correct: Binary correctness for method 2
            method1_name: Name of first method
            method2_name: Name of second method
            
        Returns:
            McNemar test results
        """
        logger.info(f"Performing McNemar's test: {method1_name} vs {method2_name}")
        
        if len(method1_correct) != len(method2_correct):
            raise ValueError("Correctness lists must have same length")
        
        # Create contingency table
        both_correct = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if m1 and m2)
        method1_only = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if m1 and not m2)
        method2_only = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if not m1 and m2)
        both_wrong = sum(1 for m1, m2 in zip(method1_correct, method2_correct) if not m1 and not m2)
        
        # Contingency table
        contingency_table = np.array([[both_correct, method1_only],
                                     [method2_only, both_wrong]])
        
        # Perform McNemar's test
        try:
            result = mcnemar(contingency_table, exact=True)
            statistic = result.statistic
            p_value = result.pvalue
        except:
            # Fallback to standard chi-square calculation
            if (method1_only + method2_only) > 0:
                statistic = (method1_only - method2_only)**2 / (method1_only + method2_only)
                p_value = 1 - stats.chi2.cdf(statistic, 1)
            else:
                statistic = 0.0
                p_value = 1.0
        
        return {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'test_name': "McNemar's test",
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'contingency_table': contingency_table.tolist(),
            'both_correct': both_correct,
            'method1_only_correct': method1_only,
            'method2_only_correct': method2_only,
            'both_wrong': both_wrong,
            'method1_accuracy': np.mean(method1_correct),
            'method2_accuracy': np.mean(method2_correct)
        }
    
    def _calculate_cohens_d(self, 
                          group1: List[float], 
                          group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    def _calculate_difference_ci(self, 
                               group1: List[float], 
                               group2: List[float],
                               confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for the difference between means"""
        
        differences = np.array(group1) - np.array(group2)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        # t-distribution critical value
        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = critical_value * (std_diff / np.sqrt(n))
        
        return {
            'mean_difference': mean_diff,
            'lower_bound': mean_diff - margin_error,
            'upper_bound': mean_diff + margin_error,
            'confidence_level': confidence_level
        }
    
    def perform_anova(self, 
                     groups: Dict[str, List[float]],
                     post_hoc: bool = True) -> Dict[str, Any]:
        """
        Perform one-way ANOVA with optional post-hoc tests
        
        Args:
            groups: Dictionary mapping group names to result lists
            post_hoc: Whether to perform post-hoc pairwise comparisons
            
        Returns:
            ANOVA results with optional post-hoc tests
        """
        logger.info(f"Performing ANOVA with {len(groups)} groups")
        
        group_names = list(groups.keys())
        group_data = [groups[name] for name in group_names]
        
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*group_data)
        
        # Calculate effect size (eta-squared)
        eta_squared = self._calculate_eta_squared(group_data)
        
        results = {
            'test_name': 'One-way ANOVA',
            'group_names': group_names,
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'eta_squared': eta_squared,
            'group_statistics': {}
        }
        
        # Calculate group statistics
        for name, data in groups.items():
            results['group_statistics'][name] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data)
            }
        
        # Post-hoc pairwise comparisons if requested and ANOVA is significant
        if post_hoc and p_value < self.significance_level:
            post_hoc_results = self._perform_post_hoc_tests(groups)
            results['post_hoc'] = post_hoc_results
        
        return results
    
    def _calculate_eta_squared(self, groups: List[List[float]]) -> float:
        """Calculate eta-squared effect size for ANOVA"""
        
        # Calculate overall mean
        all_data = [item for group in groups for item in group]
        overall_mean = np.mean(all_data)
        
        # Calculate sum of squares between groups (SSB)
        ssb = 0
        for group in groups:
            group_mean = np.mean(group)
            n_group = len(group)
            ssb += n_group * (group_mean - overall_mean)**2
        
        # Calculate total sum of squares (SST)
        sst = sum((x - overall_mean)**2 for x in all_data)
        
        # Eta-squared
        if sst == 0:
            return 0.0
        
        eta_squared = ssb / sst
        return eta_squared
    
    def _perform_post_hoc_tests(self, groups: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform post-hoc pairwise comparisons using Bonferroni correction"""
        
        group_names = list(groups.keys())
        post_hoc_results = {}
        
        # Calculate number of comparisons for Bonferroni correction
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonferroni_alpha = self.significance_level / n_comparisons
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                
                # Perform t-test
                statistic, p_value = stats.ttest_ind(groups[name1], groups[name2])
                
                # Calculate effect size
                effect_size = self._calculate_cohens_d(groups[name1], groups[name2])
                
                comparison_key = f"{name1}_vs_{name2}"
                post_hoc_results[comparison_key] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'bonferroni_corrected_p': float(p_value * n_comparisons),
                    'significant_bonferroni': (p_value * n_comparisons) < self.significance_level,
                    'effect_size': effect_size
                }
        
        return post_hoc_results
    
    def calculate_baseline_reproducibility(self, 
                                         literature_results: Dict[str, float],
                                         reproduced_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze baseline reproducibility compared to literature
        
        Args:
            literature_results: Literature-reported results
            reproduced_results: Our reproduced results (multiple runs)
            
        Returns:
            Reproducibility analysis
        """
        logger.info("Analyzing baseline reproducibility")
        
        reproducibility_analysis = {}
        
        for method_name in literature_results:
            if method_name in reproduced_results:
                lit_result = literature_results[method_name]
                repro_results = reproduced_results[method_name]
                
                repro_mean = np.mean(repro_results)
                repro_std = np.std(repro_results, ddof=1)
                
                # Calculate difference from literature
                difference = repro_mean - lit_result
                relative_difference = (difference / lit_result) * 100 if lit_result != 0 else 0
                
                # One-sample t-test against literature value
                t_stat, p_value = stats.ttest_1samp(repro_results, lit_result)
                
                # Calculate confidence interval
                n = len(repro_results)
                critical_value = stats.t.ppf(0.975, n - 1)  # 95% CI
                margin_error = critical_value * (repro_std / np.sqrt(n))
                
                reproducibility_analysis[method_name] = {
                    'literature_result': lit_result,
                    'reproduced_mean': repro_mean,
                    'reproduced_std': repro_std,
                    'n_runs': n,
                    'difference': difference,
                    'relative_difference_percent': relative_difference,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significantly_different': p_value < self.significance_level,
                    'confidence_interval': {
                        'lower': repro_mean - margin_error,
                        'upper': repro_mean + margin_error
                    }
                }
        
        return reproducibility_analysis
    
    def perform_cross_validation_analysis(self, 
                                        cv_results: List[List[float]],
                                        method_names: List[str]) -> Dict[str, Any]:
        """
        Analyze cross-validation results for statistical significance
        
        Args:
            cv_results: List of CV results for each method
            method_names: Names of the methods
            
        Returns:
            Cross-validation statistical analysis
        """
        logger.info("Performing cross-validation statistical analysis")
        
        if len(cv_results) != len(method_names):
            raise ValueError("Number of CV results must match number of method names")
        
        cv_analysis = {}
        
        # Calculate statistics for each method
        for i, (results, name) in enumerate(zip(cv_results, method_names)):
            cv_analysis[name] = {
                'mean': np.mean(results),
                'std': np.std(results, ddof=1),
                'min': np.min(results),
                'max': np.max(results),
                'cv_coefficient': np.std(results, ddof=1) / np.mean(results) if np.mean(results) != 0 else 0
            }
        
        # Perform ANOVA if more than 2 methods
        if len(method_names) > 2:
            groups = {name: results for name, results in zip(method_names, cv_results)}
            anova_results = self.perform_anova(groups, post_hoc=True)
            cv_analysis['anova'] = anova_results
        
        # Pairwise comparisons for all method pairs
        pairwise_comparisons = {}
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                name1, name2 = method_names[i], method_names[j]
                comparison = self.perform_pairwise_comparison(
                    cv_results[i], cv_results[j], name1, name2, test_type="paired_t"
                )
                pairwise_comparisons[f"{name1}_vs_{name2}"] = comparison
        
        cv_analysis['pairwise_comparisons'] = pairwise_comparisons
        
        return cv_analysis
    
    def test_normality(self, data: List[float], 
                      method: str = "shapiro") -> Dict[str, Any]:
        """
        Test data normality using various methods
        
        Args:
            data: Data to test for normality
            method: Test method ('shapiro', 'kstest', 'anderson')
            
        Returns:
            Normality test results
        """
        logger.info(f"Testing normality using {method}")
        
        if method == "shapiro":
            statistic, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk test"
            
        elif method == "kstest":
            statistic, p_value = stats.kstest(data, 'norm')
            test_name = "Kolmogorov-Smirnov test"
            
        elif method == "anderson":
            result = stats.anderson(data, dist='norm')
            # Use 5% significance level
            critical_value = result.critical_values[2]  # 5% level
            statistic = result.statistic
            p_value = 0.05 if statistic > critical_value else 0.1  # Approximate
            test_name = "Anderson-Darling test"
            
        else:
            raise ValueError(f"Unsupported normality test: {method}")
        
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_normal': p_value > self.significance_level,
            'data_mean': np.mean(data),
            'data_std': np.std(data, ddof=1),
            'data_skewness': stats.skew(data),
            'data_kurtosis': stats.kurtosis(data)
        }
    
    def generate_statistical_report(self, 
                                  analysis_results: Dict[str, Any],
                                  output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive statistical analysis report
        
        Args:
            analysis_results: Results from statistical analyses
            output_file: Optional output file path
            
        Returns:
            Formatted statistical report
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "CHEMISTRY REACTION EXTRACTION - STATISTICAL ANALYSIS REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Significance Level: {self.significance_level}",
            ""
        ])
        
        # Pairwise Comparisons
        if 'pairwise_comparisons' in analysis_results:
            report_lines.extend([
                "PAIRWISE METHOD COMPARISONS",
                "-" * 50
            ])
            
            for comparison_name, result in analysis_results['pairwise_comparisons'].items():
                report_lines.extend([
                    f"\n{comparison_name}:",
                    f"  Test: {result['test_name']}",
                    f"  Statistic: {result['statistic']:.4f}",
                    f"  p-value: {result['p_value']:.6f}",
                    f"  Effect Size (Cohen's d): {result['effect_size']:.3f}",
                    f"  Significant: {'Yes' if result['significant'] else 'No'}",
                    f"  Mean Difference: {result['difference_ci']['mean_difference']:.4f}",
                    f"  95% CI: [{result['difference_ci']['lower_bound']:.4f}, {result['difference_ci']['upper_bound']:.4f}]"
                ])
            
            report_lines.append("")
        
        # ANOVA Results
        if 'anova' in analysis_results:
            anova = analysis_results['anova']
            report_lines.extend([
                "ANALYSIS OF VARIANCE (ANOVA)",
                "-" * 40,
                f"F-statistic: {anova['f_statistic']:.4f}",
                f"p-value: {anova['p_value']:.6f}",
                f"Effect Size (η²): {anova['eta_squared']:.4f}",
                f"Significant: {'Yes' if anova['significant'] else 'No'}",
                ""
            ])
            
            # Group statistics
            report_lines.append("Group Statistics:")
            for group_name, stats in anova['group_statistics'].items():
                report_lines.append(f"  {group_name}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, n={stats['n']}")
            
            report_lines.append("")
        
        # McNemar Test Results
        if 'mcnemar_tests' in analysis_results:
            report_lines.extend([
                "MCNEMAR'S TESTS",
                "-" * 20
            ])
            
            for test_name, result in analysis_results['mcnemar_tests'].items():
                report_lines.extend([
                    f"\n{test_name}:",
                    f"  χ² statistic: {result['statistic']:.4f}",
                    f"  p-value: {result['p_value']:.6f}",
                    f"  Significant: {'Yes' if result['significant'] else 'No'}",
                    f"  {result['method1_name']} only correct: {result['method1_only_correct']}",
                    f"  {result['method2_name']} only correct: {result['method2_only_correct']}"
                ])
            
            report_lines.append("")
        
        # Reproducibility Analysis
        if 'reproducibility' in analysis_results:
            report_lines.extend([
                "BASELINE REPRODUCIBILITY ANALYSIS",
                "-" * 40
            ])
            
            for method_name, repro_data in analysis_results['reproducibility'].items():
                report_lines.extend([
                    f"\n{method_name}:",
                    f"  Literature: {repro_data['literature_result']:.4f}",
                    f"  Reproduced: {repro_data['reproduced_mean']:.4f} ± {repro_data['reproduced_std']:.4f}",
                    f"  Difference: {repro_data['difference']:.4f} ({repro_data['relative_difference_percent']:+.1f}%)",
                    f"  Significantly Different: {'Yes' if repro_data['significantly_different'] else 'No'}",
                    f"  95% CI: [{repro_data['confidence_interval']['lower']:.4f}, {repro_data['confidence_interval']['upper']:.4f}]"
                ])
        
        # Generate report string
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Statistical report saved to {output_file}")
        
        return report_text
    
    def export_results_to_dataframe(self, 
                                   analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Export statistical analysis results to pandas DataFrame
        
        Args:
            analysis_results: Results from statistical analyses
            
        Returns:
            DataFrame with analysis results
        """
        export_data = []
        
        # Process pairwise comparisons
        if 'pairwise_comparisons' in analysis_results:
            for comparison_name, result in analysis_results['pairwise_comparisons'].items():
                export_data.append({
                    'Analysis_Type': 'Pairwise_Comparison',
                    'Comparison': comparison_name,
                    'Test_Name': result['test_name'],
                    'Statistic': result['statistic'],
                    'P_Value': result['p_value'],
                    'Effect_Size': result['effect_size'],
                    'Significant': result['significant'],
                    'Method1_Mean': result['method1_mean'],
                    'Method2_Mean': result['method2_mean']
                })
        
        # Process McNemar tests
        if 'mcnemar_tests' in analysis_results:
            for test_name, result in analysis_results['mcnemar_tests'].items():
                export_data.append({
                    'Analysis_Type': 'McNemar_Test',
                    'Comparison': test_name,
                    'Test_Name': result['test_name'],
                    'Statistic': result['statistic'],
                    'P_Value': result['p_value'],
                    'Significant': result['significant'],
                    'Method1_Accuracy': result['method1_accuracy'],
                    'Method2_Accuracy': result['method2_accuracy']
                })
        
        return pd.DataFrame(export_data)
