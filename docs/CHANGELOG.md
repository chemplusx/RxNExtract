# Changelog

All notable changes to the RxNExtract project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI package distribution for easy installation
- Docker containerization with GPU support
- HuggingFace model hub integration
- Conda-Forge recipe for cross-platform installation
- Interactive Jupyter notebook examples
- Web API integration examples
- Enhanced documentation with comprehensive guides
- Community support channels and resources

### Changed
- Restructured documentation into modular guides
- Improved installation process with multiple options
- Enhanced CLI interface with better error handling
- Updated model loading to support HuggingFace Hub

### Fixed
- Memory optimization issues in batch processing
- Cross-platform compatibility improvements
- Enhanced error messages and logging

## [1.2.0] - 2025-08-21

### Added - Comprehensive Analysis Suite
- **Error Analysis Framework**: Systematic error categorization and analysis
  - Entity recognition error analysis (missing entities, false positives, incorrect types)
  - Role classification error analysis (reactant/product confusion, catalyst misidentification)
  - Condition extraction error analysis (missing temperature/time, incomplete procedures)
  - Chain-of-Thought failure analysis (implicit reasoning, generic entity handling)
- **Ablation Study Framework**: Component-level performance analysis
  - 8 ablation configurations from direct extraction to complete framework
  - Dynamic prompt component analysis
  - Complexity-stratified evaluation (simple, moderate, complex reactions)
  - Component contribution and interaction effects analysis
- **Statistical Analysis Suite**: Research-grade significance testing
  - Pairwise method comparison (paired t-tests, Wilcoxon signed-rank)
  - McNemar's test for classification performance comparison
  - One-way ANOVA with post-hoc tests for multiple method comparison
  - Bootstrap confidence intervals and effect size calculations
  - Baseline reproducibility analysis for literature validation
- **Uncertainty Quantification Module**: Confidence calibration and analysis
  - Expected Calibration Error (ECE) and Brier Score calculation
  - Temperature scaling, Platt scaling, and isotonic regression
  - Confidence-stratified performance analysis
  - Reliability diagram generation and visualization
- **Comprehensive Metrics Calculator**: Complete performance assessment
  - Complete Reaction Accuracy (CRA), Entity F1, Role Classification Accuracy
  - Condition extraction metrics and complexity-based analysis
  - Error reduction calculations and custom metrics support
- **Command-line Analysis Scripts**: Easy-to-use analysis tools
  - `run_error_analysis.py`: Comprehensive error analysis
  - `run_ablation_study.py`: Complete ablation studies
  - `run_statistical_analysis.py`: Statistical significance testing
  - `run_uncertainty_analysis.py`: Uncertainty quantification
  - `run_complete_analysis.py`: Full analysis pipeline

### Added - Research Reproducibility Features
- **Complete Analysis Pipeline**: End-to-end analysis workflow
  - Automated generation of all paper figures and tables
  - Research reproducibility with configurable parameters
  - Export functionality for publication-ready results
- **Performance Benchmarking**: Systematic evaluation framework
  - Literature baseline comparison and validation
  - Cross-domain performance analysis
  - Complexity-stratified evaluation metrics

### Added - Enhanced Documentation
- **Analysis Guide**: Comprehensive analysis framework documentation
- **Advanced Examples**: Research applications and custom analysis
- **Configuration Guide**: Detailed configuration options
- **API Reference**: Complete analysis module documentation

### Performance Improvements
- **Error Reduction**: 47.8-55.2% across major error categories
  - Entity Recognition: 52.4% reduction in missing entities, 54.8% in false positives
  - Role Classification: 55.2% reduction in reactant/product confusion
  - Condition Extraction: 49.1% reduction in missing temperature conditions
- **Statistical Significance**: McNemar's χ² = 134.67 (p < 0.001), Cohen's d = 0.82
- **Calibration Improvement**: 57.1% ECE reduction with temperature scaling
- **Overall Performance**: +122.6% improvement in Complete Reaction Accuracy

### Technical Enhancements
- Enhanced memory efficiency with 4-bit quantization support
- Improved batch processing with progress tracking
- Advanced prompt engineering with dynamic selection
- Robust XML parsing with error recovery
- Professional logging with configurable levels

## [1.1.0] - 2025-06-15

### Added
- **Self-Grounding Mechanism**: Entity validation and consistency checking
- **Iterative Refinement**: Multi-pass extraction with quality improvement
- **Dynamic Prompt Selection**: Context-aware prompt optimization
- **Confidence Scoring**: Prediction confidence estimation
- **Batch Processing Optimization**: Efficient large-scale processing

### Improved
- **Chain-of-Thought Reasoning**: Enhanced step-by-step extraction
- **XML Structure Validation**: Robust output parsing
- **Error Handling**: Graceful failure recovery
- **Memory Management**: Reduced memory footprint

### Performance Gains
- **Complete Reaction Accuracy**: Improved from 23.4% to 52.1%
- **Entity F1 Score**: Improved from 0.674 to 0.856
- **Role Classification**: Improved from 68.2% to 85.9%
- **Condition Extraction**: Improved from 0.421 to 0.689 F1

### Bug Fixes
- Fixed memory leaks in long-running processes
- Resolved parsing errors for complex chemical names
- Fixed device selection issues on multi-GPU systems

## [1.0.0] - 2025-05-21 - Initial Release

### Added - Core Functionality
- **Chemical Reaction Extraction**: Core extraction engine using fine-tuned LLMs
- **Multiple Interfaces**: CLI, interactive mode, batch processing, programmatic API
- **Chain-of-Thought Prompting**: Step-by-step reasoning for better accuracy
- **XML Output Parsing**: Structured data output for easy integration
- **CLI Interface**: Command-line interface for streamlined usage
- **Comprehensive Test Suite**: Unit and integration tests for reliability