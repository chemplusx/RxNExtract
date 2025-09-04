# Contributing to Chemistry LLM Inference

We welcome contributions to the Chemistry LLM Inference project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Access to a GPU (recommended) or sufficient CPU resources

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/chemistry-llm-inference.git
cd chemistry-llm-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install additional development tools
pip install pre-commit
pre-commit install
```

### Directory Structure

```
chemistry-llm-inference/
├── src/chemistry_llm/          # Main package code
├── tests/                      # Test files
├── examples/                   # Example usage scripts
├── scripts/                    # Utility scripts
├── config/                     # Configuration files
└── docs/                       # Documentation (if added)
```

## Making Changes

### Types of Contributions

We welcome the following types of contributions:

1. **Bug fixes** - Fix issues in existing code
2. **Feature additions** - Add new functionality
3. **Documentation improvements** - Enhance or clarify documentation
4. **Performance optimizations** - Improve speed or memory usage
5. **Test improvements** - Add or improve test coverage

### Branch Naming

Use clear, descriptive branch names:

- `feature/add-batch-processing`
- `bugfix/xml-parsing-error`
- `docs/update-readme`
- `refactor/model-loading`

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

More detailed explanation if needed

- Use present tense ("Add feature" not "Added feature")
- Limit first line to 72 characters
- Reference issues and pull requests liberally
```

Examples:
- `feat(extractor): add support for custom prompt templates`
- `fix(parser): handle malformed XML gracefully`
- `docs(readme): update installation instructions`

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_extractor.py

# Run with coverage
python -m pytest tests/ --cov=src/chemistry_llm --cov-report=html
```

### Writing Tests

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies (models, APIs)

### Test Structure

```python
class TestNewFeature:
    """Test cases for new feature"""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for tests"""
        return {...}
    
    def test_feature_success_case(self, sample_data):
        """Test successful execution"""
        # Test implementation
        
    def test_feature_error_handling(self):
        """Test error handling"""
        # Test implementation
```

## Code Style

### Python Style Guidelines

We follow PEP 8 with some modifications:

- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use type hints where appropriate
- Use docstrings for all public functions and classes

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Check formatting
black --check src/ tests/ examples/
isort --check-only src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically:

- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

## Submitting Changes

### Pull Request Process

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run tests and formatting**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   python -m pytest tests/
   ```

3. **Create pull request**
   - Use a clear, descriptive title
   - Include a detailed description
   - Reference related issues
   - Add screenshots if applicable
   - Request reviews from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests for functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits if requested
5. Merge after approval

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def analyze_procedure(self, procedure_text: str, return_raw: bool = False) -> Dict[str, Any]:
    """
    Analyze a chemical procedure and extract reaction information.
    
    Args:
        procedure_text: The chemical procedure text to analyze
        return_raw: Whether to include raw model output in results
        
    Returns:
        Dictionary containing extracted reaction data and metadata
        
    Raises:
        ValueError: If procedure_text is empty or invalid
        ModelNotLoadedException: If model hasn't been loaded
        
    Example:
        >>> extractor = ChemistryReactionExtractor("./model")
        >>> result = extractor.analyze_procedure("Add 5g NaCl to water...")
        >>> print(result["extracted_data"])
    """
```