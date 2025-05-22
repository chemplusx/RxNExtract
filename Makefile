.PHONY: help install install-dev test test-cov lint format clean build upload docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"
	@echo "  upload        Upload to PyPI (requires credentials)"
	@echo "  docs          Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	python -m pytest tests/

test-cov:
	python -m pytest tests/ --cov=src/chemistry_llm --cov-report=html --cov-report=term

test-integration:
	python -m pytest tests/ -m integration

test-unit:
	python -m pytest tests/ -m unit

# Code quality
lint:
	flake8 src/ tests/ examples/
	mypy src/

format:
	black src/ tests/ examples/ scripts/
	isort src/ tests/ examples/ scripts/

format-check:
	black --check src/ tests/ examples/ scripts/
	isort --check-only src/ tests/ examples/ scripts/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

# Upload to PyPI
upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Consider adding Sphinx or mkdocs for documentation"

# Development helpers
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

check-all: format-check lint test
	@echo "All checks passed!"

# CI/CD helpers
ci-test:
	python -m pytest tests/ --cov=src/chemistry_llm --cov-report=xml --cov-report=term

ci-lint:
	flake8 src/ tests/ --format=github
	mypy src/ --junit-xml=mypy-results.xml