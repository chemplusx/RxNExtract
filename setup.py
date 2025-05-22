from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.35.0", 
    "peft>=0.7.0",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.24.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.64.0",
    "click>=8.0.0",
]

# Development requirements
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# Optional requirements
optional_requires = [
    "flash-attn>=2.0.0",
    "wandb>=0.15.0",
]

setup(
    name="chemistry-llm-inference",
    version="1.0.0",
    author="Your Organization",
    author_email="contact@your-org.com",
    description="Professional chemistry reaction extraction using fine-tuned LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/chemistry-llm-inference",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "optional": optional_requires,
        "all": dev_requires + optional_requires,
    },
    entry_points={
        "console_scripts": [
            "chemistry-llm=chemistry_llm.cli.interface:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chemistry_llm": ["config/*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/chemistry-llm-inference/issues",
        "Source": "https://github.com/your-org/chemistry-llm-inference",
        "Documentation": "https://github.com/your-org/chemistry-llm-inference/wiki",
    },
)
