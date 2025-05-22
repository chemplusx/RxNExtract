# Multi-stage Dockerfile for Chemistry LLM Inference

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app
USER app

# Add local bin to PATH
ENV PATH="/home/app/.local/bin:$PATH"

# Development stage
FROM base as development

# Install development dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Install the package in development mode
COPY --chown=app:app . .
RUN pip install --user -e ".[dev]"

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --user torch transformers peft bitsandbytes accelerate \
    numpy pandas pyyaml tqdm click

# Copy and install the package
COPY --chown=app:app . .
RUN pip install --user .

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["chemistry-llm", "--help"]