"""
Logging utilities for chemistry LLM inference
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(level: str = "INFO", 
                 config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional logging configuration dictionary
    """
    log_config = config.get("logging", {}) if config else {}
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Setup basic configuration
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Setup file handler if configured
    file_config = log_config.get("file_handler", {})
    if file_config.get("enabled", False):
        filename = file_config.get("filename", "chemistry_llm.log")
        max_bytes = file_config.get("max_bytes", 10485760)  # 10MB
        backup_count = file_config.get("backup_count", 5)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)