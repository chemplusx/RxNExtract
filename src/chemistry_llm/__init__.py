"""
Chemistry LLM Inference Package

A professional system for extracting chemical reaction information 
from procedure texts using fine-tuned Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "ChemPlusX"
__email__ = "na"

from .core.extractor import ChemistryReactionExtractor
from .core.model_loader import ModelLoader
from .core.prompt_builder import PromptBuilder
from .utils.xml_parser import parse_reaction_xml, validate_xml_structure
from .utils.logger import setup_logging
from .utils.device_utils import get_optimal_device, get_memory_info

__all__ = [
    "ChemistryReactionExtractor",
    "ModelLoader", 
    "PromptBuilder",
    "parse_reaction_xml",
    "validate_xml_structure",
    "setup_logging",
    "get_optimal_device",
    "get_memory_info",
]