"""
Model loading utilities for chemistry LLM inference
"""

import os
import json
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

from ..utils.logger import get_logger
from ..utils.device_utils import get_optimal_device

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles loading of fine-tuned models with LoRA adapters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model loader
        
        Args:
            config: Configuration dictionary for model loading
        """
        self.config = config or {}
        self.model_config = self.config.get("model", {})
        
    def load_tokenizer(self, model_path: str) -> AutoTokenizer:
        """
        Load tokenizer from model path
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer from {model_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=self.model_config.get("trust_remote_code", True)
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Added padding token (using EOS token)")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def detect_base_model(self, model_path: str) -> str:
        """
        Auto-detect the base model name from adapter configuration
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Base model name
        """
        config_path = Path(model_path) / "adapter_config.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
                    if base_model:
                        logger.info(f"Detected base model: {base_model}")
                        return base_model
            except Exception as e:
                logger.warning(f"Could not read adapter config: {str(e)}")
        
        # Fallback to default
        default_model = "meta-llama/Llama-2-7b-hf"
        logger.warning(f"Using default base model: {default_model}")
        return default_model
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration for 4-bit loading
        
        Returns:
            BitsAndBytesConfig object
        """
        quant_config = self.model_config.get("quantization", {})
        
        compute_dtype = torch.float16
        if quant_config.get("bnb_4bit_compute_dtype") == "bfloat16":
            compute_dtype = torch.bfloat16
        
        return BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True)
        )
    
    def load_base_model(self, base_model_name: str, device: str = "auto") -> AutoModelForCausalLM:
        """
        Load the base model with quantization
        
        Args:
            base_model_name: Name of the base model
            device: Device to load the model on
            
        Returns:
            Loaded base model
        """
        logger.info(f"Loading base model: {base_model_name}")
        
        # Create quantization config
        bnb_config = self.create_quantization_config()
        
        # Determine compute dtype
        torch_dtype = torch.float16
        if self.model_config.get("torch_dtype") == "bfloat16":
            torch_dtype = torch.bfloat16
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map=self.model_config.get("device_map", "auto"),
                torch_dtype=torch_dtype,
                trust_remote_code=self.model_config.get("trust_remote_code", True),
                use_cache=True
            )
            
            logger.info("Base model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise
    
    def load_peft_model(self, base_model: AutoModelForCausalLM, 
                       adapter_path: str) -> PeftModel:
        """
        Load LoRA adapters onto the base model
        
        Args:
            base_model: The base model to load adapters onto
            adapter_path: Path to the LoRA adapter files
            
        Returns:
            Model with LoRA adapters loaded
        """
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        
        try:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16
            )
            
            # Set to evaluation mode
            model.eval()
            
            logger.info("LoRA adapters loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {str(e)}")
            raise
    
    def load_complete_model(self, model_path: str, base_model_name: Optional[str] = None,
                           device: str = "auto") -> tuple:
        """
        Load the complete model (base + adapters) and tokenizer
        
        Args:
            model_path: Path to the fine-tuned model directory
            base_model_name: Base model name (auto-detected if None)
            device: Device to load the model on
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer(model_path)
        
        # Detect base model if not provided
        if base_model_name is None:
            base_model_name = self.detect_base_model(model_path)
        
        # Load base model
        base_model = self.load_base_model(base_model_name, device)
        
        # Load LoRA adapters
        model = self.load_peft_model(base_model, model_path)
        
        return model, tokenizer
