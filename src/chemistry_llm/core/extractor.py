"""
Main extraction engine for chemistry LLM inference
"""

import torch
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .model_loader import ModelLoader
from .prompt_builder import PromptBuilder
from ..utils.xml_parser import parse_reaction_xml
from ..utils.logger import get_logger
from ..utils.device_utils import get_optimal_device

logger = get_logger(__name__)


class ChemistryReactionExtractor:
    """
    Main class for extracting chemical reaction information using fine-tuned LLMs
    """
    
    def __init__(self, 
                 model_path: str,
                 base_model_name: Optional[str] = None,
                 device: str = "auto",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chemistry reaction extractor
        
        Args:
            model_path: Path to the fine-tuned model directory
            base_model_name: Base model name (auto-detected if None)
            device: Device to use for inference
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.config = config or {}
        
        # Setup device
        self.device = get_optimal_device() if device == "auto" else device
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model_loader = ModelLoader(self.config)
        self.prompt_builder = PromptBuilder(self.config)
        
        # Model and tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Generation parameters
        self.generation_config = self._setup_generation_config()
        
        logger.info(f"ChemistryReactionExtractor initialized with model: {model_path}")
    
    def _setup_generation_config(self) -> Dict[str, Any]:
        """Setup generation configuration from config"""
        model_config = self.config.get("model", {})
        
        return {
            "max_new_tokens": model_config.get("max_new_tokens", 512),
            "temperature": model_config.get("default_temperature", 0.1),
            "top_p": model_config.get("default_top_p", 0.95),
            "repetition_penalty": model_config.get("repetition_penalty", 1.1),
            "do_sample": model_config.get("default_temperature", 0.1) > 0,
            "pad_token_id": None,  # Set after tokenizer is loaded
            "eos_token_id": None,  # Set after tokenizer is loaded
        }
    
    def _load_model_if_needed(self):
        """Load model and tokenizer if not already loaded"""
        if not self._model_loaded:
            logger.info("Loading model and tokenizer...")
            start_time = datetime.now()
            
            self.model, self.tokenizer = self.model_loader.load_complete_model(
                model_path=self.model_path,
                base_model_name=self.base_model_name,
                device=self.device
            )
            
            # Update generation config with tokenizer info
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            self._model_loaded = True
    
    def extract_reaction(self, 
                        procedure_text: str,
                        use_cot: Optional[bool] = None,
                        **generation_kwargs) -> str:
        """
        Extract chemical reaction information from procedure text
        
        Args:
            procedure_text: The chemical procedure text
            use_cot: Whether to use Chain-of-Thought prompting
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text with reaction information
        """
        self._load_model_if_needed()
        
        # Build prompt
        if use_cot is not None:
            # Temporarily override CoT setting
            original_cot = self.prompt_builder.use_cot
            self.prompt_builder.use_cot = use_cot
            prompt = self.prompt_builder.build_cot_prompt(procedure_text)
            self.prompt_builder.use_cot = original_cot
        else:
            prompt = self.prompt_builder.build_cot_prompt(procedure_text)
        
        # Prepare generation config
        gen_config = self.generation_config.copy()
        gen_config.update(generation_kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.get("model", {}).get("max_length", 1024),
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        # Generate response
        logger.debug(f"Generating response for procedure: {procedure_text[:100]}...")
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_config
                )
                
                # Decode response (only new tokens)
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise
    
    def parse_extraction(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse the raw model output into structured data
        
        Args:
            raw_output: Raw model output text
            
        Returns:
            Parsed reaction data
        """
        try:
            return parse_reaction_xml(raw_output)
        except Exception as e:
            logger.error(f"Failed to parse extraction: {str(e)}")
            return {"error": f"Parsing failed: {str(e)}"}
    
    def analyze_procedure(self, 
                         procedure_text: str,
                         return_raw: bool = False,
                         include_timing: bool = True,
                         **generation_kwargs) -> Dict[str, Any]:
        """
        Complete analysis of a chemical procedure
        
        Args:
            procedure_text: The chemical procedure text
            return_raw: Whether to include raw model output
            include_timing: Whether to include timing information
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # Extract reaction information
            raw_output = self.extract_reaction(procedure_text, **generation_kwargs)
            
            # Parse the output
            extracted_data = self.parse_extraction(raw_output)
            
            # Build results
            results = {
                "procedure": procedure_text,
                "extracted_data": extracted_data,
                "timestamp": start_time.isoformat(),
                "success": "error" not in extracted_data
            }
            
            if return_raw:
                results["raw_output"] = raw_output
            
            if include_timing:
                processing_time = (datetime.now() - start_time).total_seconds()
                results["processing_time_seconds"] = processing_time
            
            logger.info(f"Analysis completed successfully for procedure: {procedure_text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "procedure": procedure_text,
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "success": False
            }
    
    def batch_analyze(self, 
                     procedures: list,
                     return_raw: bool = False,
                     progress_callback: Optional[callable] = None) -> list:
        """
        Analyze multiple procedures in batch
        
        Args:
            procedures: List of procedure texts
            return_raw: Whether to include raw outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results
        """
        results = []
        
        logger.info(f"Starting batch analysis of {len(procedures)} procedures")
        
        for i, procedure in enumerate(procedures):
            try:
                result = self.analyze_procedure(procedure, return_raw=return_raw)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(procedures), result)
                    
            except Exception as e:
                logger.error(f"Failed to analyze procedure {i+1}: {str(e)}")
                results.append({
                    "procedure": procedure,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                })
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        self._load_model_if_needed()
        
        return {
            "model_path": self.model_path,
            "base_model_name": self.base_model_name,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0,
            "generation_config": self.generation_config
        }