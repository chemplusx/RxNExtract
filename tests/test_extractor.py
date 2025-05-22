"""
Tests for the ChemistryReactionExtractor class
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from chemistry_llm.core.extractor import ChemistryReactionExtractor
from chemistry_llm.utils.xml_parser import parse_reaction_xml


class TestChemistryReactionExtractor:
    """Test cases for ChemistryReactionExtractor"""
    
    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model path for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            model_path.mkdir()
            
            # Create mock adapter config
            adapter_config = {
                "base_model_name_or_path": "test-model",
                "task_type": "CAUSAL_LM"
            }
            with open(model_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)
            
            yield str(model_path)
    
    @pytest.fixture
    def sample_procedures(self):
        """Sample chemical procedures for testing"""
        return [
            "Add 5g of sodium chloride to 100mL of water and stir for 30 minutes.",
            "Heat the mixture of benzene and aluminum chloride to 80°C for 2 hours.",
            "Cool the reaction mixture and filter to obtain the crystalline product."
        ]
    
    def test_initialization(self, mock_model_path):
        """Test extractor initialization"""
        extractor = ChemistryReactionExtractor(model_path=mock_model_path)
        
        assert extractor.model_path == mock_model_path
        assert extractor.device in ["cuda", "cpu", "mps"]
        assert not extractor._model_loaded
    
    def test_generation_config_setup(self, mock_model_path):
        """Test generation configuration setup"""
        config = {
            "model": {
                "max_new_tokens": 256,
                "default_temperature": 0.2,
                "default_top_p": 0.8
            }
        }
        
        extractor = ChemistryReactionExtractor(
            model_path=mock_model_path, 
            config=config
        )
        
        assert extractor.generation_config["max_new_tokens"] == 256
        assert extractor.generation_config["temperature"] == 0.2
        assert extractor.generation_config["top_p"] == 0.8
    
    @patch('chemistry_llm.core.extractor.ModelLoader')
    @patch('chemistry_llm.core.extractor.PromptBuilder')
    def test_model_loading(self, mock_prompt_builder, mock_model_loader, mock_model_path):
        """Test model loading functionality"""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        mock_loader_instance = Mock()
        mock_loader_instance.load_complete_model.return_value = (mock_model, mock_tokenizer)
        mock_model_loader.return_value = mock_loader_instance
        
        # Test loading
        extractor = ChemistryReactionExtractor(model_path=mock_model_path)
        extractor._load_model_if_needed()
        
        assert extractor._model_loaded
        assert extractor.model == mock_model
        assert extractor.tokenizer == mock_tokenizer
    
    def test_batch_analyze_structure(self, mock_model_path, sample_procedures):
        """Test batch analysis structure (without actual model)"""
        extractor = ChemistryReactionExtractor(model_path=mock_model_path)
        
        # Mock the analyze_procedure method
        def mock_analyze(procedure, **kwargs):
            return {
                "procedure": procedure,
                "extracted_data": {"reactants": [{"name": "test"}]},
                "success": True
            }
        
        extractor.analyze_procedure = mock_analyze
        
        results = extractor.batch_analyze(sample_procedures)
        
        assert len(results) == len(sample_procedures)
        for result in results:
            assert "procedure" in result
            assert "extracted_data" in result
            assert "success" in result