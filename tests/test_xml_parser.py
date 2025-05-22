"""
Tests for XML parsing utilities
"""

import pytest
from chemistry_llm.utils.xml_parser import (
    parse_reaction_xml, 
    validate_xml_structure,
    extract_xml_from_text,
    format_reaction_summary
)


class TestXMLParser:
    """Test cases for XML parsing functionality"""
    
    @pytest.fixture
    def valid_xml(self):
        """Valid XML reaction data"""
        return """
        <reaction>
          <reactant name="benzoic acid" amount="5.0 g" />
          <reagent name="hydrochloric acid" amount="10 mL" />
          <solvent name="water" amount="100 mL" />
          <workup type="cooling" duration="30 min" note="ice bath" />
          <product name="benzoic acid crystals" yield="84%" />
        </reaction>
        """
    
    @pytest.fixture
    def text_with_xml(self):
        """Text containing XML"""
        return """
        Based on the procedure, I can extract the following information:
        
        <reaction>
          <reactant name="sodium chloride" amount="5 g" />
          <solvent name="water" amount="100 mL" />
          <product name="salt solution" />
        </reaction>
        
        This represents a simple dissolution process.
        """
    
    @pytest.fixture
    def invalid_xml(self):
        """Invalid XML structure"""
        return """
        <reaction>
          <reactant name="test" amount="5g"
          <product name="result" />
        </reaction>
        """
    
    def test_validate_xml_structure_valid(self, valid_xml):
        """Test XML validation with valid XML"""
        assert validate_xml_structure(valid_xml) == True
    
    def test_validate_xml_structure_invalid(self, invalid_xml):
        """Test XML validation with invalid XML"""
        assert validate_xml_structure(invalid_xml) == False
    
    def test_extract_xml_from_text(self, text_with_xml):
        """Test XML extraction from text"""
        extracted = extract_xml_from_text(text_with_xml)
        assert extracted is not None
        assert "<reaction>" in extracted
        assert "</reaction>" in extracted
        assert "sodium chloride" in extracted
    
    def test_parse_valid_xml(self, valid_xml):
        """Test parsing valid XML"""
        result = parse_reaction_xml(valid_xml)
        
        assert "error" not in result
        assert "reactants" in result
        assert "reagents" in result
        assert "solvents" in result
        assert "workups" in result
        assert "products" in result
        
        # Check specific content
        assert len(result["reactants"]) == 1
        assert result["reactants"][0]["name"] == "benzoic acid"
        assert result["reactants"][0]["amount"] == "5.0 g"
        
        assert len(result["products"]) == 1
        assert result["products"][0]["yield"] == "84%"
    
    def test_parse_invalid_xml_strict(self, invalid_xml):
        """Test parsing invalid XML in strict mode"""
        with pytest.raises(ValueError):
            parse_reaction_xml(invalid_xml, strict=True)
    
    def test_parse_invalid_xml_lenient(self, invalid_xml):
        """Test parsing invalid XML in lenient mode"""
        result = parse_reaction_xml(invalid_xml, strict=False)
        assert "error" in result
    
    def test_format_reaction_summary(self, valid_xml):
        """Test reaction summary formatting"""
        parsed_data = parse_reaction_xml(valid_xml)
        summary = format_reaction_summary(parsed_data)
        
        assert "Reactants:" in summary
        assert "benzoic acid" in summary
        assert "Products:" in summary
        assert "84%" in summary
    
    def test_empty_xml_extraction(self):
        """Test extraction from text without XML"""
        text = "This is just plain text without any XML content."
        extracted = extract_xml_from_text(text)
        assert extracted is None