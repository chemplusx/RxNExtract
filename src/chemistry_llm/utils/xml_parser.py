"""
XML parsing utilities for chemical reaction data
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def validate_xml_structure(xml_text: str) -> bool:
    """
    Validate if the XML text has proper structure
    
    Args:
        xml_text: XML text to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check for reaction tags
        if not re.search(r'<reaction.*?>.*</reaction>', xml_text, re.DOTALL):
            return False
        
        # Try to parse
        xml_match = re.search(r'<reaction.*?>(.*?)</reaction>', xml_text, re.DOTALL)
        if xml_match:
            xml_content = "<reaction>" + xml_match.group(1) + "</reaction>"
            ET.fromstring(xml_content)
            return True
        
        return False
    except ET.ParseError:
        return False
    except Exception:
        return False


def extract_xml_from_text(text: str) -> Optional[str]:
    """
    Extract XML content from text that may contain other content
    
    Args:
        text: Text that may contain XML
        
    Returns:
        Extracted XML string or None if not found
    """
    # Look for reaction XML block
    xml_pattern = r'<reaction.*?>(.*?)</reaction>'
    match = re.search(xml_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return "<reaction>" + match.group(1) + "</reaction>"
    
    # Alternative patterns
    patterns = [
        r'```xml\s*(<reaction.*?</reaction>)\s*```',
        r'```\s*(<reaction.*?</reaction>)\s*```',
        r'(?:xml|XML):\s*(<reaction.*?</reaction>)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def parse_reaction_xml(xml_text: str, strict: bool = False) -> Dict[str, Any]:
    """
    Parse XML text containing chemical reaction information
    
    Args:
        xml_text: XML text to parse
        strict: Whether to use strict parsing mode
        
    Returns:
        Parsed reaction data as dictionary
    """
    try:
        # Extract XML content
        xml_content = extract_xml_from_text(xml_text)
        
        if xml_content is None:
            if strict:
                raise ValueError("No valid XML reaction found in text")
            else:
                logger.warning("No XML found, attempting fallback parsing")
                return parse_text_fallback(xml_text)
        
        # Parse XML
        root = ET.fromstring(xml_content)
        
        # Initialize reaction data structure
        reaction_data = {
            "reactants": [],
            "reagents": [],
            "solvents": [],
            "catalysts": [],
            "workups": [],
            "products": [],
            "conditions": []
        }
        
        # Parse each element
        for element in root:
            tag = element.tag.lower()
            
            if tag == "reactant":
                reaction_data["reactants"].append(_parse_compound_element(element))
            elif tag == "reagent":
                reaction_data["reagents"].append(_parse_compound_element(element))
            elif tag == "solvent":
                reaction_data["solvents"].append(_parse_compound_element(element))
            elif tag == "catalyst":
                reaction_data["catalysts"].append(_parse_compound_element(element))
            elif tag == "workup":
                reaction_data["workups"].append(_parse_workup_element(element))
            elif tag == "product":
                reaction_data["products"].append(_parse_product_element(element))
            elif tag == "condition":
                reaction_data["conditions"].append(_parse_condition_element(element))
        
        # Remove empty categories
        reaction_data = {k: v for k, v in reaction_data.items() if v}
        
        logger.debug(f"Successfully parsed XML with {len(reaction_data)} categories")
        return reaction_data
        
    except ET.ParseError as e:
        error_msg = f"XML parsing error: {str(e)}"
        logger.error(error_msg)
        
        if strict:
            raise ValueError(error_msg)
        else:
            return {"error": error_msg, "raw_text": xml_text}
    
    except Exception as e:
        error_msg = f"Unexpected parsing error: {str(e)}"
        logger.error(error_msg)
        
        if strict:
            raise ValueError(error_msg)
        else:
            return {"error": error_msg}


def _parse_compound_element(element: ET.Element) -> Dict[str, str]:
    """Parse a compound element (reactant, reagent, solvent, catalyst)"""
    return {
        "name": element.get("name", "").strip(),
        "amount": element.get("amount", "").strip(),
        "purity": element.get("purity", "").strip(),
        "supplier": element.get("supplier", "").strip()
    }


def _parse_workup_element(element: ET.Element) -> Dict[str, str]:
    """Parse a workup element"""
    return {
        "type": element.get("type", "").strip(),
        "duration": element.get("duration", "").strip(),
        "temperature": element.get("temperature", "").strip(),
        "note": element.get("note", "").strip(),
        "description": element.text.strip() if element.text else ""
    }


def _parse_product_element(element: ET.Element) -> Dict[str, str]:
    """Parse a product element"""
    return {
        "name": element.get("name", "").strip(),
        "yield": element.get("yield", "").strip(),
        "purity": element.get("purity", "").strip(),
        "appearance": element.get("appearance", "").strip(),
        "melting_point": element.get("mp", "").strip(),
        "boiling_point": element.get("bp", "").strip()
    }


def _parse_condition_element(element: ET.Element) -> Dict[str, str]:
    """Parse a condition element"""
    return {
        "type": element.get("type", "").strip(),
        "value": element.get("value", "").strip(),
        "unit": element.get("unit", "").strip(),
        "duration": element.get("duration", "").strip(),
        "note": element.get("note", "").strip()
    }


def parse_text_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback parser for text that doesn't contain proper XML
    
    Args:
        text: Text to parse
        
    Returns:
        Best-effort parsed data
    """
    logger.info("Using fallback text parsing")
    
    # Simple regex patterns to extract information
    patterns = {
        "reactants": [
            r"reactant[s]?:?\s*([^\n]+)",
            r"starting material[s]?:?\s*([^\n]+)",
        ],
        "reagents": [
            r"reagent[s]?:?\s*([^\n]+)",
            r"catalyst[s]?:?\s*([^\n]+)",
        ],
        "products": [
            r"product[s]?:?\s*([^\n]+)",
            r"yield[s]?:?\s*([^\n]+)",
        ],
        "solvents": [
            r"solvent[s]?:?\s*([^\n]+)",
        ]
    }
    
    fallback_data = {"parsing_method": "fallback"}
    
    for category, category_patterns in patterns.items():
        items = []
        for pattern in category_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    items.append({"name": match.strip(), "source": "fallback_parsing"})
        
        if items:
            fallback_data[category] = items
    
    return fallback_data


def format_reaction_summary(reaction_data: Dict[str, Any]) -> str:
    """
    Format reaction data into a readable summary
    
    Args:
        reaction_data: Parsed reaction data
        
    Returns:
        Formatted text summary
    """
    if "error" in reaction_data:
        return f"Parsing Error: {reaction_data['error']}"
    
    summary_parts = []
    
    # Format each category
    for category, items in reaction_data.items():
        if not items or category in ["parsing_method"]:
            continue
        
        category_title = category.replace("_", " ").title()
        summary_parts.append(f"\n{category_title}:")
        
        for item in items:
            if isinstance(item, dict):
                item_parts = []
                for key, value in item.items():
                    if value and key not in ["source"]:
                        if key == "name":
                            item_parts.insert(0, value)
                        else:
                            item_parts.append(f"{key}: {value}")
                
                if item_parts:
                    summary_parts.append(f"  - {', '.join(item_parts)}")
            else:
                summary_parts.append(f"  - {item}")
    
    return "\n".join(summary_parts) if summary_parts else "No reaction information extracted"
