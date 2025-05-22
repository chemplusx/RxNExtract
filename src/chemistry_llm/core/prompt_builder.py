"""
Prompt building utilities with Chain-of-Thought support
"""

from typing import List, Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds optimized prompts for chemical reaction extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt builder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.prompt_config = self.config.get("prompts", {})
        self.use_cot = self.prompt_config.get("use_cot", True)
        self.cot_steps = self.prompt_config.get("cot_steps", self._default_cot_steps())
        self.xml_template = self.prompt_config.get("xml_template", self._default_xml_template())
    
    def _default_cot_steps(self) -> List[str]:
        """Default Chain-of-Thought steps"""
        return [
            "Identify Reactants: Look for starting materials, main chemical compounds being transformed",
            "Identify Reagents: Find chemicals that facilitate the reaction (catalysts, bases, acids, etc.)",
            "Identify Solvents: Look for liquids used to dissolve reactants (water, ethanol, etc.)",
            "Identify Conditions: Note temperature, time, pressure, and other reaction conditions",
            "Identify Workup Steps: Find purification, separation, or processing steps",
            "Identify Products: Look for final compounds formed, yields mentioned"
        ]
    
    def _default_xml_template(self) -> str:
        """Default XML template for output format"""
        return """<reaction>
  <reactant name="compound_name" amount="quantity units" />
  <reagent name="compound_name" amount="quantity units" />
  <solvent name="compound_name" amount="quantity units" />
  <workup type="operation_type" duration="time" note="description" />
  <product name="compound_name" yield="percentage%" />
</reaction>"""
    
    def build_cot_prompt(self, procedure_text: str) -> str:
        """
        Build a Chain-of-Thought prompt for reaction extraction
        
        Args:
            procedure_text: The chemical procedure text
            
        Returns:
            Complete CoT prompt
        """
        if not self.use_cot:
            return self.build_simple_prompt(procedure_text)
        
        prompt_parts = [
            "Extract chemical information from this procedure in XML format using step-by-step reasoning:",
            "",
            f"**Procedure:** {procedure_text.strip()}",
            "",
            "**Step-by-Step Analysis:**"
        ]
        
        # Add CoT steps
        for i, step in enumerate(self.cot_steps, 1):
            prompt_parts.append(f"{i}. **{step}**")
        
        prompt_parts.extend([
            "",
            "**XML Output Format:**",
            "Extract the information in this structured XML format:",
            self.xml_template,
            "",
            "**Analysis and XML Result:**"
        ])
        
        return "\n".join(prompt_parts)
    
    def build_simple_prompt(self, procedure_text: str) -> str:
        """
        Build a simple prompt without Chain-of-Thought
        
        Args:
            procedure_text: The chemical procedure text
            
        Returns:
            Simple prompt
        """
        return f"""Extract chemical information from this procedure in XML format:

**Procedure:** {procedure_text.strip()}

**XML Output:**
{self.xml_template}

**Result:**"""
    
    def build_custom_prompt(self, procedure_text: str, 
                           custom_instructions: str = "",
                           custom_format: str = "") -> str:
        """
        Build a custom prompt with user-defined instructions
        
        Args:
            procedure_text: The chemical procedure text
            custom_instructions: Custom extraction instructions
            custom_format: Custom output format
            
        Returns:
            Custom prompt
        """
        format_section = custom_format or self.xml_template
        instructions = custom_instructions or "Extract chemical reaction information"
        
        return f"""{instructions}

**Procedure:** {procedure_text.strip()}

**Output Format:**
{format_section}

**Result:**"""