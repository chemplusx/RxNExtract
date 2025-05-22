"""
Custom prompting example for chemistry LLM inference
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor, PromptBuilder, setup_logging


def main():
    """Custom prompting example"""
    # Setup logging
    setup_logging(level="INFO")
    
    print("Custom Prompting Example")
    print("=" * 30)
    
    # Custom configuration
    custom_config = {
        "prompts": {
            "use_cot": True,
            "cot_steps": [
                "Identify all chemical compounds mentioned",
                "Determine their roles (reactant, product, solvent, etc.)",
                "Extract quantities and conditions",
                "Note any safety or procedural details"
            ]
        },
        "model": {
            "default_temperature": 0.05,  # Lower temperature for more focused output
            "max_new_tokens": 400
        }
    }
    
    try:
        # Initialize extractor with custom config
        model_path = "./chemistry-qlora-xml-model"
        extractor = ChemistryReactionExtractor(
            model_path=model_path,
            config=custom_config
        )
        
        # Test different prompting approaches
        procedure = """
        In a fume hood, carefully add 25 mL of concentrated sulfuric acid to 100 mL of cold water.
        Dissolve 5.0 g of copper sulfate pentahydrate in this solution.
        Add 2.0 g of zinc powder slowly while stirring (CAUTION: hydrogen gas evolution).
        Continue stirring for 30 minutes until the blue color disappears.
        Filter the mixture and collect the copper precipitate.
        Wash with distilled water and dry at 110°C for 2 hours.
        """
        
        print("Testing custom Chain-of-Thought prompting...")
        print(f"Procedure: {procedure.strip()}")
        print("\n" + "-" * 50)
        
        # Analyze with custom CoT
        result = extractor.analyze_procedure(procedure, return_raw=True)
        
        if result["success"]:
            print("CUSTOM CoT ANALYSIS RESULTS:")
            
            data = result["extracted_data"]
            for category, items in data.items():
                if items:
                    print(f"\n{category.upper()}:")
                    for item in items:
                        if isinstance(item, dict):
                            parts = [f"{k}: {v}" for k, v in item.items() if v and k != "source"]
                            print(f"  - {', '.join(parts)}")
                        else:
                            print(f"  - {item}")
            
            print(f"\nProcessing time: {result.get('processing_time_seconds', 0):.2f}s")
            
            print(f"\nRaw model output:")
            print("-" * 30)
            print(result["raw_output"])
        else:
            print(f"Analysis failed: {result.get('error')}")
        
        # Demonstrate different temperature settings
        print(f"\n{'='*50}")
        print("COMPARING DIFFERENT TEMPERATURES")
        print(f"{'='*50}")
        
        temperatures = [0.0, 0.3, 0.7]
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            print("-" * 20)
            
            try:
                result = extractor.extract_reaction(
                    procedure, 
                    temperature=temp,
                    max_new_tokens=200
                )
                print(f"Output length: {len(result)} characters")
                print(f"First 100 chars: {result[:100]}...")
            except Exception as e:
                print(f"Error at temperature {temp}: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to update the model_path to point to your trained model.")