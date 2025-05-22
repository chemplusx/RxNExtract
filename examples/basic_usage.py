"""
Basic usage example for chemistry LLM inference
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor, setup_logging


def main():
    """Basic usage example"""
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize extractor (update path to your model)
    model_path = "./chemistry-qlora-xml-model"
    
    print("Basic Chemistry LLM Usage Example")
    print("=" * 40)
    
    try:
        extractor = ChemistryReactionExtractor(model_path=model_path)
        
        # Example procedure
        procedure = """
        Dissolve 3.2 g of sodium acetate in 50 mL of water. 
        Add 5 mL of glacial acetic acid and heat to 80°C for 2 hours.
        Cool the mixture and extract with ethyl acetate (3 × 20 mL).
        Dry the organic layer over anhydrous sodium sulfate.
        Remove the solvent under vacuum to give 2.1 g of product (65% yield).
        """
        
        print(f"Analyzing procedure:")
        print(procedure.strip())
        print("\n" + "-" * 40)
        
        # Analyze the procedure
        result = extractor.analyze_procedure(procedure)
        
        if result["success"]:
            print("EXTRACTED COMPONENTS:")
            
            data = result["extracted_data"]
            for category, items in data.items():
                if items:
                    print(f"\n{category.upper()}:")
                    for item in items:
                        if isinstance(item, dict):
                            parts = [f"{k}: {v}" for k, v in item.items() if v]
                            print(f"  - {', '.join(parts)}")
                        else:
                            print(f"  - {item}")
            
            print(f"\nProcessing time: {result.get('processing_time_seconds', 0):.2f}s")
        else:
            print(f"Analysis failed: {result.get('error')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to update the model_path to point to your trained model.")


if __name__ == "__main__":
    main()