"""
Script to run chemistry LLM with example procedures
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor
from chemistry_llm.utils import setup_logging, get_memory_info


def main():
    """Run example analysis with sample procedures"""
    # Setup logging
    setup_logging(level="INFO")
    
    # Print system info
    print("="*60)
    print("CHEMISTRY LLM EXAMPLE")
    print("="*60)
    
    memory_info = get_memory_info()
    print(f"System Memory: {memory_info['cpu_memory']['available_gb']:.1f}GB available")
    if "gpu_memory" in memory_info:
        print(f"GPU Memory: {memory_info['gpu_memory']['free_gb']:.1f}GB available")
    
    # Model path (you need to update this)
    model_path = "./chemistry-qlora-xml-model"  # Update this path
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please update the model_path in this script to point to your trained model.")
        return
    
    # Initialize extractor
    print(f"\nLoading model from: {model_path}")
    extractor = ChemistryReactionExtractor(model_path=model_path)
    
    # Example procedures
    examples = [
        {
            "name": "Benzoic Acid Synthesis",
            "procedure": """
            Add 5.0 g of benzoic acid to 100 mL of hot water and stir until dissolved.
            Slowly add 10 mL of concentrated hydrochloric acid while maintaining temperature.
            Cool the solution in an ice bath for 30 minutes to precipitate the product.
            Filter the crystals and wash with cold water.
            Dry the product in a vacuum oven at 60°C for 2 hours to obtain 4.2 g (84% yield).
            """
        },
        {
            "name": "Esterification Reaction",
            "procedure": """
            In a 250 mL round-bottom flask, combine 10.0 g of acetic acid with 15 mL of methanol.
            Add 2 drops of concentrated sulfuric acid as catalyst.
            Attach a reflux condenser and heat the mixture at 65°C for 4 hours.
            Cool to room temperature and neutralize with sodium bicarbonate solution.
            Extract the product with diethyl ether (3 × 25 mL).
            Dry over anhydrous magnesium sulfate and concentrate to give 8.7 g of methyl acetate.
            """
        }
    ]
    
    # Process examples
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}: {example['name']}")
        print(f"{'='*60}")
        print(f"Procedure: {example['procedure'].strip()}")
        print(f"\nAnalyzing...")
        
        try:
            result = extractor.analyze_procedure(example["procedure"], return_raw=True)
            
            if result["success"]:
                print("\nEXTRACTED INFORMATION:")
                print("-" * 40)
                
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
                
                print(f"\nProcessing time: {result.get('processing_time_seconds', 0):.2f} seconds")
                print(f"\nRaw model output:")
                print(result["raw_output"])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error processing example: {str(e)}")
    
    print(f"\n{'='*60}")
    print("EXAMPLE ANALYSIS COMPLETED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    print("Exiting script.")
    print("="*60)