"""
Batch processing example for chemistry LLM inference
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor, setup_logging


def main():
    """Batch processing example"""
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize extractor (update path to your model)
    model_path = "./chemistry-qlora-xml-model"
    
    print("Batch Processing Example")
    print("=" * 30)
    
    try:
        extractor = ChemistryReactionExtractor(model_path=model_path)
        
        # Example procedures
        procedures = [
            "Mix 10 g of compound A with 20 mL of solvent B and stir for 1 hour.",
            "Heat the reaction mixture to 150°C and maintain for 3 hours under nitrogen.",
            "Cool to room temperature and add 50 mL of water dropwise.",
            "Filter the precipitate and wash with cold methanol.",
            "Dry the product under vacuum to obtain 8.5 g of pure material (85% yield)."
        ]
        
        print(f"Processing {len(procedures)} procedures...")
        
        # Process in batch
        results = extractor.batch_analyze(procedures, return_raw=False)
        
        # Display results
        successful = 0
        for i, result in enumerate(results, 1):
            print(f"\n--- Procedure {i} ---")
            print(f"Text: {result['procedure'][:60]}...")
            
            if result["success"]:
                successful += 1
                data = result["extracted_data"]
                
                # Count extracted components
                total_components = sum(len(items) for items in data.values() if isinstance(items, list))
                print(f"Status: ✓ Success ({total_components} components extracted)")
                
                # Show summary
                for category, items in data.items():
                    if items:
                        print(f"  {category}: {len(items)} items")
            else:
                print(f"Status: ✗ Failed - {result.get('error', 'Unknown error')}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Processed: {len(procedures)} procedures")
        print(f"Successful: {successful}")
        print(f"Failed: {len(procedures) - successful}")
        
        # Save results to file
        output_file = "batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to update the model_path to point to your trained model.")


if __name__ == "__main__":
    main()