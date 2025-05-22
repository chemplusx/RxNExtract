"""
Script to run chemistry LLM in interactive mode
"""

import sys
import argparse
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor
from chemistry_llm.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run Chemistry LLM in interactive mode")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", help="Base model name")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Initialize extractor
    print("Loading Chemistry LLM...")
    extractor = ChemistryReactionExtractor(
        model_path=args.model_path,
        base_model_name=args.base_model,
        device=args.device
    )
    
    # Interactive loop
    print("\n" + "="*60)
    print("CHEMISTRY LLM INTERACTIVE MODE")
    print("="*60)
    print("Enter chemical procedures to analyze. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            procedure = input("Enter procedure: ").strip()
            
            if procedure.lower() in ['quit', 'exit', 'q']:
                break
            
            if not procedure:
                continue
            
            print("\nAnalyzing...")
            result = extractor.analyze_procedure(procedure)
            
            if result["success"]:
                print("\nExtracted Information:")
                data = result["extracted_data"]
                
                for category, items in data.items():
                    if items:
                        print(f"\n{category.upper()}:")
                        for item in items:
                            if isinstance(item, dict):
                                item_str = ", ".join([f"{k}: {v}" for k, v in item.items() if v])
                                print(f"  - {item_str}")
                            else:
                                print(f"  - {item}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()