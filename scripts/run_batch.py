"""
Script to run chemistry LLM in batch processing mode
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import ChemistryReactionExtractor
from chemistry_llm.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run Chemistry LLM in batch mode")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--input-file", required=True, help="Input file with procedures")
    parser.add_argument("--output-file", required=True, help="Output JSON file")
    parser.add_argument("--base-model", help="Base model name")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load procedures
    print(f"Loading procedures from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        procedures = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(procedures)} procedures to process")
    
    # Initialize extractor
    print("Loading Chemistry LLM...")
    extractor = ChemistryReactionExtractor(
        model_path=args.model_path,
        base_model_name=args.base_model,
        device=args.device
    )
    
    # Process procedures
    results = []
    successful = 0
    
    print("Processing procedures...")
    for i, procedure in enumerate(tqdm(procedures), 1):
        try:
            result = extractor.analyze_procedure(procedure)
            results.append(result)
            
            if result["success"]:
                successful += 1
                
        except Exception as e:
            results.append({
                "procedure": procedure,
                "error": str(e),
                "success": False
            })
    
    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Completed: {successful}/{len(procedures)} procedures processed successfully")


if __name__ == "__main__":
    main()