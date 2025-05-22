"""
Command line interface for chemistry LLM inference
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

from ..core.extractor import ChemistryReactionExtractor
from ..utils.logger import setup_logging, get_logger
from ..utils.device_utils import get_memory_info, check_memory_requirements
from ..utils.xml_parser import format_reaction_summary

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return {}


def print_memory_info():
    """Print system memory information"""
    memory_info = get_memory_info()
    print("\n" + "="*50)
    print("SYSTEM MEMORY INFORMATION")
    print("="*50)
    
    cpu_mem = memory_info["cpu_memory"]
    print(f"CPU Memory: {cpu_mem['available_gb']:.1f}GB available / {cpu_mem['total_gb']:.1f}GB total")
    
    if "gpu_memory" in memory_info:
        gpu_mem = memory_info["gpu_memory"]
        print(f"GPU ({gpu_mem['device_name']}): {gpu_mem['free_gb']:.1f}GB free / {gpu_mem['total_gb']:.1f}GB total")
    else:
        print("GPU: Not available")
    
    print("="*50)


def run_interactive_mode(extractor: ChemistryReactionExtractor):
    """Run interactive mode for procedure analysis"""
    print("\n" + "="*60)
    print("CHEMISTRY REACTION EXTRACTOR - INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  - Enter a procedure to analyze")
    print("  - Type 'info' to see model information")
    print("  - Type 'memory' to see memory usage")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter procedure (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'info':
                model_info = extractor.get_model_info()
                print("\nModel Information:")
                for key, value in model_info.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == 'memory':
                print_memory_info()
                continue
            
            if not user_input:
                print("Please enter a procedure or command.")
                continue
            
            print(f"\nAnalyzing: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
            print("-" * 60)
            
            # Analyze the procedure
            result = extractor.analyze_procedure(user_input, return_raw=False)
            
            if result["success"]:
                # Display formatted results
                summary = format_reaction_summary(result["extracted_data"])
                print("EXTRACTED INFORMATION:")
                print(summary)
                
                if "processing_time_seconds" in result:
                    print(f"\nProcessing time: {result['processing_time_seconds']:.2f} seconds")
            else:
                print(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def run_batch_mode(extractor: ChemistryReactionExtractor, 
                   input_file: str, 
                   output_file: str,
                   batch_size: int = 10):
    """Run batch processing mode"""
    print(f"Processing batch file: {input_file}")
    
    # Load procedures from file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            procedures = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    if not procedures:
        print("No procedures found in input file.")
        return
    
    print(f"Found {len(procedures)} procedures to process")
    
    # Progress callback
    def progress_callback(current: int, total: int, result: Dict[str, Any]):
        print(f"Processed {current}/{total}: {'✓' if result['success'] else '✗'}")
    
    # Process in batches
    all_results = []
    for i in range(0, len(procedures), batch_size):
        batch = procedures[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} procedures)...")
        
        results = extractor.batch_analyze(
            batch, 
            return_raw=False,
            progress_callback=progress_callback
        )
        all_results.extend(results)
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        successful = sum(1 for r in all_results if r["success"])
        print(f"Summary: {successful}/{len(all_results)} procedures processed successfully")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")


def run_single_mode(extractor: ChemistryReactionExtractor, procedure: str):
    """Run single procedure analysis"""
    print(f"Analyzing procedure: {procedure[:100]}{'...' if len(procedure) > 100 else ''}")
    print("-" * 60)
    
    result = extractor.analyze_procedure(procedure, return_raw=True)
    
    if result["success"]:
        # Display formatted results
        summary = format_reaction_summary(result["extracted_data"])
        print("EXTRACTED INFORMATION:")
        print(summary)
        
        print(f"\nRaw Model Output:")
        print(result["raw_output"])
        
        if "processing_time_seconds" in result:
            print(f"\nProcessing time: {result['processing_time_seconds']:.2f} seconds")
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Chemistry LLM Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  chemistry-llm --model-path ./model --interactive
  
  # Single procedure
  chemistry-llm --model-path ./model --procedure "Add 5g NaCl to water..."
  
  # Batch processing
  chemistry-llm --model-path ./model --input procedures.txt --output results.json
  
  # Check system info
  chemistry-llm --system-info
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to the fine-tuned model directory"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    mode_group.add_argument(
        "--procedure", 
        type=str,
        help="Analyze a single procedure"
    )
    mode_group.add_argument(
        "--input", 
        type=str,
        help="Input file with procedures (one per line)"
    )
    mode_group.add_argument(
        "--system-info", 
        action="store_true",
        help="Show system information and exit"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file for batch processing results"
    )
    parser.add_argument(
        "--base-model", 
        type=str,
        help="Base model name (auto-detected if not provided)"
    )
    parser.add_argument(
        "--device", 
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=10,
        help="Batch size for batch processing"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        help="Generation temperature (overrides config)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int,
        help="Maximum tokens to generate (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Handle system info mode
    if args.system_info:
        print_memory_info()
        
        # Check requirements
        requirements = check_memory_requirements()
        print("\nMEMORY REQUIREMENTS CHECK:")
        print(f"CPU Memory: {'✓' if requirements['cpu_sufficient'] else '✗'} "
              f"({requirements['cpu_available_gb']:.1f}GB available, "
              f"{requirements['cpu_required_gb']:.1f}GB required)")
        
        if requirements.get("gpu_recommended"):
            print(f"GPU Memory: {'✓' if requirements['gpu_sufficient'] else '✗'} "
                  f"({requirements['gpu_available_gb']:.1f}GB available, "
                  f"{requirements['gpu_required_gb']:.1f}GB required)")
        
        return
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.temperature is not None:
        config.setdefault("model", {})["default_temperature"] = args.temperature
    if args.max_tokens is not None:
        config.setdefault("model", {})["max_new_tokens"] = args.max_tokens
    
    # Initialize extractor
    print("Initializing Chemistry LLM Extractor...")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    
    try:
        extractor = ChemistryReactionExtractor(
            model_path=args.model_path,
            base_model_name=args.base_model,
            device=args.device,
            config=config
        )
    except Exception as e:
        print(f"Error initializing extractor: {str(e)}")
        sys.exit(1)
    
    # Run appropriate mode
    try:
        if args.interactive:
            run_interactive_mode(extractor)
        elif args.procedure:
            run_single_mode(extractor, args.procedure)
        elif args.input:
            output_file = args.output or "extraction_results.json"
            run_batch_mode(extractor, args.input, output_file, args.batch_size)
        else:
            # Default to interactive mode
            print("No mode specified, running in interactive mode...")
            run_interactive_mode(extractor)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()