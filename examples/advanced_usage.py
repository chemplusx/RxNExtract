"""
Advanced usage example demonstrating all features
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemistry_llm import (
    ChemistryReactionExtractor, 
    setup_logging, 
    get_memory_info,
    parse_reaction_xml,
    format_reaction_summary
)


def performance_benchmark(extractor, procedures, num_runs=3):
    """Benchmark extraction performance"""
    print(f"\nPerformance Benchmark ({num_runs} runs)")
    print("-" * 40)
    
    times = []
    for run in range(num_runs):
        start_time = time.time()
        
        for procedure in procedures:
            extractor.extract_reaction(procedure)
        
        run_time = time.time() - start_time
        times.append(run_time)
        print(f"Run {run + 1}: {run_time:.2f}s ({len(procedures)} procedures)")
    
    avg_time = sum(times) / len(times)
    avg_per_procedure = avg_time / len(procedures)
    
    print(f"Average total time: {avg_time:.2f}s")
    print(f"Average per procedure: {avg_per_procedure:.2f}s")


def analyze_extraction_quality(results):
    """Analyze the quality of extractions"""
    print(f"\nExtraction Quality Analysis")
    print("-" * 30)
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    
    # Count extracted components
    component_counts = {
        "reactants": 0,
        "reagents": 0,
        "solvents": 0,
        "products": 0,
        "workups": 0
    }
    
    for result in results:
        if result["success"]:
            data = result["extracted_data"]
            for category in component_counts:
                if category in data:
                    component_counts[category] += len(data[category])
    
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Component extraction:")
    for category, count in component_counts.items():
        print(f"  {category}: {count} total")


def main():
    """Advanced usage demonstration"""
    # Setup logging with file output
    setup_logging(level="INFO")
    
    print("Advanced Chemistry LLM Usage Example")
    print("=" * 45)
    
    # Display system information
    memory_info = get_memory_info()
    print(f"System Memory: {memory_info['cpu_memory']['available_gb']:.1f}GB available")
    if "gpu_memory" in memory_info:
        print(f"GPU Memory: {memory_info['gpu_memory']['free_gb']:.1f}GB available")
    
    # Advanced configuration
    advanced_config = {
        "model": {
            "default_temperature": 0.1,
            "max_new_tokens": 512,
            "repetition_penalty": 1.15
        },
        "prompts": {
            "use_cot": True,
            "cot_steps": [
                "Identify starting materials and their quantities",
                "Identify catalysts, reagents, and reaction conditions",
                "Identify solvents and their volumes",
                "Identify workup procedures and purification steps",
                "Identify final products and their yields",
                "Note any safety considerations or special equipment"
            ]
        },
        "output": {
            "include_raw": True,
            "include_timing": True
        }
    }
    
    try:
        # Initialize extractor
        model_path = "./chemistry-qlora-xml-model"
        extractor = ChemistryReactionExtractor(
            model_path=model_path,
            config=advanced_config
        )
        
        # Display model information
        model_info = extractor.get_model_info()
        print(f"\nModel Information:")
        for key, value in model_info.items():
            if key != "generation_config":
                print(f"  {key}: {value}")
        
        # Complex test procedures
        complex_procedures = [
            """
            Preparation of Aspirin (Acetylsalicylic Acid):
            Place 2.0 g of salicylic acid in a 125 mL Erlenmeyer flask.
            Add 5 mL of acetic anhydride and 3 drops of concentrated phosphoric acid as catalyst.
            Heat the mixture in a water bath at 80°C for 15 minutes with occasional swirling.
            Remove from heat and carefully add 20 mL of cold distilled water to hydrolyze excess acetic anhydride.
            Cool the mixture in an ice bath for 10 minutes to complete crystallization.
            Filter the crystals using a Büchner funnel and wash with cold water (2 × 10 mL).
            Dry the product at 100°C for 30 minutes to obtain 1.8 g of crude aspirin (75% yield).
            Recrystallize from hot ethanol-water mixture to obtain pure product, mp 135-136°C.
            """,
            
            """
            Williamson Ether Synthesis:
            In a dry 250 mL round-bottom flask under nitrogen atmosphere, dissolve 3.2 g of phenol in 50 mL of dry DMF.
            Add 2.4 g of potassium carbonate and stir for 15 minutes.
            Slowly add 2.8 mL of methyl iodide via syringe and heat to 100°C for 4 hours.
            Monitor reaction progress by TLC (hexane:ethyl acetate 4:1).
            Cool to room temperature and pour into 200 mL of water.
            Extract with diethyl ether (3 × 30 mL) and wash organic layer with brine.
            Dry over anhydrous magnesium sulfate, filter, and concentrate under reduced pressure.
            Purify by column chromatography (silica gel, hexane:ethyl acetate 9:1) to give 2.9 g of anisole (89% yield).
            """,
            
            """
            Grignard Reaction - Benzoic Acid Synthesis:
            In a dry 500 mL three-neck flask equipped with reflux condenser and addition funnel, 
            place 2.4 g of magnesium turnings and 100 mL of dry diethyl ether.
            Add a crystal of iodine and slowly add 15.7 g of bromobenzene in 50 mL of ether over 45 minutes.
            Maintain gentle reflux during addition and continue heating for 1 hour after complete addition.
            Cool the Grignard reagent to 0°C using an ice bath.
            Slowly bubble dry CO2 gas through the solution for 2 hours while maintaining temperature.
            Quench the reaction carefully with 100 mL of dilute HCl (1:1) and separate layers.
            Extract aqueous layer with ether (2 × 25 mL) and acidify to pH 1 with concentrated HCl.
            Filter the precipitated benzoic acid and recrystallize from hot water to obtain 9.8 g (80% yield).
            """
        ]
        
        print(f"\nProcessing {len(complex_procedures)} complex procedures...")
        
        # Process procedures and collect results
        results = []
        for i, procedure in enumerate(complex_procedures, 1):
            print(f"\n{'='*50}")
            print(f"PROCEDURE {i}")
            print(f"{'='*50}")
            
            result = extractor.analyze_procedure(procedure, return_raw=False, include_timing=True)
            results.append(result)
            
            if result["success"]:
                # Display formatted summary
                summary = format_reaction_summary(result["extracted_data"])
                print(summary)
                
                print(f"\nProcessing time: {result.get('processing_time_seconds', 0):.2f}s")
            else:
                print(f"Analysis failed: {result.get('error')}")
        
        # Performance analysis
        performance_benchmark(extractor, [p[:200] for p in complex_procedures])
        
        # Quality analysis
        analyze_extraction_quality(results)
        
        # Save detailed results
        output_file = "advanced_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to update the model_path to point to your trained model.")


if __name__ == "__main__":
    main()