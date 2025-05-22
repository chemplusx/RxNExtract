"""
Device management utilities
"""

import torch
import psutil
from typing import Dict, Any, Optional


def get_optimal_device() -> str:
    """
    Automatically detect the optimal device for inference
    
    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_memory_info() -> Dict[str, Any]:
    """
    Get system memory information
    
    Returns:
        Dictionary with memory information
    """
    info = {
        "cpu_memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "percent_used": psutil.virtual_memory().percent
        }
    }
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_reserved = torch.cuda.memory_reserved(0)
        
        info["gpu_memory"] = {
            "device_name": torch.cuda.get_device_name(0),
            "total_gb": gpu_memory / (1024**3),
            "allocated_gb": gpu_allocated / (1024**3),
            "reserved_gb": gpu_reserved / (1024**3),
            "free_gb": (gpu_memory - gpu_reserved) / (1024**3)
        }
    
    return info


def check_memory_requirements(model_size_gb: float = 7.0) -> Dict[str, bool]:
    """
    Check if system meets memory requirements for model loading
    
    Args:
        model_size_gb: Estimated model size in GB
        
    Returns:
        Dictionary with requirement check results
    """
    memory_info = get_memory_info()
    
    # Rough estimates for memory requirements
    cpu_required = model_size_gb * 2  # 2x for safety margin
    gpu_required = model_size_gb * 0.5  # 4-bit quantization reduces size
    
    results = {
        "cpu_sufficient": memory_info["cpu_memory"]["available_gb"] >= cpu_required,
        "cpu_available_gb": memory_info["cpu_memory"]["available_gb"],
        "cpu_required_gb": cpu_required
    }
    
    if "gpu_memory" in memory_info:
        results.update({
            "gpu_sufficient": memory_info["gpu_memory"]["free_gb"] >= gpu_required,
            "gpu_available_gb": memory_info["gpu_memory"]["free_gb"],
            "gpu_required_gb": gpu_required,
            "gpu_recommended": True
        })
    else:
        results["gpu_recommended"] = False
    
    return results