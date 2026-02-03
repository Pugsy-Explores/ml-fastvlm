"""
Utility functions for applying configuration settings.
"""

import os
from .config import load_system_config


def apply_system_config():
    """
    Apply system-level configuration to environment variables.
    
    This should be called early in script execution, before importing
    PyTorch or other libraries that read these environment variables.
    """
    sys_config = load_system_config()
    
    os.environ.setdefault("CUDA_DEVICE_ORDER", sys_config.cuda_device_order)
    os.environ.setdefault("PYTORCH_SDP_KERNEL", sys_config.pytorch_sdp_kernel)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", sys_config.tf_cpp_min_log_level)


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from configuration.
    
    Returns:
        API key string
        
    Raises:
        RuntimeError: If API key is not configured
    """
    from .config import load_external_config
    
    ext_config = load_external_config()
    if not ext_config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in environment or config file.")
    
    return ext_config.openai_api_key
