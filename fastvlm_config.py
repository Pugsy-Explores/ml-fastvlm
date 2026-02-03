# fastvlm_config.py
"""
Legacy FastVLM config module.

This module is maintained for backward compatibility.
New code should use core.config.load_fastvlm_config() instead.
"""

import os
import logging
from typing import Optional

# Import from centralized config
from .core.config import FastVLMConfig, load_fastvlm_config as _load_fastvlm_config

logger = logging.getLogger("FastVLMConfig")


# Re-export for backward compatibility
__all__ = ["FastVLMConfig", "load_fastvlm_config"]


def load_fastvlm_config(toml_path: Optional[str] = None) -> FastVLMConfig:
    """
    Load config from TOML file, then apply env overrides.
    
    This is a compatibility wrapper around core.config.load_fastvlm_config().
    """
    return _load_fastvlm_config(toml_path)
