# FastVLM Core Configuration Module

This module provides centralized configuration management for all FastVLM components.

## Overview

The core configuration system consolidates all environment variables and configuration settings across the FastVLM codebase into a single, well-organized configuration system. This provides:

- **Centralized Configuration**: All configs in one place
- **Type Safety**: Dataclass-based configs with type hints
- **Flexible Loading**: Support for TOML files and environment variables
- **Backward Compatibility**: Existing code continues to work

## Structure

```
core/
├── __init__.py          # Module exports
├── config.py            # Configuration classes and loaders
├── utils.py             # Utility functions for applying configs
└── README.md           # This file
```

## Configuration Classes

### FastVLMConfig
Core engine configuration for model paths, device settings, video processing parameters, etc.

### RouterConfig
Router configuration for worker management, load balancing, and port settings.

### ServerConfig
HTTP server configuration for port and worker thread settings.

### MediaConfig
Media download and processing configuration (timeouts, size limits, etc.).

### SystemConfig
System-level configuration (CUDA, PyTorch, TensorFlow settings).

### ExternalConfig
External service configuration (OpenAI API keys, etc.).

## Usage

### Basic Usage

```python
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.core.config import (
    load_fastvlm_config,
    load_router_config,
    load_server_config,
    load_media_config,
)

# Load configurations
fastvlm_cfg = load_fastvlm_config()
router_cfg = load_router_config()
server_cfg = load_server_config()
media_cfg = load_media_config()

# Use the configs
print(f"Model path: {fastvlm_cfg.model_path}")
print(f"Router port: {router_cfg.router_port}")
print(f"Server port: {server_cfg.port}")
```

### Loading All Configs

```python
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.core.config import get_all_configs

configs = get_all_configs()
fastvlm_cfg = configs["fastvlm"]
router_cfg = configs["router"]
server_cfg = configs["server"]
media_cfg = configs["media"]
system_cfg = configs["system"]
external_cfg = configs["external"]
```

### Applying System Configuration

For scripts that need to set environment variables early (before importing PyTorch, etc.):

```python
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.core.utils import apply_system_config

# Call this before importing PyTorch or other libraries
apply_system_config()
```

### Using External API Keys

```python
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.core.utils import get_openai_api_key

api_key = get_openai_api_key()
```

## Configuration Sources

Configurations are loaded in the following order (later sources override earlier ones):

1. **Default values** (hardcoded in config classes)
2. **TOML file** (if `fastvlm.toml` exists or `FASTVLM_CONFIG_PATH` is set)
3. **Environment variables** (highest priority)

## Environment Variables

All configuration can be overridden via environment variables. See `fastvlm.toml.example` for a complete list of available settings.

Key environment variables:
- `FASTVLM_MODEL_PATH` - Path to the model
- `FASTVLM_DEVICE` - Device to use (cuda, cpu, mps)
- `FASTVLM_ROUTER_PORT` - Router port
- `FASTVLM_PORT` - Server port
- `FASTVLM_LOG_LEVEL` - Logging level
- And many more...

## TOML Configuration

Create a `fastvlm.toml` file (or set `FASTVLM_CONFIG_PATH` to point to your config file):

```toml
[fastvlm]
model_path = "/path/to/model"
device = "cuda"
max_video_seconds = 90.0

[router]
router_port = 9000
max_workers = 4

[server]
port = 7860
workers = 1

[media]
max_download_size_bytes = 2147483648
download_timeout_seconds = 300.0
```

See `fastvlm.toml.example` for a complete example.

## Backward Compatibility

The old `fastvlm_config.py` module is maintained for backward compatibility. It now delegates to the core config system, so existing code continues to work:

```python
# Old code still works
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.fastvlm_config import (
    FastVLMConfig,
    load_fastvlm_config,
)

cfg = load_fastvlm_config()
```

## Migration Guide

### Before (Direct Environment Variable Access)

```python
import os

GPU_INDEX = int(os.getenv("FASTVLM_GPU_INDEX", "0"))
ROUTER_PORT = int(os.getenv("FASTVLM_ROUTER_PORT", "9000"))
```

### After (Centralized Config)

```python
from .core.config import load_router_config

router_config = load_router_config()
GPU_INDEX = router_config.gpu_index
ROUTER_PORT = router_config.router_port
```

## Testing

To test the configuration system:

```python
from pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.core.config import (
    load_router_config,
    load_server_config,
    load_media_config,
)

# Test loading with defaults
router_cfg = load_router_config()
assert router_cfg.router_port == 9000
assert router_cfg.max_workers == 4
```
