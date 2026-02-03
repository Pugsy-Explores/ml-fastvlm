"""
Centralized configuration management for FastVLM.

This module consolidates all environment variables and configuration settings
across the FastVLM codebase into a single, well-organized configuration system.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

logger = logging.getLogger("FastVLMConfig")


def _bool_from_env(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _int_from_env(name: str, default: int) -> int:
    """Parse integer from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid integer value for %s: %s, using default: %d", name, val, default)
        return default


def _float_from_env(name: str, default: float) -> float:
    """Parse float from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float value for %s: %s, using default: %f", name, val, default)
        return default


@dataclass
class FastVLMConfig:
    """Core FastVLM engine configuration."""
    model_path: str
    device: str = "cuda"
    scene_threshold: float = 30.0
    frame_similarity_threshold: float = 0.90
    max_video_seconds: float = 90.0
    max_resolution: int = 1080
    max_frames: int = 24
    max_context_chars: int = 256
    enable_summary: bool = False
    enable_analysis: bool = False
    log_level: str = "INFO"


@dataclass
class RouterConfig:
    """Router configuration for worker management and load balancing."""
    gpu_index: int = 0
    backend_base_port: int = 7860
    router_port: int = 9000
    target_vram_fraction: float = 0.7
    target_ram_fraction: float = 0.8
    max_workers: int = 4
    python_bin: str = "python3"
    server_module: str = "pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.fastvlm_server"
    max_concurrent_per_worker: int = 2
    check_ready_timeout_sec: int = 300
    check_ready_interval_sec: int = 2
    log_level: str = "INFO"


@dataclass
class ServerConfig:
    """HTTP server configuration."""
    port: int = 7860
    workers: int = 1
    log_level: str = "INFO"


@dataclass
class MediaConfig:
    """Media download and processing configuration."""
    max_download_size_bytes: int = 2 * 1024**3  # 2 GiB
    download_timeout_seconds: float = 300.0  # 5 minutes
    chunk_size: int = 1024 * 1024  # 1 MiB


@dataclass
class SystemConfig:
    """System-level configuration (CUDA, PyTorch, etc.)."""
    cuda_device_order: str = "PCI_BUS_ID"
    pytorch_sdp_kernel: str = "math"
    tf_cpp_min_log_level: str = "3"


@dataclass
class ExternalConfig:
    """External service configuration."""
    openai_api_key: Optional[str] = None


def load_fastvlm_config(toml_path: Optional[str] = None) -> FastVLMConfig:
    """
    Load FastVLM engine config from TOML file, then apply env overrides.
    
    Args:
        toml_path: Path to TOML config file. If None, uses FASTVLM_CONFIG_PATH env var
                  or defaults to "fastvlm.toml"
    
    Returns:
        FastVLMConfig instance
    
    Raises:
        RuntimeError: If model_path is not configured
    """
    toml_path = toml_path or os.getenv("FASTVLM_CONFIG_PATH", "fastvlm.toml")

    data = {}
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("fastvlm", {})
    else:
        logger.warning("Config TOML not found at %s, using defaults + env overrides", toml_path)

    def g(key, default):
        return data.get(key, default)

    cfg = FastVLMConfig(
        model_path=os.getenv("FASTVLM_MODEL_PATH", g("model_path", "")),
        device=os.getenv("FASTVLM_DEVICE", g("device", "cuda")),
        scene_threshold=_float_from_env("FASTVLM_SCENE_THRESHOLD", g("scene_threshold", 30.0)),
        frame_similarity_threshold=_float_from_env(
            "FASTVLM_FRAME_SIM_THRESHOLD", g("frame_similarity_threshold", 0.90)
        ),
        max_video_seconds=_float_from_env("FASTVLM_MAX_VIDEO_SEC", g("max_video_seconds", 90.0)),
        max_resolution=_int_from_env("FASTVLM_MAX_RES", g("max_resolution", 1080)),
        max_frames=_int_from_env("FASTVLM_MAX_FRAMES", g("max_frames", 24)),
        max_context_chars=_int_from_env("FASTVLM_MAX_CONTEXT_CHARS", g("max_context_chars", 256)),
        enable_summary=_bool_from_env("FASTVLM_ENABLE_SUMMARY", g("enable_summary", False)),
        enable_analysis=_bool_from_env("FASTVLM_ENABLE_ANALYSIS", g("enable_analysis", False)),
        log_level=os.getenv("FASTVLM_LOG_LEVEL", g("log_level", "INFO")),
    )

    if not cfg.model_path:
        raise RuntimeError("FASTVLM model_path is not configured (TOML or FASTVLM_MODEL_PATH missing).")

    return cfg


def load_router_config(toml_path: Optional[str] = None) -> RouterConfig:
    """
    Load router configuration from TOML file, then apply env overrides.
    
    Args:
        toml_path: Path to TOML config file. If None, uses FASTVLM_CONFIG_PATH env var
                  or defaults to "fastvlm.toml"
    
    Returns:
        RouterConfig instance
    """
    toml_path = toml_path or os.getenv("FASTVLM_CONFIG_PATH", "fastvlm.toml")

    data = {}
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("router", {})

    def g(key, default):
        return data.get(key, default)

    return RouterConfig(
        gpu_index=_int_from_env("FASTVLM_GPU_INDEX", g("gpu_index", 0)),
        backend_base_port=_int_from_env("FASTVLM_BACKEND_BASE_PORT", g("backend_base_port", 7860)),
        router_port=_int_from_env("FASTVLM_ROUTER_PORT", g("router_port", 9000)),
        target_vram_fraction=_float_from_env("FASTVLM_TARGET_VRAM_FRACTION", g("target_vram_fraction", 0.7)),
        target_ram_fraction=_float_from_env("FASTVLM_TARGET_RAM_FRACTION", g("target_ram_fraction", 0.8)),
        max_workers=_int_from_env("FASTVLM_MAX_WORKERS", g("max_workers", 4)),
        python_bin=os.getenv("FASTVLM_PYTHON_BIN", g("python_bin", "python3")),
        server_module=os.getenv(
            "FASTVLM_SERVER_MODULE",
            g("server_module", "pugsy_ai.pipelines.vlm_pipeline.fastvlm.ml_fastvlm.fastvlm_server")
        ),
        max_concurrent_per_worker=_int_from_env("FASTVLM_MAX_CONCURRENT_PER_WORKER", g("max_concurrent_per_worker", 2)),
        check_ready_timeout_sec=_int_from_env("FASTVLM_CHECK_READY_TIMEOUT_SEC", g("check_ready_timeout_sec", 300)),
        check_ready_interval_sec=_int_from_env("FASTVLM_CHECK_READY_INTERVAL_SEC", g("check_ready_interval_sec", 2)),
        log_level=os.getenv("FASTVLM_LOG_LEVEL", g("log_level", "INFO")),
    )


def load_server_config(toml_path: Optional[str] = None) -> ServerConfig:
    """
    Load server configuration from TOML file, then apply env overrides.
    
    Args:
        toml_path: Path to TOML config file. If None, uses FASTVLM_CONFIG_PATH env var
                  or defaults to "fastvlm.toml"
    
    Returns:
        ServerConfig instance
    """
    toml_path = toml_path or os.getenv("FASTVLM_CONFIG_PATH", "fastvlm.toml")

    data = {}
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("server", {})

    def g(key, default):
        return data.get(key, default)

    return ServerConfig(
        port=_int_from_env("FASTVLM_PORT", g("port", 7860)),
        workers=_int_from_env("FASTVLM_WORKERS", g("workers", 1)),
        log_level=os.getenv("FASTVLM_LOG_LEVEL", g("log_level", "INFO")),
    )


def load_media_config(toml_path: Optional[str] = None) -> MediaConfig:
    """
    Load media configuration from TOML file, then apply env overrides.
    
    Args:
        toml_path: Path to TOML config file. If None, uses FASTVLM_CONFIG_PATH env var
                  or defaults to "fastvlm.toml"
    
    Returns:
        MediaConfig instance
    """
    toml_path = toml_path or os.getenv("FASTVLM_CONFIG_PATH", "fastvlm.toml")

    data = {}
    if os.path.exists(toml_path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        data = raw.get("media", {})

    def g(key, default):
        return data.get(key, default)

    return MediaConfig(
        max_download_size_bytes=_int_from_env(
            "FASTVLM_MAX_DOWNLOAD_SIZE_BYTES",
            g("max_download_size_bytes", 2 * 1024**3)
        ),
        download_timeout_seconds=_float_from_env(
            "FASTVLM_DOWNLOAD_TIMEOUT_SECONDS",
            g("download_timeout_seconds", 300.0)
        ),
        chunk_size=_int_from_env("FASTVLM_CHUNK_SIZE", g("chunk_size", 1024 * 1024)),
    )


def load_system_config() -> SystemConfig:
    """
    Load system-level configuration from environment variables.
    
    Returns:
        SystemConfig instance
    """
    return SystemConfig(
        cuda_device_order=os.getenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        pytorch_sdp_kernel=os.getenv("PYTORCH_SDP_KERNEL", "math"),
        tf_cpp_min_log_level=os.getenv("TF_CPP_MIN_LOG_LEVEL", "3"),
    )


def load_external_config() -> ExternalConfig:
    """
    Load external service configuration from environment variables.
    
    Returns:
        ExternalConfig instance
    """
    return ExternalConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_all_configs(toml_path: Optional[str] = None) -> dict:
    """
    Load all configurations and return as a dictionary.
    
    Args:
        toml_path: Path to TOML config file
    
    Returns:
        Dictionary containing all config objects
    """
    return {
        "fastvlm": load_fastvlm_config(toml_path),
        "router": load_router_config(toml_path),
        "server": load_server_config(toml_path),
        "media": load_media_config(toml_path),
        "system": load_system_config(),
        "external": load_external_config(),
    }
