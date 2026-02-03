"""
Comprehensive tests for FastVLM centralized configuration system.

Tests cover:
- TOML file loading and parsing
- Environment variable overrides
- Import functionality
- All config classes
- Utility functions
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# Import config classes and functions
from core.config import (
    FastVLMConfig,
    RouterConfig,
    ServerConfig,
    MediaConfig,
    SystemConfig,
    ExternalConfig,
    load_fastvlm_config,
    load_router_config,
    load_server_config,
    load_media_config,
    load_system_config,
    load_external_config,
    get_all_configs,
)

# Import utility functions
from core.utils import apply_system_config, get_openai_api_key


class TestTOMLLoading:
    """Test TOML file loading and parsing."""

    def test_load_fastvlm_config_from_toml(self, tmp_path):
        """Test loading FastVLM config from TOML file."""
        toml_content = """
[fastvlm]
model_path = "/test/model/path"
device = "cpu"
scene_threshold = 25.0
frame_similarity_threshold = 0.85
max_video_seconds = 60.0
max_resolution = 720
max_frames = 16
max_context_chars = 128
enable_summary = false
enable_analysis = true
log_level = "DEBUG"
"""
        toml_file = tmp_path / "test_fastvlm.toml"
        toml_file.write_text(toml_content)

        cfg = load_fastvlm_config(str(toml_file))

        assert cfg.model_path == "/test/model/path"
        assert cfg.device == "cpu"
        assert cfg.scene_threshold == 25.0
        assert cfg.frame_similarity_threshold == 0.85
        assert cfg.max_video_seconds == 60.0
        assert cfg.max_resolution == 720
        assert cfg.max_frames == 16
        assert cfg.max_context_chars == 128
        assert cfg.enable_summary is False
        assert cfg.enable_analysis is True
        assert cfg.log_level == "DEBUG"

    def test_load_router_config_from_toml(self, tmp_path):
        """Test loading Router config from TOML file."""
        toml_content = """
[router]
gpu_index = 1
backend_base_port = 8000
router_port = 9001
target_vram_fraction = 0.8
target_ram_fraction = 0.9
max_workers = 8
python_bin = "python"
server_module = "test.server"
max_concurrent_per_worker = 4
check_ready_timeout_sec = 600
check_ready_interval_sec = 5
log_level = "WARNING"
"""
        toml_file = tmp_path / "test_router.toml"
        toml_file.write_text(toml_content)

        cfg = load_router_config(str(toml_file))

        assert cfg.gpu_index == 1
        assert cfg.backend_base_port == 8000
        assert cfg.router_port == 9001
        assert cfg.target_vram_fraction == 0.8
        assert cfg.target_ram_fraction == 0.9
        assert cfg.max_workers == 8
        assert cfg.python_bin == "python"
        assert cfg.server_module == "test.server"
        assert cfg.max_concurrent_per_worker == 4
        assert cfg.check_ready_timeout_sec == 600
        assert cfg.check_ready_interval_sec == 5
        assert cfg.log_level == "WARNING"

    def test_load_server_config_from_toml(self, tmp_path):
        """Test loading Server config from TOML file."""
        toml_content = """
[server]
port = 8080
workers = 4
log_level = "ERROR"
"""
        toml_file = tmp_path / "test_server.toml"
        toml_file.write_text(toml_content)

        cfg = load_server_config(str(toml_file))

        assert cfg.port == 8080
        assert cfg.workers == 4
        assert cfg.log_level == "ERROR"

    def test_load_media_config_from_toml(self, tmp_path):
        """Test loading Media config from TOML file."""
        toml_content = """
[media]
max_download_size_bytes = 1073741824
download_timeout_seconds = 600.0
chunk_size = 2097152
"""
        toml_file = tmp_path / "test_media.toml"
        toml_file.write_text(toml_content)

        cfg = load_media_config(str(toml_file))

        assert cfg.max_download_size_bytes == 1073741824  # 1 GiB
        assert cfg.download_timeout_seconds == 600.0
        assert cfg.chunk_size == 2097152  # 2 MiB

    def test_load_all_configs_from_toml(self, tmp_path):
        """Test loading all configs from a single TOML file."""
        toml_content = """
[fastvlm]
model_path = "/test/model"
device = "cuda"

[router]
gpu_index = 0
router_port = 9000

[server]
port = 7860

[media]
max_download_size_bytes = 2147483648
"""
        toml_file = tmp_path / "test_all.toml"
        toml_file.write_text(toml_content)

        configs = get_all_configs(str(toml_file))

        assert "fastvlm" in configs
        assert "router" in configs
        assert "server" in configs
        assert "media" in configs
        assert "system" in configs
        assert "external" in configs

        assert configs["fastvlm"].model_path == "/test/model"
        assert configs["router"].router_port == 9000
        assert configs["server"].port == 7860

    def test_missing_toml_file_uses_defaults(self):
        """Test that missing TOML file falls back to defaults."""
        with patch.dict(os.environ, {"FASTVLM_MODEL_PATH": "/test/model"}):
            cfg = load_fastvlm_config("/nonexistent/path.toml")
            assert cfg.model_path == "/test/model"
            assert cfg.device == "cuda"  # default
            assert cfg.scene_threshold == 30.0  # default


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_env_override_fastvlm_config(self, tmp_path):
        """Test that environment variables override TOML values."""
        toml_content = """
[fastvlm]
model_path = "/toml/model/path"
device = "cpu"
scene_threshold = 25.0
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        env_vars = {
            "FASTVLM_MODEL_PATH": "/env/model/path",
            "FASTVLM_DEVICE": "cuda",
            "FASTVLM_SCENE_THRESHOLD": "35.0",
        }

        with patch.dict(os.environ, env_vars):
            cfg = load_fastvlm_config(str(toml_file))

            assert cfg.model_path == "/env/model/path"  # env overrides
            assert cfg.device == "cuda"  # env overrides
            assert cfg.scene_threshold == 35.0  # env overrides

    def test_env_override_router_config(self, tmp_path):
        """Test environment variable overrides for Router config."""
        toml_content = """
[router]
router_port = 9000
max_workers = 4
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        env_vars = {
            "FASTVLM_ROUTER_PORT": "9500",
            "FASTVLM_MAX_WORKERS": "8",
        }

        with patch.dict(os.environ, env_vars):
            cfg = load_router_config(str(toml_file))

            assert cfg.router_port == 9500  # env overrides
            assert cfg.max_workers == 8  # env overrides

    def test_env_override_server_config(self, tmp_path):
        """Test environment variable overrides for Server config."""
        toml_content = """
[server]
port = 7860
workers = 1
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        env_vars = {
            "FASTVLM_PORT": "8080",
            "FASTVLM_WORKERS": "2",
        }

        with patch.dict(os.environ, env_vars):
            cfg = load_server_config(str(toml_file))

            assert cfg.port == 8080  # env overrides
            assert cfg.workers == 2  # env overrides

    def test_env_override_media_config(self, tmp_path):
        """Test environment variable overrides for Media config."""
        toml_content = """
[media]
max_download_size_bytes = 1073741824
download_timeout_seconds = 300.0
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        env_vars = {
            "FASTVLM_MAX_DOWNLOAD_SIZE_BYTES": "2147483648",
            "FASTVLM_DOWNLOAD_TIMEOUT_SECONDS": "600.0",
        }

        with patch.dict(os.environ, env_vars):
            cfg = load_media_config(str(toml_file))

            assert cfg.max_download_size_bytes == 2147483648  # env overrides
            assert cfg.download_timeout_seconds == 600.0  # env overrides

    def test_env_override_system_config(self):
        """Test environment variable overrides for System config."""
        env_vars = {
            "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
            "PYTORCH_SDP_KERNEL": "flash",
            "TF_CPP_MIN_LOG_LEVEL": "0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            cfg = load_system_config()

            assert cfg.cuda_device_order == "FASTEST_FIRST"
            assert cfg.pytorch_sdp_kernel == "flash"
            assert cfg.tf_cpp_min_log_level == "0"

    def test_env_override_external_config(self):
        """Test environment variable overrides for External config."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key-12345",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            cfg = load_external_config()

            assert cfg.openai_api_key == "sk-test-key-12345"

    def test_bool_env_vars(self, tmp_path):
        """Test boolean environment variable parsing."""
        toml_content = """
[fastvlm]
model_path = "/test/model"
enable_summary = true
enable_analysis = false
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        # Test various boolean formats
        for env_val in ["1", "true", "yes", "y", "on", "TRUE", "True"]:
            with patch.dict(os.environ, {"FASTVLM_ENABLE_SUMMARY": env_val}):
                cfg = load_fastvlm_config(str(toml_file))
                assert cfg.enable_summary is True, f"Failed for: {env_val}"

        for env_val in ["0", "false", "no", "n", "off", "FALSE", "False", ""]:
            with patch.dict(os.environ, {"FASTVLM_ENABLE_SUMMARY": env_val}):
                cfg = load_fastvlm_config(str(toml_file))
                assert cfg.enable_summary is False, f"Failed for: {env_val}"


class TestConfigDefaults:
    """Test default values when no TOML or env vars are provided."""

    def test_fastvlm_config_defaults(self):
        """Test FastVLM config defaults."""
        with patch.dict(os.environ, {"FASTVLM_MODEL_PATH": "/test/model"}, clear=False):
            cfg = load_fastvlm_config("/nonexistent.toml")

            assert cfg.device == "cuda"
            assert cfg.scene_threshold == 30.0
            assert cfg.frame_similarity_threshold == 0.90
            assert cfg.max_video_seconds == 90.0
            assert cfg.max_resolution == 1080
            assert cfg.max_frames == 24
            assert cfg.max_context_chars == 256
            assert cfg.enable_summary is True
            assert cfg.enable_analysis is False
            assert cfg.log_level == "INFO"

    def test_router_config_defaults(self):
        """Test Router config defaults."""
        cfg = load_router_config("/nonexistent.toml")

        assert cfg.gpu_index == 0
        assert cfg.backend_base_port == 7860
        assert cfg.router_port == 9000
        assert cfg.target_vram_fraction == 0.7
        assert cfg.target_ram_fraction == 0.8
        assert cfg.max_workers == 4
        assert cfg.python_bin == "python3"
        assert cfg.max_concurrent_per_worker == 2
        assert cfg.check_ready_timeout_sec == 300
        assert cfg.check_ready_interval_sec == 2

    def test_server_config_defaults(self):
        """Test Server config defaults."""
        cfg = load_server_config("/nonexistent.toml")

        assert cfg.port == 7860
        assert cfg.workers == 1
        assert cfg.log_level == "INFO"

    def test_media_config_defaults(self):
        """Test Media config defaults."""
        cfg = load_media_config("/nonexistent.toml")

        assert cfg.max_download_size_bytes == 2 * 1024**3  # 2 GiB
        assert cfg.download_timeout_seconds == 300.0
        assert cfg.chunk_size == 1024 * 1024  # 1 MiB

    def test_system_config_defaults(self):
        """Test System config defaults."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_system_config()

            assert cfg.cuda_device_order == "PCI_BUS_ID"
            assert cfg.pytorch_sdp_kernel == "math"
            assert cfg.tf_cpp_min_log_level == "3"


class TestErrorHandling:
    """Test error handling and validation."""

    def test_missing_model_path_raises_error(self):
        """Test that missing model_path raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="model_path is not configured"):
                load_fastvlm_config("/nonexistent.toml")

    def test_invalid_integer_env_var_uses_default(self, tmp_path):
        """Test that invalid integer env vars fall back to defaults."""
        toml_content = """
[fastvlm]
model_path = "/test/model"
max_resolution = 1080
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        with patch.dict(os.environ, {"FASTVLM_MAX_RES": "invalid"}):
            cfg = load_fastvlm_config(str(toml_file))
            # Should use TOML value or default, not crash
            assert cfg.max_resolution == 1080

    def test_invalid_float_env_var_uses_default(self, tmp_path):
        """Test that invalid float env vars fall back to defaults."""
        toml_content = """
[fastvlm]
model_path = "/test/model"
scene_threshold = 30.0
"""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(toml_content)

        with patch.dict(os.environ, {"FASTVLM_SCENE_THRESHOLD": "not_a_float"}):
            cfg = load_fastvlm_config(str(toml_file))
            # Should use TOML value or default, not crash
            assert cfg.scene_threshold == 30.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_apply_system_config(self):
        """Test that apply_system_config sets environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            apply_system_config()

            assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            assert os.environ.get("PYTORCH_SDP_KERNEL") == "math"
            assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "3"

    def test_apply_system_config_with_overrides(self):
        """Test that apply_system_config respects existing env vars."""
        env_vars = {
            "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # setdefault should not override existing values
            apply_system_config()
            assert os.environ.get("CUDA_DEVICE_ORDER") == "FASTEST_FIRST"

    def test_get_openai_api_key_from_env(self):
        """Test getting OpenAI API key from environment."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key-from-env",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            key = get_openai_api_key()
            assert key == "sk-test-key-from-env"

    def test_get_openai_api_key_missing_raises_error(self):
        """Test that missing OpenAI API key raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
                get_openai_api_key()


class TestImports:
    """Test import functionality."""

    def test_import_config_classes(self):
        """Test that all config classes can be imported."""
        from core.config import (
            FastVLMConfig,
            RouterConfig,
            ServerConfig,
            MediaConfig,
            SystemConfig,
            ExternalConfig,
        )

        assert FastVLMConfig is not None
        assert RouterConfig is not None
        assert ServerConfig is not None
        assert MediaConfig is not None
        assert SystemConfig is not None
        assert ExternalConfig is not None

    def test_import_load_functions(self):
        """Test that all load functions can be imported."""
        from core.config import (
            load_fastvlm_config,
            load_router_config,
            load_server_config,
            load_media_config,
            load_system_config,
            load_external_config,
            get_all_configs,
        )

        assert callable(load_fastvlm_config)
        assert callable(load_router_config)
        assert callable(load_server_config)
        assert callable(load_media_config)
        assert callable(load_system_config)
        assert callable(load_external_config)
        assert callable(get_all_configs)

    def test_import_utils(self):
        """Test that utility functions can be imported."""
        from core.utils import apply_system_config, get_openai_api_key

        assert callable(apply_system_config)
        assert callable(get_openai_api_key)

    def test_import_from_core_init(self):
        """Test importing from core.__init__."""
        from core import config, utils

        assert config is not None
        assert utils is not None


class TestConfigPathResolution:
    """Test config path resolution via FASTVLM_CONFIG_PATH."""

    def test_custom_config_path_env_var(self, tmp_path):
        """Test that FASTVLM_CONFIG_PATH env var is respected."""
        toml_content = """
[fastvlm]
model_path = "/custom/path/model"
"""
        toml_file = tmp_path / "custom_config.toml"
        toml_file.write_text(toml_content)

        env_vars = {
            "FASTVLM_CONFIG_PATH": str(toml_file),
        }

        # Remove FASTVLM_MODEL_PATH if it exists to test TOML loading
        with patch.dict(os.environ, env_vars, clear=False):
            if "FASTVLM_MODEL_PATH" in os.environ:
                del os.environ["FASTVLM_MODEL_PATH"]
            cfg = load_fastvlm_config()  # No path provided, should use env var
            assert cfg.model_path == "/custom/path/model"

    def test_explicit_path_overrides_env_var(self, tmp_path):
        """Test that explicit path parameter overrides FASTVLM_CONFIG_PATH."""
        toml1_content = """
[fastvlm]
model_path = "/path1/model"
"""
        toml2_content = """
[fastvlm]
model_path = "/path2/model"
"""
        toml1 = tmp_path / "config1.toml"
        toml2 = tmp_path / "config2.toml"
        toml1.write_text(toml1_content)
        toml2.write_text(toml2_content)

        env_vars = {
            "FASTVLM_CONFIG_PATH": str(toml1),
        }

        with patch.dict(os.environ, env_vars, clear=False):
            cfg = load_fastvlm_config(str(toml2))  # Explicit path should override
            assert cfg.model_path == "/path2/model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
