"""
Tests for FastVLM API endpoints and server functionality.

Tests cover:
- Server initialization
- Health check endpoints
- Configuration usage in server
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Try importing Flask first
try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Import server components
SERVER_AVAILABLE = False
IMPORT_ERROR = None
if FLASK_AVAILABLE:
    try:
        from fastvlm_server import app, server_config
        from core.config import load_server_config, load_fastvlm_config
        SERVER_AVAILABLE = True
    except ImportError as e:
        IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = "Flask not available"


@pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server imports failed: {IMPORT_ERROR}")
class TestServerConfig:
    """Test server configuration loading."""

    def test_server_config_loaded(self):
        """Test that server config is loaded."""
        assert server_config is not None
        assert hasattr(server_config, "port")
        assert hasattr(server_config, "workers")
        assert hasattr(server_config, "log_level")

    def test_server_config_from_env(self):
        """Test that server config respects environment variables."""
        with patch.dict(os.environ, {"FASTVLM_PORT": "8080", "FASTVLM_WORKERS": "4"}):
            cfg = load_server_config()
            assert cfg.port == 8080
            assert cfg.workers == 4


@pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server imports failed: {IMPORT_ERROR}")
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_healthz_endpoint(self):
        """Test /healthz endpoint."""
        with app.test_client() as client:
            response = client.get("/healthz")
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] == "ok"

    def test_readyz_endpoint_structure(self):
        """Test /readyz endpoint structure (may fail if model not loaded)."""
        with app.test_client() as client:
            response = client.get("/readyz")
            # Should return 200 or 500, but structure should be consistent
            assert response.status_code in [200, 500]
            data = response.get_json()
            assert "status" in data


@pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server imports failed: {IMPORT_ERROR}")
class TestServerImports:
    """Test that server can import all required modules."""

    def test_server_imports_config(self):
        """Test that server imports config correctly."""
        from fastvlm_server import app
        assert app is not None
        assert isinstance(app, Flask)

    def test_server_uses_centralized_config(self):
        """Test that server uses centralized config system."""
        # Verify that server_config is loaded from core.config
        from fastvlm_server import server_config
        from core.config import ServerConfig

        assert isinstance(server_config, ServerConfig)


@pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server imports failed: {IMPORT_ERROR}")
class TestConfigIntegration:
    """Test configuration integration with server."""

    def test_fastvlm_config_loaded_in_server(self):
        """Test that FastVLM config is loaded in server context."""
        # This test verifies that the server can load configs
        # without errors (actual model loading may fail, but config should work)
        try:
            cfg = load_fastvlm_config()
            assert cfg is not None
            assert hasattr(cfg, "model_path")
        except RuntimeError as e:
            # Expected if model_path is not set, but config loading should work
            assert "model_path" in str(e).lower()

    def test_server_config_environment_override(self):
        """Test that server config can be overridden via environment."""
        original_port = server_config.port

        with patch.dict(os.environ, {"FASTVLM_PORT": "9999"}):
            new_cfg = load_server_config()
            assert new_cfg.port == 9999
            assert new_cfg.port != original_port


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
