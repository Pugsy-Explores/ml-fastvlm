"""
Tests for media_source_type flag support in API endpoints.

Tests cover:
- Local file path handling with explicit source_type
- HTTP/HTTPS URL handling with explicit source_type
- Auto-detection (backward compatibility)
- Validation and error handling
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Import components
from tmp_media import TempMedia


class TestTempMediaSourceType:
    """Test TempMedia with explicit source_type parameter."""

    def test_local_path_with_explicit_local_type(self, tmp_path):
        """Test local path with source_type='local'."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")

        with TempMedia(str(test_file), source_type="local") as local_path:
            assert os.path.exists(local_path)
            assert os.path.abspath(local_path) == os.path.abspath(str(test_file))

    def test_local_path_not_found_with_explicit_local_type(self):
        """Test that missing local file raises FileNotFoundError when source_type='local'."""
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            with TempMedia("/nonexistent/file.jpg", source_type="local"):
                pass

    @patch("requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_http_url_with_explicit_http_type(self, mock_tempfile, mock_requests_get):
        """Test HTTP URL with source_type='http'."""
        test_url = "https://example.com/image.jpg"
        
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake image data"])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        mock_requests_get.return_value = mock_response

        mock_file = Mock()
        mock_file.name = "/tmp/fake_temp_file.jpg"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_file.close = Mock()
        mock_tempfile.return_value = mock_file

        with TempMedia(test_url, source_type="http") as local_path:
            assert local_path == "/tmp/fake_temp_file.jpg"
            mock_requests_get.assert_called_once()

    def test_http_url_with_wrong_scheme(self):
        """Test that HTTP source_type with non-HTTP URL raises ValueError."""
        with pytest.raises(ValueError, match="does not match source_type 'http'"):
            with TempMedia("file:///path/to/file.jpg", source_type="http"):
                pass

    def test_invalid_source_type(self):
        """Test that invalid source_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_type"):
            TempMedia("test.jpg", source_type="invalid")

    def test_auto_detection_local_path(self, tmp_path):
        """Test auto-detection for local path (backward compatibility)."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")

        with TempMedia(str(test_file), source_type="auto") as local_path:
            assert os.path.exists(local_path)
            assert os.path.abspath(local_path) == os.path.abspath(str(test_file))

    @patch("requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_auto_detection_http_url(self, mock_tempfile, mock_requests_get):
        """Test auto-detection for HTTP URL (backward compatibility)."""
        test_url = "https://example.com/image.jpg"
        
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake image data"])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        
        mock_requests_get.return_value = mock_response

        mock_file = Mock()
        mock_file.name = "/tmp/fake_temp_file.jpg"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_file.close = Mock()
        mock_tempfile.return_value = mock_file

        with TempMedia(test_url, source_type="auto") as local_path:
            assert local_path == "/tmp/fake_temp_file.jpg"

    def test_default_source_type_is_auto(self, tmp_path):
        """Test that default source_type is 'auto' (backward compatibility)."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")

        # No source_type specified, should default to "auto"
        with TempMedia(str(test_file)) as local_path:
            assert os.path.exists(local_path)


# Try importing Flask and server components
try:
    from flask import Flask
    from fastvlm_server import app
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False
    IMPORT_ERROR = "Flask or server components not available"


class TestAPIMediaSourceType:
    """Test API endpoints with media_source_type flag."""

    @pytest.fixture
    def mock_engine(self):
        """Mock FastVLMEngine."""
        engine = MagicMock()
        engine.describe_image = MagicMock(return_value="Test description")
        engine.summarize_video = MagicMock(return_value=MagicMock(
            summary="Test summary",
            visual_tags=[],
            vibe_tags=[],
            safety_flags={},
            scenes_detected=0,
            frames_used=10,
            captions=[],
            timing={},
        ))
        return engine

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    def test_predict_image_with_local_source_type(self, tmp_path, mock_engine):
        """Test /predict_image endpoint with media_source_type='local'."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")

        with patch("fastvlm_server.engine", mock_engine):
            with patch("fastvlm_server.executor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = "Test description"
                mock_executor.submit.return_value = mock_future

                from fastvlm_server import app
                with app.test_client() as client:
                    response = client.post(
                        "/predict_image",
                        json={
                            "image_path": str(test_file),
                            "prompt": "Describe this image",
                            "media_source_type": "local",
                        },
                    )
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["output"] == "Test description"
                    assert data["image_path"] == str(test_file)

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    @patch("requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_predict_image_with_http_source_type(self, mock_tempfile, mock_requests_get, mock_engine):
        """Test /predict_image endpoint with media_source_type='http'."""
        test_url = "https://example.com/image.jpg"

        mock_response = Mock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake image data"])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_requests_get.return_value = mock_response

        mock_file = Mock()
        mock_file.name = "/tmp/fake_temp_file.jpg"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_file.close = Mock()
        mock_tempfile.return_value = mock_file

        with patch("fastvlm_server.engine", mock_engine):
            with patch("fastvlm_server.executor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = "Test description"
                mock_executor.submit.return_value = mock_future

                from fastvlm_server import app
                with app.test_client() as client:
                    response = client.post(
                        "/predict_image",
                        json={
                            "image_path": test_url,
                            "prompt": "Describe this image",
                            "media_source_type": "http",
                        },
                    )
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["output"] == "Test description"

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    def test_predict_image_with_invalid_source_type(self):
        """Test /predict_image endpoint with invalid media_source_type."""
        from fastvlm_server import app
        with app.test_client() as client:
            response = client.post(
                "/predict_image",
                json={
                    "image_path": "/path/to/image.jpg",
                    "prompt": "Describe this image",
                    "media_source_type": "invalid",
                },
            )
            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "Invalid 'media_source_type'" in data["error"]["message"]

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    def test_predict_image_without_source_type_defaults_to_auto(self, tmp_path, mock_engine):
        """Test /predict_image endpoint defaults to 'auto' when media_source_type not provided."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")

        with patch("fastvlm_server.engine", mock_engine):
            with patch("fastvlm_server.executor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = "Test description"
                mock_executor.submit.return_value = mock_future

                from fastvlm_server import app
                with app.test_client() as client:
                    response = client.post(
                        "/predict_image",
                        json={
                            "image_path": str(test_file),
                            "prompt": "Describe this image",
                            # media_source_type not provided, should default to "auto"
                        },
                    )
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["output"] == "Test description"

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    def test_summarize_video_with_local_source_type(self, tmp_path, mock_engine):
        """Test /summarize_video endpoint with media_source_type='local'."""
        test_file = tmp_path / "test_video.mp4"
        test_file.write_bytes(b"fake video data")

        with patch("fastvlm_server.engine", mock_engine):
            with patch("fastvlm_server.executor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = mock_engine.summarize_video.return_value
                mock_executor.submit.return_value = mock_future

                from fastvlm_server import app
                with app.test_client() as client:
                    response = client.post(
                        "/summarize_video",
                        json={
                            "video_path": str(test_file),
                            "media_source_type": "local",
                        },
                    )
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["summary"] == "Test summary"

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    @patch("requests.get")
    @patch("tempfile.NamedTemporaryFile")
    def test_summarize_video_with_http_source_type(self, mock_tempfile, mock_requests_get, mock_engine):
        """Test /summarize_video endpoint with media_source_type='http'."""
        test_url = "https://example.com/video.mp4"

        mock_response = Mock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake video data"])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_requests_get.return_value = mock_response

        mock_file = Mock()
        mock_file.name = "/tmp/fake_temp_file.mp4"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_file.close = Mock()
        mock_tempfile.return_value = mock_file

        with patch("fastvlm_server.engine", mock_engine):
            with patch("fastvlm_server.executor") as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = mock_engine.summarize_video.return_value
                mock_executor.submit.return_value = mock_future

                from fastvlm_server import app
                with app.test_client() as client:
                    response = client.post(
                        "/summarize_video",
                        json={
                            "video_path": test_url,
                            "media_source_type": "http",
                        },
                    )
                    assert response.status_code == 200
                    data = response.get_json()
                    assert data["summary"] == "Test summary"

    @pytest.mark.skipif(not SERVER_AVAILABLE, reason=f"Server not available: {IMPORT_ERROR}")
    def test_summarize_video_with_invalid_source_type(self):
        """Test /summarize_video endpoint with invalid media_source_type."""
        from fastvlm_server import app
        with app.test_client() as client:
            response = client.post(
                "/summarize_video",
                json={
                    "video_path": "/path/to/video.mp4",
                    "media_source_type": "invalid",
                },
            )
            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "Invalid 'media_source_type'" in data["error"]["message"]
