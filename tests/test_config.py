"""Tests for ovox configuration."""

from unittest.mock import patch

import pytest

from ovox.config import Config, _get_env_var, _get_optional_env_var


def test_get_env_var_found():
    """Test getting an environment variable that exists."""
    with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
        assert _get_env_var("TEST_VAR") == "test_value"


def test_get_env_var_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting an environment variable that doesn't exist."""
    monkeypatch.delenv("TEST_VAR", raising=False)

    with pytest.raises(ValueError, match="Required environment variable TEST_VAR not set"):
        _get_env_var("TEST_VAR")


def test_get_env_var_with_default():
    """Test getting an environment variable with a default value."""
    with patch.dict("os.environ", {}):
        assert _get_env_var("TEST_VAR", "default_value") == "default_value"


def test_get_optional_env_var_found():
    """Test getting an optional environment variable that exists."""
    with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
        assert _get_optional_env_var("TEST_VAR") == "test_value"


def test_get_optional_env_var_not_found():
    """Test getting an optional environment variable that doesn't exist."""
    with patch.dict("os.environ", {}):
        assert _get_optional_env_var("TEST_VAR") is None


def test_get_optional_env_var_with_default():
    """Test getting an optional environment variable with a default value."""
    with patch.dict("os.environ", {}):
        assert _get_optional_env_var("TEST_VAR", "default_value") == "default_value"


def test_config_from_env():
    """Test creating a Config instance from environment variables."""
    env_vars = {
        "STT_ENDPOINT": "https://api.example.com/v1/audio/transcriptions",
        "STT_API_KEY": "sk-1234567890",
        "STT_MODEL": "whisper-1",
        "AUDIO_SAMPLE_RATE": "22050",
        "AUDIO_CHANNELS": "2",
    }

    with patch.dict("os.environ", env_vars):
        config = Config.from_env()

        assert config.stt_endpoint == "https://api.example.com/v1/audio/transcriptions"
        assert config.stt_api_key == "sk-1234567890"
        assert config.stt_model == "whisper-1"
        assert config.audio_sample_rate == 22050
        assert config.audio_channels == 2


def test_config_from_env_uses_defaults(
    stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("STT_MODEL", "AUDIO_SAMPLE_RATE", "AUDIO_CHANNELS"):
        monkeypatch.delenv(var, raising=False)

    config = Config.from_env()

    assert config.stt_endpoint == stt_env["STT_ENDPOINT"]
    assert config.stt_api_key == stt_env["STT_API_KEY"]
    assert config.stt_model == "whisper-1"
    assert config.audio_sample_rate == 16000
    assert config.audio_channels == 1
