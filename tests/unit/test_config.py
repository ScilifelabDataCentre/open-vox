"""Tests for ovox configuration."""

from unittest.mock import patch

import pytest

from ovox.config import Config, SttConfig, _get_env_var, _get_optional_env_var


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
        "STT_MODEL": "voxtral-1",
    }

    with patch.dict("os.environ", env_vars):
        config = Config.from_env()

        assert config.stt.endpoint == "https://api.example.com/v1/audio/transcriptions"
        assert config.stt.api_key == "sk-1234567890"
        assert config.stt.model == "voxtral-1"


def test_config_from_env_uses_defaults(
    stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STT_MODEL", raising=False)

    config = Config.from_env()

    assert config.stt.endpoint == stt_env["STT_ENDPOINT"]
    assert config.stt.api_key == stt_env["STT_API_KEY"]
    assert config.stt.model == "voxtral-1"


def test_stt_config_from_env(stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STT_MODEL", raising=False)

    stt = SttConfig.from_env()

    assert stt.endpoint == stt_env["STT_ENDPOINT"]
    assert stt.api_key == stt_env["STT_API_KEY"]
    assert stt.model == "voxtral-1"


def test_stt_config_from_env_missing_required(
    stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STT_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="STT_ENDPOINT"):
        SttConfig.from_env()
