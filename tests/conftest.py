# tests/conftest.py
import pytest


@pytest.fixture
def stt_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set dummy STT settings in the environment for one test."""
    env = {
        "STT_ENDPOINT": "https://api.example.com/v1/audio/transcriptions",
        "STT_API_KEY": "test-key",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env
