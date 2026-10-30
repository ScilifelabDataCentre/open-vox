"""Configuration management for ovox."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env_var(name: str, default: str | None = None) -> str:
    """Get an environment variable, raising an error if not found (unless default provided)."""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Required environment variable {name} not set")
    return value


def _get_optional_env_var(name: str, default: str | None = None) -> str | None:
    """Get an optional environment variable."""
    return os.getenv(name, default)


@dataclass
class Config:
    """Configuration for ovox."""

    # STT API configuration
    stt_endpoint: str = ""
    stt_api_key: str = ""
    stt_model: str = "whisper-1"

    # Audio processing configuration
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config instance from environment variables."""
        return cls(
            stt_endpoint=_get_env_var("STT_ENDPOINT"),
            stt_api_key=_get_env_var("STT_API_KEY"),
            stt_model=_get_optional_env_var("STT_MODEL", "whisper-1"),
            audio_sample_rate=int(_get_optional_env_var("AUDIO_SAMPLE_RATE", "16000")),
            audio_channels=int(_get_optional_env_var("AUDIO_CHANNELS", "1")),
        )
