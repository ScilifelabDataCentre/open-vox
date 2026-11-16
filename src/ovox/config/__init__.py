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


@dataclass(frozen=True)
class SttConfig:
    """Settings for the STT endpoint."""

    endpoint: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> SttConfig:
        """Create an SttConfig instance from environment variables."""
        return cls(
            endpoint=_get_env_var("STT_ENDPOINT"),
            api_key=_get_env_var("STT_API_KEY"),
            model=_get_optional_env_var("STT_MODEL", "voxtral-1") or "voxtral-1",
        )


@dataclass
class Config:
    """Configuration for ovox."""

    stt: SttConfig

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config instance from environment variables."""
        return cls(stt=SttConfig.from_env())
