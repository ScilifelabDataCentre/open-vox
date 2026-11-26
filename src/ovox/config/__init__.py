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


def _get_env_int(name: str, default: int) -> int:
    """Get an integer environment variable, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def _get_env_float(name: str, default: float) -> float:
    """Get a float environment variable, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got: {raw!r}") from exc


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


# MP3 only allows certain constant bitrates, and which ones depend on the
# sample rate (see the libmp3lame bitrate table). A bitrate outside the table
# for a rate makes ffmpeg pick a *different* bitrate quietly, which would break
# the size calculations that assume a known CBR rate.
_MP3_BITRATES_BY_RATE_CLASS: dict[frozenset[int], list[int]] = {
    frozenset({8000, 11025, 12000, 16000, 22050, 24000}): [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
    ],
    frozenset({32000, 44100, 48000}): [
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
    ],
}


def _valid_bitrates_kbps(sample_rate: int) -> list[int] | None:
    """Allowed MP3 bitrates (in kbps) for *sample_rate*, or None if the rate is
    not a supported MP3 rate at all."""
    for rates, bitrates in _MP3_BITRATES_BY_RATE_CLASS.items():
        if sample_rate in rates:
            return bitrates
    return None


@dataclass(frozen=True)
class AudioConfig:
    """Format of the MP3 the audio is converted to before upload.

    These are deployment settings (the "Audio processing" block in ``.env``):
    they describe the uploaded audio. The bitrate is stored in bits per second
    internally; the environment value is in kilobits per second
    (``AUDIO_BITRATE_KBPS``) and is converted here. Extraction
    (:mod:`ovox.audio`) and chunk planning (:mod:`ovox.chunking`) both read
    these values, so the encoded rate and the size estimates always agree.
    """

    sample_rate: int
    channels: int
    bitrate_bps: int

    @classmethod
    def from_env(cls) -> AudioConfig:
        """Create an AudioConfig from environment variables, with defaults.

        Raises:
            ValueError: a value is out of range, or the bitrate is not a valid
                MP3 bitrate for the sample rate.
        """
        # Defaults live in ovox.audio so the encoder and the config agree.
        from ovox.audio import (
            DEFAULT_BITRATE_BPS,
            DEFAULT_CHANNELS,
            DEFAULT_SAMPLE_RATE,
        )

        sample_rate = _get_env_int("AUDIO_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)
        channels = _get_env_int("AUDIO_CHANNELS", DEFAULT_CHANNELS)
        bitrate_kbps = _get_env_int("AUDIO_BITRATE_KBPS", DEFAULT_BITRATE_BPS // 1000)

        if sample_rate <= 0:
            raise ValueError(f"AUDIO_SAMPLE_RATE must be positive, got: {sample_rate}")
        if channels <= 0:
            raise ValueError(f"AUDIO_CHANNELS must be positive, got: {channels}")

        valid = _valid_bitrates_kbps(sample_rate)
        if valid is None:
            raise ValueError(
                f"AUDIO_SAMPLE_RATE {sample_rate} is not a supported MP3 sample rate. "
                "Use one of 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100 or 48000."
            )
        if bitrate_kbps not in valid:
            pretty = ", ".join(str(b) for b in valid)
            raise ValueError(
                f"AUDIO_BITRATE_KBPS {bitrate_kbps} is not a valid MP3 bitrate for "
                f"AUDIO_SAMPLE_RATE {sample_rate}. Valid values: {pretty} kbps."
            )
        return cls(
            sample_rate=sample_rate,
            channels=channels,
            bitrate_bps=bitrate_kbps * 1000,
        )


@dataclass
class ChunkConfig:
    """Per-target limits and options for splitting audio into chunks.

    These describe what the *target* STT endpoint accepts. The maximum file
    size is the primary limit now; a maximum upload duration can be added later
    as another input to the same maximum-length calculation (see
    ``ovox.chunking.max_chunk_seconds``) without changing how splitting works.
    """

    max_chunk_size_bytes: int
    max_chunk_duration_s: float
    safety: float

    @classmethod
    def from_env(cls) -> ChunkConfig:
        """Create a ChunkConfig from environment variables, with defaults.

        Raises:
            ValueError: a value is out of range.
        """
        # Defaults live in ovox.chunking so the planner and config agree.
        from ovox.chunking import (
            DEFAULT_MAX_CHUNK_SECONDS,
            DEFAULT_MAX_CHUNK_SIZE_BYTES,
            DEFAULT_SAFETY,
        )

        max_chunk_size_bytes = _get_env_int("CHUNK_MAX_SIZE_BYTES", DEFAULT_MAX_CHUNK_SIZE_BYTES)
        max_chunk_duration_s = _get_env_float("CHUNK_MAX_DURATION_S", DEFAULT_MAX_CHUNK_SECONDS)
        safety = _get_env_float("CHUNK_SAFETY", DEFAULT_SAFETY)

        if max_chunk_size_bytes <= 0:
            raise ValueError(f"CHUNK_MAX_SIZE_BYTES must be positive, got: {max_chunk_size_bytes}")
        if max_chunk_duration_s <= 0:
            raise ValueError(f"CHUNK_MAX_DURATION_S must be positive, got: {max_chunk_duration_s}")
        if not 0 < safety <= 1:
            raise ValueError(f"CHUNK_SAFETY must be in (0, 1], got: {safety}")
        return cls(
            max_chunk_size_bytes=max_chunk_size_bytes,
            max_chunk_duration_s=max_chunk_duration_s,
            safety=safety,
        )


# Defaults for TranscribeConfig, kept here so the config and the API client
# agree on what "the default" means.
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_CAP_S = 60.0


@dataclass(frozen=True)
class TranscribeConfig:
    """How ovox talks to the STT endpoint during transcription.

    The retry settings bound how ovox retries a failing chunk request (see
    ``ovox.api``): at most ``max_retries`` retries after the first attempt, with
    capped exponential backoff (full jitter) between attempts. The concurrency
    setting bounds how many chunks are sent in parallel.
    """

    max_concurrency: int
    max_retries: int
    backoff_base_s: float
    backoff_cap_s: float

    @classmethod
    def from_env(cls) -> TranscribeConfig:
        """Create a TranscribeConfig from environment variables, with defaults.

        Raises:
            ValueError: a value is out of range.
        """
        max_concurrency = _get_env_int("STT_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        max_retries = _get_env_int("STT_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        backoff_base_s = _get_env_float("STT_BACKOFF_BASE_S", DEFAULT_BACKOFF_BASE_S)
        backoff_cap_s = _get_env_float("STT_BACKOFF_CAP_S", DEFAULT_BACKOFF_CAP_S)

        if max_concurrency <= 0:
            raise ValueError(f"STT_MAX_CONCURRENCY must be positive, got: {max_concurrency}")
        if max_retries < 0:
            raise ValueError(f"STT_MAX_RETRIES must be >= 0, got: {max_retries}")
        if backoff_base_s <= 0:
            raise ValueError(f"STT_BACKOFF_BASE_S must be positive, got: {backoff_base_s}")
        if backoff_cap_s < backoff_base_s:
            raise ValueError(
                f"STT_BACKOFF_CAP_S must be >= STT_BACKOFF_BASE_S, "
                f"got: {backoff_cap_s} < {backoff_base_s}"
            )
        return cls(
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_cap_s=backoff_cap_s,
        )


@dataclass
class Config:
    """Configuration for ovox."""

    stt: SttConfig
    chunk: ChunkConfig
    audio: AudioConfig
    transcribe: TranscribeConfig

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config instance from environment variables."""
        return cls(
            stt=SttConfig.from_env(),
            chunk=ChunkConfig.from_env(),
            audio=AudioConfig.from_env(),
            transcribe=TranscribeConfig.from_env(),
        )
