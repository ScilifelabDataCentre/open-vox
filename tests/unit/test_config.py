"""Tests for ovox configuration."""

from unittest.mock import patch

import pytest

from ovox.config import (
    AudioConfig,
    ChunkConfig,
    Config,
    SttConfig,
    TranscribeConfig,
    _get_env_var,
    _get_optional_env_var,
)


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


# --- AudioConfig ---------------------------------------------------------------
#
# The repo's .env (loaded at import time) sets AUDIO_* and CHUNK_* values, so
# these tests clear them first and set exactly what they test.

_DEPLOYMENT_VARS = (
    "AUDIO_SAMPLE_RATE",
    "AUDIO_CHANNELS",
    "AUDIO_BITRATE_KBPS",
    "CHUNK_MAX_SIZE_BYTES",
    "CHUNK_MAX_DURATION_S",
    "CHUNK_SAFETY",
    "STT_MAX_CONCURRENCY",
    "STT_MAX_RETRIES",
    "STT_BACKOFF_BASE_S",
    "STT_BACKOFF_CAP_S",
    "STT_MAX_TIMEOUT_RETRIES",
)


@pytest.fixture
def clean_deployment_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DEPLOYMENT_VARS:
        monkeypatch.delenv(name, raising=False)


def test_audio_config_defaults(clean_deployment_env) -> None:
    # With no env set: mono, 16 kHz, 64 kbps (the built-in upload format).
    audio = AudioConfig.from_env()

    assert audio.sample_rate == 16000
    assert audio.channels == 1
    assert audio.bitrate_bps == 64_000  # kbps is converted to bps internally


def test_audio_config_reads_env(clean_deployment_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "44100")
    monkeypatch.setenv("AUDIO_CHANNELS", "2")
    monkeypatch.setenv("AUDIO_BITRATE_KBPS", "192")

    audio = AudioConfig.from_env()

    assert audio.sample_rate == 44100
    assert audio.channels == 2
    assert audio.bitrate_bps == 192_000


def test_audio_config_rejects_bitrate_invalid_for_rate(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 320 kbps is not offered at 16 kHz (it is at 44.1/48 kHz).
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "16000")
    monkeypatch.setenv("AUDIO_BITRATE_KBPS", "320")

    with pytest.raises(ValueError, match="not a valid MP3 bitrate"):
        AudioConfig.from_env()


def test_audio_config_rejects_unsupported_sample_rate(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "9999")

    with pytest.raises(ValueError, match="not a supported MP3 sample rate"):
        AudioConfig.from_env()


def test_audio_config_rejects_non_integer_bitrate(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIO_BITRATE_KBPS", "48.5")

    with pytest.raises(ValueError, match="must be an integer"):
        AudioConfig.from_env()


@pytest.mark.parametrize(
    ("sample_rate", "bitrate_kbps", "ok"),
    [
        (16000, 8, True),
        (16000, 160, True),
        (16000, 320, False),  # too high for the low-rate table
        (44100, 32, True),
        (48000, 320, True),
        (48000, 8, False),  # too low for the high-rate table
    ],
)
def test_audio_config_bitrate_table(
    clean_deployment_env,
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: int,
    bitrate_kbps: int,
    ok: bool,
) -> None:
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", str(sample_rate))
    monkeypatch.setenv("AUDIO_BITRATE_KBPS", str(bitrate_kbps))

    if ok:
        assert AudioConfig.from_env().bitrate_bps == bitrate_kbps * 1000
    else:
        with pytest.raises(ValueError, match="not a valid MP3 bitrate"):
            AudioConfig.from_env()


# --- ChunkConfig ---------------------------------------------------------------


def test_chunk_config_defaults(clean_deployment_env) -> None:
    from ovox.chunking import (
        DEFAULT_MAX_CHUNK_SECONDS,
        DEFAULT_MAX_CHUNK_SIZE_BYTES,
        DEFAULT_SAFETY,
    )

    config = ChunkConfig.from_env()

    assert config.max_chunk_size_bytes == DEFAULT_MAX_CHUNK_SIZE_BYTES
    assert config.max_chunk_duration_s == DEFAULT_MAX_CHUNK_SECONDS
    assert config.safety == DEFAULT_SAFETY


def test_chunk_config_reads_safety(clean_deployment_env, monkeypatch: pytest.MonkeyPatch) -> None:
    # CHUNK_SAFETY is read from the environment, not hard-coded.
    monkeypatch.setenv("CHUNK_SAFETY", "0.9")

    assert ChunkConfig.from_env().safety == pytest.approx(0.9)


@pytest.mark.parametrize("value", ["0", "1.5", "abc"])
def test_chunk_config_rejects_bad_safety(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CHUNK_SAFETY", value)

    if value == "abc":
        with pytest.raises(ValueError, match="must be a number"):
            ChunkConfig.from_env()
    else:
        with pytest.raises(ValueError, match="CHUNK_SAFETY must be in"):
            ChunkConfig.from_env()


def test_config_includes_audio(stt_env: dict[str, str], clean_deployment_env) -> None:
    config = Config.from_env()

    assert isinstance(config.audio, AudioConfig)
    assert config.audio.bitrate_bps == 64_000  # default with the deployment vars cleared


# --- TranscribeConfig ----------------------------------------------------------


def test_transcribe_config_defaults(clean_deployment_env) -> None:
    transcribe = TranscribeConfig.from_env()

    assert transcribe.max_concurrency == 4
    assert transcribe.max_retries == 5
    assert transcribe.backoff_base_s == 1.0
    assert transcribe.backoff_cap_s == 60.0
    # Timeouts get their own, smaller retry budget (see the module docs).
    assert transcribe.max_timeout_retries == 1


def test_transcribe_config_reads_env(clean_deployment_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("STT_MAX_RETRIES", "0")
    monkeypatch.setenv("STT_BACKOFF_BASE_S", "0.5")
    monkeypatch.setenv("STT_BACKOFF_CAP_S", "30")
    monkeypatch.setenv("STT_MAX_TIMEOUT_RETRIES", "3")

    transcribe = TranscribeConfig.from_env()

    assert transcribe.max_concurrency == 2
    assert transcribe.max_retries == 0
    assert transcribe.backoff_base_s == 0.5
    assert transcribe.backoff_cap_s == 30.0
    assert transcribe.max_timeout_retries == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("STT_MAX_CONCURRENCY", "0", id="zero-concurrency"),
        pytest.param("STT_MAX_CONCURRENCY", "-1", id="negative-concurrency"),
        pytest.param("STT_MAX_RETRIES", "-1", id="negative-retries"),
        pytest.param("STT_MAX_TIMEOUT_RETRIES", "-1", id="negative-timeout-retries"),
        pytest.param("STT_BACKOFF_BASE_S", "0", id="zero-base"),
    ],
)
def test_transcribe_config_rejects_out_of_range(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        TranscribeConfig.from_env()


def test_transcribe_config_rejects_cap_below_base(
    clean_deployment_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_BACKOFF_BASE_S", "10")
    monkeypatch.setenv("STT_BACKOFF_CAP_S", "5")

    with pytest.raises(ValueError, match="STT_BACKOFF_CAP_S"):
        TranscribeConfig.from_env()


def test_config_includes_transcribe(stt_env: dict[str, str], clean_deployment_env) -> None:
    config = Config.from_env()

    assert isinstance(config.transcribe, TranscribeConfig)
