"""Unit tests for ovox.audio. ffmpeg is mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ovox import audio
from ovox.audio import (
    AudioExtractionError,
    cut_audio,
    decoded_duration,
    extract_audio,
    workdir,
)


class FakeFFmpeg:
    """Stands in for audio.run_ffmpeg and records the commands it was called with."""

    def __init__(self) -> None:
        self.returncode = 0
        self.stderr = ""
        self.raises: Exception | None = None
        self.writes_output = True
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, timeout))
        if self.raises is not None:
            raise self.raises
        if self.writes_output:
            Path(command[-1]).write_bytes(b"fake mp3")
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr=self.stderr)


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFFmpeg:
    fake = FakeFFmpeg()
    monkeypatch.setattr(audio, "run_ffmpeg", fake)
    return fake


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake media")
    return source


# --- workdir ------------------------------------------------------------------


def test_workdir_is_cleaned_up() -> None:
    with workdir() as d:
        assert d.is_dir()
        (d / "file.txt").write_text("x")

    assert not d.exists()


def test_workdir_is_cleaned_up_on_error() -> None:
    with pytest.raises(RuntimeError, match="boom"), workdir() as d:
        raise RuntimeError("boom")

    assert not d.exists()


# --- extract_audio: success -----------------------------------------------------


def test_extract_audio_command(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    with workdir() as d:
        out = extract_audio(source_file, d)
        assert out == d / "audio.mp3"
        assert out.read_bytes() == b"fake mp3"

    command, _ = fake_ffmpeg.calls[0]
    # Defaults are mono, 16 kHz, 64 kbps CBR MP3 (the built-in upload format).
    assert command == [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_file.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(d / "audio.mp3"),
    ]


def test_extract_audio_uses_configured_format(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    # A non-default upload format changes -ac, -ar and -b:a accordingly.
    with workdir() as d:
        extract_audio(source_file, d, sample_rate=48000, channels=2, bitrate_bps=128_000)

    command, _ = fake_ffmpeg.calls[0]
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-b:a") + 1] == "128k"


@pytest.mark.parametrize(
    ("duration", "expected_timeout"),
    [
        pytest.param(None, 600.0, id="unknown-duration"),
        pytest.param(1.0, 120.0, id="short-file-hits-floor"),
        pytest.param(60.0, 180.0, id="scales-with-duration"),
    ],
)
def test_extract_audio_timeout(
    fake_ffmpeg: FakeFFmpeg, source_file: Path, duration: float | None, expected_timeout: float
) -> None:
    with workdir() as d:
        extract_audio(source_file, d, duration=duration)

    assert fake_ffmpeg.calls[0][1] == expected_timeout


# --- extract_audio: failures ----------------------------------------------------


def test_extract_audio_failure_includes_stderr_tail(
    fake_ffmpeg: FakeFFmpeg, source_file: Path
) -> None:
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.stderr = "line one\nline two\nline three\nline four\nline five\nline six"

    with workdir() as d, pytest.raises(AudioExtractionError) as excinfo:
        extract_audio(source_file, d)

    message = str(excinfo.value)
    assert "ffmpeg failed while extracting audio from" in message
    assert "line six" in message  # last lines are shown
    assert "line one" not in message  # but not the whole log


def test_extract_audio_failure_without_stderr(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.stderr = ""

    with workdir() as d, pytest.raises(AudioExtractionError, match="no error output"):
        extract_audio(source_file, d)


def test_extract_audio_timeout_is_an_error(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.raises = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

    with workdir() as d, pytest.raises(AudioExtractionError, match="timed out after 120s"):
        extract_audio(source_file, d, duration=30.0)


def test_extract_audio_empty_output_is_an_error(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.writes_output = False

    with workdir() as d, pytest.raises(AudioExtractionError, match="produced no audio"):
        extract_audio(source_file, d)


# --- cut_audio: success --------------------------------------------------------


def test_cut_audio_command(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    with workdir() as d:
        out = cut_audio(source_file, d / "chunk.mp3", 10.0, 25.0)
        assert out == d / "chunk.mp3"

    command, _ = fake_ffmpeg.calls[0]
    # Cuts [10, 25) to mono 16 kHz CBR MP3 at the built-in upload format. The
    # seek is an input seek (-ss before -i) and the cut is re-encoded (not
    # -c copy), so it is sample-accurate on every ffmpeg version.
    assert command == [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        "10.000",
        "-i",
        str(source_file.resolve()),
        "-t",
        "15.000",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(d / "chunk.mp3"),
    ]
    # Not a stream copy.
    assert "copy" not in command


def test_cut_audio_uses_configured_format(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    with workdir() as d:
        cut_audio(
            source_file,
            d / "chunk.mp3",
            0.0,
            30.0,
            sample_rate=48000,
            channels=2,
            bitrate_bps=128_000,
        )

    command, _ = fake_ffmpeg.calls[0]
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-b:a") + 1] == "128k"


def test_cut_audio_rejects_invalid_range(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    with workdir() as d, pytest.raises(AudioExtractionError, match="Invalid chunk range"):
        cut_audio(source_file, d / "chunk.mp3", 5.0, 5.0)


def test_cut_audio_failure_includes_stderr_tail(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.stderr = "one\ntwo\nthree\nfour\nfive\nsix"

    with workdir() as d, pytest.raises(AudioExtractionError) as excinfo:
        cut_audio(source_file, d / "chunk.mp3", 0.0, 10.0)

    message = str(excinfo.value)
    assert "ffmpeg failed while cutting" in message
    assert "six" in message


def test_cut_audio_timeout_is_an_error(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.raises = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

    with workdir() as d, pytest.raises(AudioExtractionError, match="timed out after 120s"):
        cut_audio(source_file, d / "chunk.mp3", 0.0, 30.0)


def test_cut_audio_empty_output_is_an_error(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.writes_output = False

    with workdir() as d, pytest.raises(AudioExtractionError, match="produced no audio"):
        cut_audio(source_file, d / "chunk.mp3", 0.0, 10.0)


# --- decoded_duration ----------------------------------------------------------


def test_decoded_duration_decodes_to_pcm(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    # The fake writes a known byte count to the PCM output; decoded_duration
    # converts it (2 bytes per s16le sample) back to seconds at the rate.
    # b"fake mp3" is 8 bytes -> 4 s16le samples -> 4/16000 s at the default rate.
    assert decoded_duration(source_file) == pytest.approx(8 // 2 / 16000)
    command, _ = fake_ffmpeg.calls[0]
    # Decodes to mono 16 kHz s16le so the sample count maps to seconds.
    assert command[command.index("-f") + 1] == "s16le"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"


def test_decoded_duration_uses_configured_rate(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    # A different decode rate changes the seconds-per-sample conversion.
    assert decoded_duration(source_file, sample_rate=48000) == pytest.approx(8 // 2 / 48000)

    command, _ = fake_ffmpeg.calls[0]
    assert command[command.index("-ar") + 1] == "48000"


def test_decoded_duration_failure_is_an_error(fake_ffmpeg: FakeFFmpeg, source_file: Path) -> None:
    fake_ffmpeg.returncode = 1

    with pytest.raises(AudioExtractionError, match="measuring the duration"):
        decoded_duration(source_file)
