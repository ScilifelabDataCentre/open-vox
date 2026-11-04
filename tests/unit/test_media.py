"""Unit tests for ovox.media. ffprobe and PATH lookups are mocked."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ovox import media
from ovox.media import (
    FFmpegNotFoundError,
    InputValidationError,
    MediaInfo,
    check_ffmpeg,
    validate_input,
)

# --- Helpers and fixtures -----------------------------------------------------

AUDIO_STREAM = {"codec_type": "audio", "codec_name": "pcm_s16le"}
VIDEO_STREAM = {"codec_type": "video", "codec_name": "h264", "disposition": {"attached_pic": 0}}
COVER_ART = {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}


def ffprobe_output(
    streams: list[dict[str, Any]], format_name: str = "wav", duration: str | None = "1.5"
) -> str:
    fmt: dict[str, Any] = {"format_name": format_name}
    if duration is not None:
        fmt["duration"] = duration
    return json.dumps({"streams": streams, "format": fmt})


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    """A non-empty file; its content doesn't matter since ffprobe is mocked."""
    path = tmp_path / "input.wav"
    path.write_bytes(b"fake media")
    return path


@pytest.fixture
def ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: f"/usr/bin/{tool}")


@pytest.fixture
def fake_ffprobe(monkeypatch: pytest.MonkeyPatch, ffmpeg_on_path: None) -> FakeFFprobe:
    fake = FakeFFprobe()
    monkeypatch.setattr(media, "_run_ffprobe", fake)
    return fake


class FakeFFprobe:
    """Stands in for media._run_ffprobe and records the files it was called with."""

    def __init__(self) -> None:
        self.stdout = ffprobe_output([AUDIO_STREAM])
        self.stderr = ""
        self.returncode = 0
        self.raises: Exception | None = None
        self.calls: list[Path] = []

    def __call__(self, file: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append(file)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            args=["ffprobe"], returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


# --- check_ffmpeg -------------------------------------------------------------


def test_check_ffmpeg_passes_when_both_tools_found(ffmpeg_on_path: None) -> None:
    check_ffmpeg()


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        pytest.param({"ffprobe"}, "ffmpeg not found", id="ffmpeg-missing"),
        pytest.param({"ffmpeg"}, "ffprobe not found", id="ffprobe-missing"),
        pytest.param(set(), "ffmpeg and ffprobe not found", id="both-missing"),
    ],
)
def test_check_ffmpeg_reports_missing_tools(
    monkeypatch: pytest.MonkeyPatch, available: set[str], expected: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: tool if tool in available else None)

    with pytest.raises(FFmpegNotFoundError, match=expected) as exc_info:
        check_ffmpeg()

    assert "Install ffmpeg" in str(exc_info.value)


# --- File checks (before ffprobe runs) ----------------------------------------


def test_missing_file(tmp_path: Path, fake_ffprobe: FakeFFprobe) -> None:
    with pytest.raises(InputValidationError, match="File not found"):
        validate_input(tmp_path / "missing.wav")
    assert fake_ffprobe.calls == []


def test_directory_is_rejected(tmp_path: Path, fake_ffprobe: FakeFFprobe) -> None:
    with pytest.raises(InputValidationError, match="Not a regular file"):
        validate_input(tmp_path)


def test_unreadable_file(
    media_file: Path, monkeypatch: pytest.MonkeyPatch, fake_ffprobe: FakeFFprobe
) -> None:
    monkeypatch.setattr(os, "access", lambda path, mode: False)

    with pytest.raises(InputValidationError, match="not readable"):
        validate_input(media_file)


def test_empty_file(tmp_path: Path, fake_ffprobe: FakeFFprobe) -> None:
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(InputValidationError, match="File is empty"):
        validate_input(empty)
    assert fake_ffprobe.calls == []


def test_file_errors_reported_before_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: None)

    with pytest.raises(InputValidationError, match="File not found"):
        validate_input(tmp_path / "missing.wav")


def test_missing_ffmpeg_raises_before_probing(
    media_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: None)

    with pytest.raises(FFmpegNotFoundError):
        validate_input(media_file)


# --- ffprobe results ----------------------------------------------------------


def test_valid_audio_file(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    info = validate_input(media_file)

    assert info == MediaInfo(
        path=media_file, format_name="wav", duration=1.5, audio_streams=1, video_streams=0
    )
    assert info.has_video is False
    assert fake_ffprobe.calls == [media_file]


def test_accepts_str_path(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    assert validate_input(str(media_file)).path == media_file


def test_valid_video_with_audio(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.stdout = ffprobe_output([VIDEO_STREAM, AUDIO_STREAM], "mov,mp4,m4a,3gp,3g2,mj2")

    info = validate_input(media_file)

    assert info.audio_streams == 1
    assert info.has_video is True


def test_cover_art_is_not_counted_as_video(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.stdout = ffprobe_output([AUDIO_STREAM, COVER_ART], "mp3")

    assert validate_input(media_file).has_video is False


def test_missing_duration_is_none(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.stdout = ffprobe_output([AUDIO_STREAM], duration=None)

    assert validate_input(media_file).duration is None


def test_video_without_audio(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.stdout = ffprobe_output([VIDEO_STREAM], "mov,mp4,m4a,3gp,3g2,mj2")

    with pytest.raises(InputValidationError, match="Video file has no audio stream"):
        validate_input(media_file)


@pytest.mark.parametrize(
    "streams",
    [pytest.param([], id="no-streams"), pytest.param([COVER_ART], id="cover-art-only")],
)
def test_no_audio_stream(
    media_file: Path, fake_ffprobe: FakeFFprobe, streams: list[dict[str, Any]]
) -> None:
    fake_ffprobe.stdout = ffprobe_output(streams)

    with pytest.raises(InputValidationError, match="No audio stream found"):
        validate_input(media_file)


def test_ffprobe_failure_is_reported_with_detail(
    media_file: Path, fake_ffprobe: FakeFFprobe
) -> None:
    fake_ffprobe.returncode = 1
    fake_ffprobe.stdout = ""
    fake_ffprobe.stderr = f"{media_file}: Invalid data found when processing input\n"

    with pytest.raises(InputValidationError, match="Unsupported or corrupt") as exc_info:
        validate_input(media_file)

    assert "Invalid data found" in str(exc_info.value)


def test_ffprobe_failure_without_stderr(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.returncode = 1
    fake_ffprobe.stdout = ""

    with pytest.raises(InputValidationError, match="unknown error"):
        validate_input(media_file)


def test_ffprobe_timeout(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    with pytest.raises(InputValidationError, match="timed out"):
        validate_input(media_file)


def test_unparsable_ffprobe_output(media_file: Path, fake_ffprobe: FakeFFprobe) -> None:
    fake_ffprobe.stdout = "not json"

    with pytest.raises(InputValidationError, match="Could not read ffprobe output"):
        validate_input(media_file)
