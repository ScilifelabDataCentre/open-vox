"""Input media validation.

Checks that an input file exists, is readable and non-empty, and that
ffprobe finds at least one audio stream in it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_TOOLS = ("ffmpeg", "ffprobe")
FFPROBE_TIMEOUT_S = 30.0
INSTALL_HINT = (
    "Install ffmpeg (e.g. 'sudo apt install ffmpeg' or 'brew install ffmpeg') "
    "and make sure it is on PATH."
)


class MediaError(Exception):
    """Base class for media-related errors."""


class FFmpegNotFoundError(MediaError):
    """ffmpeg or ffprobe is not available on PATH."""


class InputValidationError(MediaError):
    """The input file cannot be used for transcription."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe found in a validated input file."""

    path: Path
    format_name: str
    duration: float | None
    audio_streams: int
    video_streams: int

    @property
    def has_video(self) -> bool:
        return self.video_streams > 0


def check_ffmpeg() -> None:
    """Raise FFmpegNotFoundError unless both ffmpeg and ffprobe are on PATH."""
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise FFmpegNotFoundError(f"{' and '.join(missing)} not found on PATH. {INSTALL_HINT}")


def validate_input(path: str | os.PathLike[str]) -> MediaInfo:
    """Validate an input file for transcription and return its media info.

    Raises:
        InputValidationError: the file is missing, unreadable, empty,
            unsupported or corrupt, or has no audio stream.
        FFmpegNotFoundError: ffmpeg or ffprobe is not on PATH.
    """
    file = Path(path)
    _check_file(file)
    check_ffmpeg()
    return _media_info(file, _probe(file))


def _check_file(file: Path) -> None:
    if not file.exists():
        raise InputValidationError(f"File not found: {file}")
    if not file.is_file():
        raise InputValidationError(f"Not a regular file: {file}")
    if not os.access(file, os.R_OK):
        raise InputValidationError(f"File is not readable (check permissions): {file}")
    if file.stat().st_size == 0:
        raise InputValidationError(f"File is empty: {file}")


def _run_ffprobe(file: Path) -> subprocess.CompletedProcess[str]:
    # An absolute path keeps names starting with "-" from being read as options.
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=FFPROBE_TIMEOUT_S,
        check=False,
    )


def _probe(file: Path) -> dict[str, Any]:
    try:
        result = _run_ffprobe(file)
    except subprocess.TimeoutExpired as exc:
        raise InputValidationError(
            f"ffprobe timed out after {FFPROBE_TIMEOUT_S:.0f}s reading: {file}"
        ) from exc

    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1] if lines else "unknown error"
        raise InputValidationError(f"Unsupported or corrupt media file: {file} ({detail})")

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"Could not read ffprobe output for: {file}") from exc
    return data


def _media_info(file: Path, data: dict[str, Any]) -> MediaInfo:
    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})

    audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    # Cover art in audio files (e.g. MP3) shows up as a video stream; ignore it.
    video = sum(
        1
        for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    )

    if audio == 0:
        if video:
            raise InputValidationError(f"Video file has no audio stream: {file}")
        raise InputValidationError(f"No audio stream found in: {file}")

    return MediaInfo(
        path=file,
        format_name=str(fmt.get("format_name", "")),
        duration=_parse_duration(fmt.get("duration")),
        audio_streams=audio,
        video_streams=video,
    )


def _parse_duration(value: Any) -> float | None:
    try:
        return float(value)
    except TypeError, ValueError:
        return None
