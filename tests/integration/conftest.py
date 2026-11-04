# tests/integration/conftest.py
"""Fixtures for integration tests: small media files generated with ffmpeg.

Without ffmpeg/ffprobe on PATH, every test in this folder is skipped, or the
run is stopped if OVOX_REQUIRE_FFMPEG is set (as in CI).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
MISSING_MSG = "ffmpeg/ffprobe not on PATH"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if FFMPEG_AVAILABLE:
        return
    # This hook receives every collected test, not only this folder's.
    integration_items = [item for item in items if item.path.is_relative_to(HERE)]
    if not integration_items:
        return
    if os.environ.get("OVOX_REQUIRE_FFMPEG"):
        pytest.exit(f"{MISSING_MSG}, and OVOX_REQUIRE_FFMPEG is set", returncode=1)
    skip = pytest.mark.skip(reason=MISSING_MSG)
    for item in integration_items:
        item.add_marker(skip)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate small media files once per test session."""
    d = tmp_path_factory.mktemp("media")
    tone = "sine=frequency=440:duration=1"
    video = "testsrc=duration=1:size=64x64:rate=5"

    _ffmpeg("-f", "lavfi", "-i", tone, "-ar", "16000", "-ac", "1", str(d / "tone.wav"))
    _ffmpeg("-f", "lavfi", "-i", tone, str(d / "tone.mp3"))
    _ffmpeg(
        "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", tone, "-shortest",
        str(d / "video_with_audio.mp4"),
    )  # fmt: skip
    _ffmpeg("-f", "lavfi", "-i", video, "-an", str(d / "video_no_audio.mp4"))

    (d / "empty.wav").touch()
    (d / "corrupt.wav").write_bytes(b"RIFF" + bytes(range(256)) * 8)  # WAV magic, garbage body
    (d / "notes.txt").write_text("not media\n")
    return d
