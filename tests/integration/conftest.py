# tests/integration/conftest.py
import shutil
import subprocess
from pathlib import Path

import pytest

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    pytest.skip("ffmpeg/ffprobe not on PATH", allow_module_level=True)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate small media files once per test session."""
    d = tmp_path_factory.mktemp("media")
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(d / "tone.wav"),
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=64x64:rate=5",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=1",
        "-shortest",
        str(d / "video_with_audio.mp4"),
    )
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=64x64:rate=5",
        "-an",
        str(d / "video_no_audio.mp4"),
    )
    (d / "empty.wav").touch()
    (d / "corrupt.wav").write_bytes(b"RIFF" + bytes(range(256)) * 8)  # WAV header, garbage body
    (d / "notes.txt").write_text("not media")
    return d
