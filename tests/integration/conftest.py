# tests/integration/conftest.py
"""Fixtures for integration tests: small media files generated with ffmpeg.

Without ffmpeg/ffprobe on PATH, every test in this folder is skipped, or the
run is stopped if OVOX_REQUIRE_FFMPEG is set (as in CI).
"""

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
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


Part = tuple[str, float]  # ("tone" | "silence", seconds)


def _media(out: Path, *parts: Part, video: bool = False) -> None:
    """Write *out* from audio parts and, if *video*, a test-pattern video track.

    Audio is stereo 44.1 kHz, so the extraction tests prove that ovox really
    downmixes and resamples. The container and codecs follow the file extension.
    With no parts, the file has no audio stream.
    """
    duration = sum(secs for _, secs in parts) or 1.0
    args: list[str] = []
    maps: list[str] = []
    if video:
        args += ["-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate=5"]
        maps += ["-map", "0:v"]
    if parts:
        first = 1 if video else 0  # input index of the first audio part
        for kind, secs in parts:
            src = "sine=frequency=440" if kind == "tone" else "aevalsrc=0"
            args += ["-f", "lavfi", "-i", f"{src}:duration={secs}"]
        labels = "".join(f"[{first + i}:a]" for i in range(len(parts)))
        args += ["-filter_complex", f"{labels}concat=n={len(parts)}:v=0:a=1[a]"]
        maps += ["-map", "[a]", "-ac", "2", "-ar", "44100"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args, *maps, str(out)], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate small media files once per test session."""
    d = tmp_path_factory.mktemp("media")
    tone: Part = ("tone", 1)

    _media(d / "tone.wav", tone)
    _media(d / "tone.mp3", tone)
    _media(d / "video_with_audio.mp4", tone, video=True)
    _media(d / "video_no_audio.mp4", video=True)
    _media(d / "tone_with_silence.wav", tone, ("silence", 1), tone)
    _media(d / "tone_with_trailing_silence.wav", ("tone", 2), ("silence", 2))
    _media(d / "tone_only.wav", ("tone", 3))
    # 54 s with a 1 s pause every 9 s: needs several chunks at a small max_len.
    _media(d / "speech_like.wav", *[("tone", 8), ("silence", 1)] * 6)

    (d / "empty.wav").touch()
    (d / "corrupt.wav").write_bytes(b"RIFF" + bytes(range(256)) * 8)  # WAV magic, garbage body
    (d / "notes.txt").write_text("not media\n")
    return d
