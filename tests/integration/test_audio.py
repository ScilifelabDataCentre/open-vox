"""Integration tests for ovox.audio: real ffmpeg on real files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ovox.audio import extract_audio, workdir


def _probe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"]


@pytest.mark.parametrize("name", ["tone.wav", "tone.mp3"])
def test_extract_audio_produces_mono_16k_mp3(media_dir: Path, name: str) -> None:
    with workdir() as d:
        out = extract_audio(media_dir / name, d)

        assert out == d / "audio.mp3"
        # 1 s of audio at 32 kbps is about 4 KB; leave generous headroom.
        assert 0 < out.stat().st_size < 10_000

        streams = _probe_streams(out)
        assert len(streams) == 1
        stream = streams[0]
        assert stream["codec_type"] == "audio"
        assert stream["codec_name"] == "mp3"
        assert stream["channels"] == 1
        assert stream["sample_rate"] == "16000"


def test_extract_audio_from_video_keeps_audio_only(media_dir: Path) -> None:
    with workdir() as d:
        out = extract_audio(media_dir / "video_with_audio.mp4", d)

        streams = _probe_streams(out)
        assert [s["codec_type"] for s in streams] == ["audio"]
        assert streams[0]["codec_name"] == "mp3"


def test_workdir_is_cleaned_up() -> None:
    with workdir() as d:
        (d / "file.txt").write_text("x")

    assert not d.exists()
