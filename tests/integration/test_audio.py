"""Integration tests for ovox.audio: real ffmpeg on real files."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ovox.audio import cut_audio, decoded_duration


def _probe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"]


@pytest.mark.parametrize("name", ["tone.wav", "tone.mp3"])
def test_cut_produces_mono_16k_cbr_mp3(media_dir: Path, name: str, tmp_path: Path) -> None:
    # 1 s of 44.1 kHz stereo source, cut [0, 1) at the built-in upload format.
    out = cut_audio(media_dir / name, tmp_path / "chunk.mp3", 0.0, 1.0)

    # 1 s of audio at 64 kbps is about 8 KB; leave generous headroom for
    # frame alignment.
    assert 0 < out.stat().st_size < 10_000

    streams = _probe_streams(out)
    assert len(streams) == 1
    stream = streams[0]
    assert stream["codec_type"] == "audio"
    assert stream["codec_name"] == "mp3"
    assert stream["channels"] == 1
    assert stream["sample_rate"] == "16000"
    # CBR at the configured bitrate (64 kbps, pinned by the deployment_env
    # fixture), which is what makes the chunk-size estimates predictable.
    assert int(stream["bit_rate"]) == 64000


def test_cut_from_video_keeps_audio_only(media_dir: Path, tmp_path: Path) -> None:
    out = cut_audio(media_dir / "video_with_audio.mp4", tmp_path / "chunk.mp3", 0.0, 1.0)

    streams = _probe_streams(out)
    assert [s["codec_type"] for s in streams] == ["audio"]
    assert streams[0]["codec_name"] == "mp3"


def test_decoded_duration_counts_real_samples(media_dir: Path) -> None:
    # tone.wav is 1 s of audio; a 1 s slice of it decodes to ~1 s.
    assert decoded_duration(media_dir / "tone.wav") == pytest.approx(1.0, abs=0.1)


def test_cut_duration_matches_decode(media_dir: Path, tmp_path: Path) -> None:
    # The middle of tone_with_silence.wav (a tone) cuts to the requested length,
    # measured by decoding, not by the (unreliable) header duration.
    out = cut_audio(media_dir / "tone_with_silence.wav", tmp_path / "chunk.mp3", 0.0, 1.0)
    assert decoded_duration(out) == pytest.approx(1.0, abs=0.1)
