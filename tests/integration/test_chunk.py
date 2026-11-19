"""Integration tests for the ovox chunk command: real ffmpeg, real files.

The central property under test is that the preview (``--dry-run``) and the
real run share one code path, so the plan a dry run prints describes exactly the
files a real run writes: same number of chunks, same names, and matching
durations. A dry run writes nothing; ``--plan-only`` writes just the manifest.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ovox.cli import main

MANIFEST = "ovox-chunk-manifest.json"


def _run_chunk(file: Path, out_dir: Path, *extra: str) -> int:
    return main(["chunk", str(file), "--output-dir", str(out_dir), *extra])


def _load_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / MANIFEST).read_text())


def _dry_run_json(file: Path, out_dir: Path, capsys, *extra: str) -> tuple[int, dict]:
    """Run a ``--dry-run --json`` and return (exit code, parsed plan).

    A dry run writes nothing, so the plan is read from stdout, not a manifest.
    """
    rc = main(["chunk", str(file), "--output-dir", str(out_dir), "--dry-run", "--json", *extra])
    return rc, json.loads(capsys.readouterr().out)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def test_dry_run_writes_nothing(media_dir: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "dry"
    rc, data = _dry_run_json(media_dir / "speech_like.wav", out, capsys, "--max-duration-s", "20")

    assert rc == 0
    # A dry run is safe by convention: it creates no output directory and no files.
    assert not out.exists()
    # The printed plan still carries no real sizes, since nothing was cut.
    assert all(c["actual_size"] is None for c in data["chunks"])
    assert data["totals"]["num_chunks"] >= 2


def test_real_run_writes_chunk_files(media_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "real"
    rc = _run_chunk(media_dir / "speech_like.wav", out, "--max-duration-s", "20")

    assert rc == 0
    data = _load_manifest(out)
    names = sorted(p.name for p in out.iterdir())
    expected = sorted([MANIFEST] + [c["name"] for c in data["chunks"]])
    assert names == expected
    # Every listed chunk file exists, is non-empty, and its real size is recorded.
    for c in data["chunks"]:
        f = out / c["name"]
        assert f.exists() and f.stat().st_size > 0
        assert c["actual_size"] == f.stat().st_size


def test_plan_only_writes_manifest_no_chunks(media_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = _run_chunk(media_dir / "speech_like.wav", out, "--plan-only", "--max-duration-s", "20")

    assert rc == 0
    # The manifest is written, but no chunk files.
    assert sorted(p.name for p in out.iterdir()) == [MANIFEST]
    data = _load_manifest(out)
    assert data["totals"]["num_chunks"] >= 2
    assert all(c["actual_size"] is None for c in data["chunks"])


def test_dry_run_and_real_run_agree(media_dir: Path, tmp_path: Path, capsys) -> None:
    """The preview and the real run must plan identical chunks (same code)."""
    real = tmp_path / "real"
    args = ["--max-duration-s", "20"]

    _, d = _dry_run_json(media_dir / "speech_like.wav", tmp_path / "dry", capsys, *args)
    assert _run_chunk(media_dir / "speech_like.wav", real, *args) == 0
    r = _load_manifest(real)

    assert d["totals"]["num_chunks"] == r["totals"]["num_chunks"]
    assert len(d["chunks"]) == len(r["chunks"]) >= 2
    for dc, rc in zip(d["chunks"], r["chunks"], strict=True):
        # Same plan, so identical to the second decimal.
        assert dc["name"] == rc["name"]
        assert dc["start"] == pytest.approx(rc["start"], abs=1e-6)
        assert dc["end"] == pytest.approx(rc["end"], abs=1e-6)
        assert dc["duration"] == pytest.approx(rc["duration"], abs=1e-6)
        assert dc["reason"] == rc["reason"]

    # The written files' real durations match the plan within MP3 tolerance.
    for c in r["chunks"]:
        measured = _probe_duration(real / c["name"])
        assert measured == pytest.approx(c["duration"], abs=0.15)


def test_chunks_cover_the_input(media_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run_chunk(media_dir / "speech_like.wav", out, "--max-duration-s", "20")
    data = _load_manifest(out)

    assert data["chunks"][0]["start"] == pytest.approx(0.0, abs=0.05)
    # speech_like.wav is 54 s; the last chunk ends at the end of the audio.
    assert data["chunks"][-1]["end"] == pytest.approx(54.0, abs=0.3)
    # No chunk exceeds the (safety-scaled) maximum duration.
    assert all(c["duration"] <= 20 * 0.8 + 0.2 for c in data["chunks"])


def test_video_produces_audio_chunks(media_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    rc = _run_chunk(media_dir / "video_long.mp4", out, "--max-duration-s", "6")

    assert rc == 0
    data = _load_manifest(out)
    # A 24 s video at a 6 s cap (4.8 s with safety) needs several chunks.
    assert data["totals"]["num_chunks"] >= 4
    # The input is reported as a video with an audio track.
    assert data["input"]["format"]
    # Each chunk is a real audio file (MP3) with only an audio stream.
    for c in data["chunks"]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=nw=1:nk=1",
                str(out / c["name"]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.split() == ["audio"]


def test_single_chunk_when_within_limits(media_dir: Path, tmp_path: Path) -> None:
    # A 1 s tone is far under the default limits: one chunk, no split.
    out = tmp_path / "out"
    rc = _run_chunk(media_dir / "tone.wav", out)

    assert rc == 0
    data = _load_manifest(out)
    assert data["totals"]["num_chunks"] == 1
    assert any("No split was needed" in w for w in data["warnings"])
    # The single chunk still gets written.
    assert (out / data["chunks"][0]["name"]).exists()


def test_size_limit_respected(media_dir: Path, tmp_path: Path) -> None:
    # A small size limit forces more, shorter chunks than the duration limit alone.
    out = tmp_path / "out"
    _run_chunk(media_dir / "speech_like.wav", out, "--max-size-bytes", "64000")
    data = _load_manifest(out)

    # 64 KB at 64 kbps is ~8 s, below the 600 s duration limit, so the size
    # limit is what drives the split. No chunk's estimated size may exceed it.
    assert all(c["estimated_size"] <= 64000 for c in data["chunks"])
    assert data["totals"]["num_chunks"] >= 3


def test_output_dir_is_created(media_dir: Path, tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    rc = _run_chunk(media_dir / "tone.wav", nested, "--plan-only")
    assert rc == 0
    assert nested.is_dir()
    assert (nested / MANIFEST).exists()


def test_dry_run_creates_no_output_dir(media_dir: Path, tmp_path: Path, capsys) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    rc, _ = _dry_run_json(media_dir / "tone.wav", nested, capsys)
    assert rc == 0
    # Unlike a real run or --plan-only, a dry run must not create the directory.
    assert not nested.exists()
