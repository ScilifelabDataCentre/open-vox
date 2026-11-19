"""Unit tests for the ovox chunk command. Planning is mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import ovox.plan as plan_mod
from ovox.cli import build_parser, main
from ovox.media import MediaInfo
from ovox.plan import (
    ChunkEntry,
    ChunkPlan,
    Setting,
)


def fake_info(tmp_path: Path) -> MediaInfo:
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fake")
    return MediaInfo(
        path=path,
        format_name="mp4",
        duration=10.0,
        audio_streams=1,
        video_streams=1,
        audio_codec="aac",
        audio_sample_rate=44100,
        audio_channels=2,
        audio_track_index=0,
    )


def make_plan(tmp_path: Path) -> ChunkPlan:
    """A real ChunkPlan for a single-chunk 10 s input (no split needed)."""
    entries = [
        ChunkEntry(
            index=0,
            start=0.0,
            end=10.0,
            duration=10.0,
            estimated_size=40000,
            name="meeting_chunk_00_0-10.mp3",
            reason="end of audio",
        )
    ]
    settings = {
        "max_chunk_size_bytes": Setting(25 * 1024 * 1024, "default", "CHUNK_MAX_SIZE_BYTES"),
        "max_chunk_duration_s": Setting(600.0, "default", "CHUNK_MAX_DURATION_S"),
        "overlap_s": Setting(1.0, "default", "CHUNK_OVERLAP_S"),
        "search_window_s": Setting(30.0, "default", "CHUNK_SEARCH_WINDOW_S"),
        "safety": Setting(0.8, "default", "CHUNK_SAFETY"),
        "split_method": Setting("silence", "default"),
        "sample_rate": Setting(16000, "default", "AUDIO_SAMPLE_RATE"),
        "channels": Setting(1, "default", "AUDIO_CHANNELS"),
        "bitrate_kbps": Setting(64, "default", "AUDIO_BITRATE_KBPS"),
    }
    return ChunkPlan(
        input={
            "file": str(tmp_path / "meeting.mp4"),
            "format": "mp4",
            "codec": "aac",
            "duration": 10.0,
            "sample_rate": 44100,
            "channels": 2,
            "audio_track": 0,
        },
        settings=settings,
        chunks=entries,
        warnings=["No split was needed; the input fits the limits in a single chunk."],
        totals={"num_chunks": 1, "total_audio_length": 10.0, "total_estimated_size": 40000},
        split_method="silence",
    )


def _fake_write_chunks(plan: ChunkPlan, output_dir: Path, *audio_config: object) -> list[Path]:
    """Stand in for write_chunks: create the files and record their sizes."""
    paths: list[Path] = []
    for entry in plan.chunks:
        out = Path(output_dir) / entry.name
        out.write_bytes(b"fake chunk")
        entry.actual_size = out.stat().st_size
        paths.append(out)
    return paths


@pytest.fixture
def plan_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChunkPlan:
    plan = make_plan(tmp_path)
    monkeypatch.setattr(plan_mod, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(plan_mod, "write_chunks", _fake_write_chunks)
    # validate_input is the only un-mocked real call in _cmd_chunk; stub it.
    monkeypatch.setattr("ovox.media.validate_input", lambda path: fake_info(tmp_path))
    return plan


def test_chunk_command_in_parser() -> None:
    parser = build_parser()
    assert "chunk" in parser._subparsers._group_actions[0].choices


def test_chunk_dry_run_writes_nothing(tmp_path: Path, capsys, plan_stub) -> None:
    out = tmp_path / "out"
    rc = main(["chunk", str(tmp_path / "meeting.mp4"), "--output-dir", str(out), "--dry-run"])

    assert rc == 0
    # A dry run is safe by convention: it creates no output directory at all,
    # so there is no manifest and no chunks to look for.
    assert not out.exists()


def test_chunk_plan_only_writes_manifest_no_chunks(tmp_path: Path, capsys, plan_stub) -> None:
    out = tmp_path / "out"
    rc = main(["chunk", str(tmp_path / "meeting.mp4"), "--output-dir", str(out), "--plan-only"])

    assert rc == 0
    # The manifest is written, but no chunk files.
    assert [p.name for p in out.iterdir()] == ["ovox-chunk-manifest.json"]
    data = json.loads((out / "ovox-chunk-manifest.json").read_text())
    assert data["totals"]["num_chunks"] == 1
    assert all(c["actual_size"] is None for c in data["chunks"])


def test_chunk_dry_run_and_plan_only_are_mutually_exclusive(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        main(
            [
                "chunk",
                str(tmp_path / "meeting.mp4"),
                "--output-dir",
                str(out),
                "--dry-run",
                "--plan-only",
            ]
        )


def test_chunk_real_run_writes_files(tmp_path: Path, capsys, plan_stub) -> None:
    out = tmp_path / "out"
    rc = main(["chunk", str(tmp_path / "meeting.mp4"), "--output-dir", str(out)])

    assert rc == 0
    data = json.loads((out / "ovox-chunk-manifest.json").read_text())
    assert all(c["actual_size"] is not None for c in data["chunks"])
    # The single planned chunk file was written (name from the original filename).
    assert (out / "meeting_chunk_00_0-10.mp3").exists()


def test_chunk_json_output(tmp_path: Path, capsys, plan_stub) -> None:
    out = tmp_path / "out"
    rc = main(
        ["chunk", str(tmp_path / "meeting.mp4"), "--output-dir", str(out), "--dry-run", "--json"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["input"]["file"].endswith("meeting.mp4")
    assert data["settings"]["safety"]["source"] == "default"
    # Nothing was written in a dry run, so no chunk has a real size.
    assert all(c["actual_size"] is None for c in data["chunks"])


def test_chunk_text_output_shows_fields(tmp_path: Path, capsys, plan_stub) -> None:
    out = tmp_path / "out"
    main(["chunk", str(tmp_path / "meeting.mp4"), "--output-dir", str(out), "--dry-run"])
    out_text = capsys.readouterr().out
    assert "meeting.mp4" in out_text
    assert "audio_track=0" in out_text
    assert "Dry run: nothing was written." in out_text
    # The single chunk row is shown.
    assert "meeting_chunk_00_0-10.mp3" in out_text


def test_chunk_missing_file_errors(tmp_path: Path, capsys) -> None:
    from ovox.media import InputValidationError

    out = tmp_path / "out"
    with patch(
        "ovox.media.validate_input",
        side_effect=InputValidationError("File not found: x"),
    ):
        rc = main(["chunk", str(tmp_path / "nope.mp4"), "--output-dir", str(out)])
    assert rc == 1
    assert "File not found" in capsys.readouterr().err
    assert not out.exists()


def test_chunk_requires_output_dir_unless_dry_run(
    tmp_path: Path, capsys, plan_stub: ChunkPlan
) -> None:
    """--output-dir is required for a real run but not for --dry-run."""
    file = tmp_path / "meeting.mp4"
    file.write_bytes(b"fake")

    # No --output-dir, no --dry-run: a clear error, non-zero exit.
    rc = main(["chunk", str(file)])
    assert rc == 1
    assert "--output-dir is required unless --dry-run" in capsys.readouterr().err

    # With --dry-run, --output-dir is not needed and nothing is created.
    out = tmp_path / "out"
    rc = main(["chunk", str(file), "--dry-run"])
    assert rc == 0
    assert not out.exists()


# --- byte-size suffix parsing --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("26214400", 26214400, id="plain-bytes"),
        pytest.param(" 4096 ", 4096, id="stripped"),
        pytest.param("1K", 1000, id="decimal-k"),
        pytest.param("25M", 25_000_000, id="decimal-m"),
        pytest.param("1G", 1_000_000_000, id="decimal-g"),
        pytest.param("25MiB", 26214400, id="binary-mib"),
        pytest.param("1KIB", 1024, id="binary-kib-case"),
        pytest.param("2MB", 2_000_000, id="decimal-mb"),
    ],
)
def test_parse_bytes_suffixes(raw: str, expected: int) -> None:
    from ovox.cli import _parse_bytes

    assert _parse_bytes(raw) == expected


@pytest.mark.parametrize("raw", ["", "12.5", "25x", "M", "1.5M", "25 MiB"])
def test_parse_bytes_rejects_bad_input(raw: str) -> None:
    import argparse

    from ovox.cli import _parse_bytes

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_bytes(raw)


def test_size_default_shows_bytes_and_human() -> None:
    from ovox.cli import _size_default

    assert _size_default(25 * 1024 * 1024) == "26214400 = 25 MiB"
    # Plain numbers without a clean human form keep just the number.
    assert _size_default(100) == "100"


# --- help text -----------------------------------------------------------------


def test_chunk_help_has_groups_and_flag_names(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["chunk", "--help"])
    help_text = capsys.readouterr().out

    # The option groups are present, in order.
    for group in ("output:", "limits:", "splitting:"):
        assert group in help_text
    # Short metavars, and the renamed safety flag.
    for token in (
        "--max-size-bytes BYTES",
        "--max-duration-s SECONDS",
        "--safety-factor FACTOR",
        "--split-method {silence,fixed}",
        "--search-window-s SECONDS",
        "--overlap-s SECONDS",
    ):
        assert token in help_text
    # The default is shown in bytes with its human equivalent.
    assert "26214400 = 25 MiB" in help_text
    # The old, vaguer flag name is gone.
    assert "--safety " not in help_text.replace("--safety-factor", "")


def test_chunk_rejects_old_safety_flag(tmp_path: Path, capsys) -> None:
    file = tmp_path / "meeting.mp4"
    file.write_bytes(b"fake")
    # allow_abbrev=False: --safety no longer prefix-matches --safety-factor.
    with pytest.raises(SystemExit) as excinfo:
        main(["chunk", str(file), "--dry-run", "--safety", "0.5"])
    assert excinfo.value.code == 2
