"""Unit tests for ovox.plan. The ffmpeg steps are mocked; planning is real."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ovox import audio
from ovox.config import AudioConfig, ChunkConfig
from ovox.media import MediaInfo
from ovox.plan import (
    build_plan,
    chunk_name,
    estimate_mp3_bytes,
    plan_manifest_text,
    write_chunks,
)


def make_info(tmp_path: Path, *, duration: float = 10.0, name: str = "video") -> MediaInfo:
    path = tmp_path / f"{name}.mp4"
    path.write_bytes(b"fake media")
    return MediaInfo(
        path=path,
        format_name="mp4",
        duration=duration,
        audio_streams=1,
        video_streams=1,
        audio_codec="aac",
        audio_sample_rate=44100,
        audio_channels=2,
        audio_track_index=0,
    )


class FakeAudio:
    """Stands in for the ffmpeg-backed audio functions used by plan."""

    def __init__(self, cut_size: int = 1000) -> None:
        self.cut_size = cut_size
        self.cuts: list[tuple[Path, float, float, Path]] = []
        self.extractions: list[Path] = []
        self.format: tuple[int | None, int | None, int | None] | None = None

    def extract_audio(
        self,
        source,
        dest_dir,
        *,
        duration=None,
        sample_rate=None,
        channels=None,
        bitrate_bps=None,
    ):
        self.extractions.append(source)
        self.format = (sample_rate, channels, bitrate_bps)
        return dest_dir / "audio.mp3"

    def cut_audio(self, source, output, start, end, **upload_format):
        output.write_bytes(b"x" * self.cut_size)
        self.cuts.append((source, start, end, output))
        return output

    def decoded_duration(self, path, *, sample_rate=16000):
        # A real MP3 of cut_size bytes at 64 kbps is cut_size * 8 / 64000 seconds.
        return Path(path).stat().st_size * 8 / 64000


@pytest.fixture
def fake_audio(monkeypatch: pytest.MonkeyPatch) -> FakeAudio:
    fake = FakeAudio()
    monkeypatch.setattr(audio, "extract_audio", fake.extract_audio)
    monkeypatch.setattr(audio, "cut_audio", fake.cut_audio)
    monkeypatch.setattr(audio, "decoded_duration", fake.decoded_duration)
    return fake


def set_silences(monkeypatch: pytest.MonkeyPatch, silences) -> None:
    import ovox.plan

    monkeypatch.setattr(ovox.plan, "detect_silence", lambda *a, **k: silences)


def default_config() -> ChunkConfig:
    return ChunkConfig(
        max_chunk_size_bytes=25 * 1024 * 1024,
        max_chunk_duration_s=600.0,
        safety=0.8,
    )


def default_audio() -> AudioConfig:
    return AudioConfig(sample_rate=16000, channels=1, bitrate_bps=64_000)


# --- estimate_mp3_bytes --------------------------------------------------------


def test_estimate_mp3_bytes_matches_bitrate() -> None:
    # Default is 64 kbps: 1 s is 8000 bytes, 60 s is 480 000 bytes.
    assert estimate_mp3_bytes(1.0) == 8000
    assert estimate_mp3_bytes(60.0) == 480_000


def test_estimate_mp3_bytes_uses_the_configured_bitrate() -> None:
    # 32 kbps for 1 s is 4000 bytes; 64 kbps is 8000.
    assert estimate_mp3_bytes(1.0, 32_000) == 4000
    assert estimate_mp3_bytes(1.0, 64_000) == 8000


# --- chunk naming --------------------------------------------------------------


def test_chunk_name_is_consistent_and_sortable() -> None:
    names = [
        chunk_name("lecture", i, s, e)
        for i, (s, e) in enumerate([(0, 300), (300, 600), (600, 754.5)])
    ]
    assert names == [
        "lecture_chunk_00_0-300.mp3",
        "lecture_chunk_01_300-600.mp3",
        "lecture_chunk_02_600-754.5.mp3",
    ]
    assert names == sorted(names)
    # Times are rounded to 0.1 s.
    assert chunk_name("v", 3, 754.46, 910.24) == "v_chunk_03_754.5-910.2.mp3"


# --- build_plan ----------------------------------------------------------------


def test_build_plan_covers_whole_input_with_no_gaps(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=750.0)
    plan = build_plan(info, default_config(), default_audio(), max_seconds=300.0)

    assert plan.chunks[0].start == 0.0
    assert plan.chunks[-1].end == pytest.approx(750.0)
    # Continuous coverage: each next chunk starts at or before the previous
    # chunk's end (equal for silence cuts, earlier by the overlap for hard cuts).
    for a, b in zip(plan.chunks, plan.chunks[1:], strict=False):
        assert b.start <= a.end + 1e-9
    assert plan.totals["num_chunks"] == len(plan.chunks)
    assert plan.totals["total_audio_length"] == pytest.approx(750.0)


def test_build_plan_single_chunk_when_within_limits(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=5.0)
    plan = build_plan(info, default_config(), default_audio())

    assert len(plan.chunks) == 1
    assert plan.chunks[0].start == 0.0
    assert plan.chunks[0].end == pytest.approx(5.0)
    assert any("No split was needed" in w for w in plan.warnings)


def test_build_plan_respects_size_limit(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=120.0)
    # 100 KB limit at 64 kbps is ~12.8 s of audio, before the safety margin.
    plan = build_plan(info, default_config(), default_audio(), max_bytes=100 * 1024)

    # No chunk may exceed the limit in estimated size (the ~10.2 s cap, at 64
    # kbps, is well under 100 KB once the safety margin is applied).
    for e in plan.chunks:
        assert e.estimated_size <= 100 * 1024
    # More than one chunk was needed for 120 s under a ~10 s cap.
    assert len(plan.chunks) >= 5


def test_build_plan_reports_setting_sources(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    # 1 MiB size limit is generous for 10 s, so overlap=2.0 stays valid.
    plan = build_plan(
        info,
        default_config(),
        default_audio(),
        max_bytes=1024 * 1024,
        overlap=2.0,
        split_method="fixed",
    )

    assert plan.settings["max_chunk_size_bytes"].source == "flag"
    assert plan.settings["max_chunk_size_bytes"].value == 1024 * 1024
    assert plan.settings["overlap_s"].source == "flag"
    assert plan.settings["split_method"].value == "fixed"
    # Un-overridden values come from config (env or default).
    assert plan.settings["safety"].source in ("config", "default")
    assert plan.split_method == "fixed"


def test_build_plan_reports_input_details(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    plan = build_plan(info, default_config(), default_audio())

    assert plan.input["file"] == str(info.path)
    assert plan.input["format"] == "mp4"
    assert plan.input["codec"] == "aac"
    assert plan.input["duration"] == pytest.approx(10.0)
    assert plan.input["sample_rate"] == 44100
    assert plan.input["channels"] == 2
    assert plan.input["audio_track"] == 0


def test_build_plan_encodes_and_plans_at_the_configured_format(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audio is encoded and the sizes estimated at the same configured rate.

    This is the invariant the whole feature relies on: extraction and chunk
    planning must read the same bitrate, or every size estimate is off.
    """
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    audio_config = AudioConfig(sample_rate=48000, channels=2, bitrate_bps=128_000)
    plan = build_plan(info, default_config(), audio_config)

    # Extraction was told to encode at the configured format.
    assert fake_audio.format == (48000, 2, 128_000)
    # The plan records that format.
    assert plan.settings["sample_rate"].value == 48000
    assert plan.settings["channels"].value == 2
    assert plan.settings["bitrate_kbps"].value == 128
    # The size estimate uses the configured bitrate: 10 s at 128 kbps.
    assert plan.chunks[0].estimated_size == 10 * 128_000 // 8


def test_build_plan_warns_when_no_silence(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=750.0)
    plan = build_plan(info, default_config(), default_audio(), max_seconds=300.0)

    assert any("No silence found" in w for w in plan.warnings)
    # With no silence, non-final chunks are hard cuts.
    assert all(c.reason in ("hard cut (no silence found)", "end of audio") for c in plan.chunks)


def test_build_plan_uses_silence_when_present(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ovox.chunking import Silence

    set_silences(monkeypatch, [Silence(290, 295), Silence(590, 595)])
    info = make_info(tmp_path, duration=750.0)
    # 375 s * 0.8 safety -> a 300 s cap, whose window covers the 290-295 pause.
    plan = build_plan(
        info, default_config(), default_audio(), max_seconds=375.0, search_window=50.0
    )

    # The cut after the first 300 s lands in the 290-295 pause, not at 300.
    assert 290 < plan.chunks[0].end <= 295
    assert plan.chunks[0].reason == "cut on silence"
    assert len(plan.chunks) == 3
    assert not any("No silence found" in w for w in plan.warnings)


def test_build_plan_invalid_split_method_rejected(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path)
    with pytest.raises(ValueError, match="split_method"):
        build_plan(info, default_config(), default_audio(), split_method="nonsense")


# --- write_chunks --------------------------------------------------------------


def test_write_chunks_writes_planned_files(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=750.0, name="lecture")
    plan = build_plan(info, default_config(), default_audio(), max_seconds=300.0)
    out_dir = tmp_path / "chunks"

    paths = write_chunks(plan, out_dir, default_audio())

    assert len(paths) == len(plan.chunks)
    for entry, p in zip(plan.chunks, paths, strict=True):
        assert p == out_dir / entry.name
        assert p.exists()
        assert entry.actual_size == fake_audio.cut_size
    # Names are derived from the original filename.
    assert all(p.name.startswith("lecture_chunk_") for p in paths)


def test_write_chunks_records_actual_sizes(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    plan = build_plan(info, default_config(), default_audio())

    write_chunks(plan, tmp_path / "chunks", default_audio())

    assert all(e.actual_size is not None for e in plan.chunks)


def test_write_chunks_records_measured_duration(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    plan = build_plan(info, default_config(), default_audio())

    write_chunks(plan, tmp_path / "chunks", default_audio())

    # The manifest records the written length (measured), not just the plan.
    # A 1000-byte fake chunk at 64 kbps is 1000 * 8 / 64000 = 0.125 s.
    assert all(e.actual_duration == pytest.approx(0.125) for e in plan.chunks)
    # And each cut was made from the original source, not an intermediate file.
    assert all(cut[0] == info.path for cut in fake_audio.cuts)


def test_write_chunks_creates_output_dir(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=10.0)
    plan = build_plan(info, default_config(), default_audio())
    nested = tmp_path / "a" / "b"

    write_chunks(plan, nested, default_audio())

    assert nested.is_dir()


# --- manifest ------------------------------------------------------------------


def test_plan_manifest_matches_chunks(
    tmp_path: Path, fake_audio: FakeAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_silences(monkeypatch, [])
    info = make_info(tmp_path, duration=750.0)
    plan = build_plan(info, default_config(), default_audio(), max_seconds=300.0)

    data = json.loads(plan_manifest_text(plan))

    assert data["totals"]["num_chunks"] == len(data["chunks"])
    assert len(data["chunks"]) == len(plan.chunks)
    # Same fields and structure as the live plan.
    for a, b in zip(data["chunks"], plan.chunks, strict=True):
        assert a["start"] == b.start
        assert a["end"] == b.end
        assert a["duration"] == b.duration
        assert a["name"] == b.name
        assert a["estimated_size"] == b.estimated_size
        assert a["actual_size"] == b.actual_size
        assert a["actual_duration"] == b.actual_duration
    # Settings carry value + source.
    assert data["settings"]["max_chunk_duration_s"]["value"] == 300.0
    assert data["settings"]["max_chunk_duration_s"]["source"] == "flag"
