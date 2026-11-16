"""Unit tests for ovox.chunking. No ffmpeg involved: planning and parsing are pure."""

import itertools
import subprocess
from pathlib import Path

import pytest

from ovox import chunking
from ovox.chunking import (
    ChunkSpec,
    Silence,
    detect_silence,
    max_chunk_seconds,
    parse_silencedetect,
    plan_chunks,
)


class FakeFFmpeg:
    """Stands in for ovox.audio.run_ffmpeg and records the commands it was called with."""

    def __init__(self) -> None:
        self.returncode = 0
        self.stderr = ""
        self.raises: Exception | None = None
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, timeout))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr=self.stderr)


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFFmpeg:
    fake = FakeFFmpeg()
    monkeypatch.setattr(chunking.audio, "run_ffmpeg", fake)
    return fake


# --- chunk planning (pure) ---------------------------------------------------


def test_short_audio_is_one_chunk() -> None:
    assert plan_chunks(50, [], max_len=100) == [ChunkSpec(0, 0, 50)]


def test_cuts_on_longest_silence_in_window_without_overlap() -> None:
    sil = [Silence(80, 80.5), Silence(90, 92), Silence(150, 150.4)]
    specs = plan_chunks(160, sil, max_len=100, search_window=25)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [(0, 91, 0), (91, 160, 0)]


def test_hard_cut_with_overlap_when_no_silence() -> None:
    specs = plan_chunks(250, [], max_len=100, overlap=1.0)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [
        (0, 100, 0),
        (99, 199, 1.0),
        (198, 250, 1.0),
    ]
    assert all(s.duration <= 100 for s in specs)


def test_duration_zero_returns_no_chunks() -> None:
    assert plan_chunks(0, [], max_len=100) == []


def test_chunks_always_cover_the_whole_audio() -> None:
    sil = [Silence(10, 12), Silence(60, 61), Silence(200, 205)]
    for overlap in (0.0, 1.0, 5.0):
        specs = plan_chunks(210, sil, max_len=100, overlap=overlap)
        assert specs[0].start == 0
        assert specs[-1].end == 210
        assert all(a.end == b.start + b.overlap_before for a, b in itertools.pairwise(specs))
        assert all(0 <= s.overlap_before <= 100 for s in specs)


def test_silence_cut_is_shared_by_nothing() -> None:
    # Cutting at the center of (48, 50) means chunk 1 owns [0, 49) and chunk 2
    # owns [49, 149): no repeated audio at the silence, unlike the hard cut at
    # the end of chunk 2 (which re-sends 5 s).
    sil = [Silence(48, 50)]
    specs = plan_chunks(150, sil, max_len=100, overlap=5.0, search_window=60)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [
        (0, 49, 0),
        (49, 149, 0),
        (144, 150, 5.0),
    ]


def test_short_silences_are_ignored() -> None:
    # 0.1 s < MIN_SILENCE_S (0.2): no cut is worth it, so fall back to the hard cut.
    sil = [Silence(50, 50.1)]
    specs = plan_chunks(150, sil, max_len=100, overlap=1.0)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [
        (0, 100, 0),
        (99, 150, 1.0),
    ]


def test_silence_outside_the_search_window_is_ignored() -> None:
    sil = [Silence(40, 45)]
    specs = plan_chunks(150, sil, max_len=100, search_window=25, overlap=1.0)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [
        (0, 100, 0),
        (99, 150, 1.0),
    ]


def test_silence_before_the_chunk_start_is_ignored() -> None:
    # Trailing silence starting just before the second chunk starts (99 s) must
    # not pull the cut back; the plan is identical to the no-silence case.
    sil = [Silence(98, 250)]
    specs = plan_chunks(250, sil, max_len=100, overlap=1.0)
    assert [(s.start, s.end, s.overlap_before) for s in specs] == [
        (0, 100, 0),
        (99, 199, 1.0),
        (198, 250, 1.0),
    ]


def test_overlap_larger_than_half_chunk_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        plan_chunks(250, [], max_len=100, overlap=60.0)


def test_default_limits_are_compatible() -> None:
    # The out-of-the-box limits (25 MiB, 600 s, 80% safety) are self-consistent.
    limit = max_chunk_seconds()
    assert limit == pytest.approx(480.0)
    specs = plan_chunks(1000, [], limit, overlap=5.0)
    assert all(s.duration <= limit for s in specs)


def test_invalid_plan_chunks_arguments() -> None:
    with pytest.raises(ValueError, match="max_len"):
        plan_chunks(100, [], max_len=0)
    with pytest.raises(ValueError, match="max_bytes and max_seconds"):
        plan_chunks(100, [], max_bytes=0)


# --- max_chunk_seconds ---------------------------------------------------------


def test_max_chunk_seconds_respects_both_limits() -> None:
    assert max_chunk_seconds(25 * 1024 * 1024, 10_000, safety=1) == pytest.approx(6553.6)
    assert max_chunk_seconds(25 * 1024 * 1024, 600, safety=1) == 600


def test_max_chunk_seconds_default_safety() -> None:
    # 600 s * 0.8: the duration limit is below the size limit, so it wins.
    assert max_chunk_seconds() == pytest.approx(480.0)


def test_max_chunk_seconds_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_bytes and max_seconds"):
        max_chunk_seconds(max_seconds=0)
    with pytest.raises(ValueError, match="safety"):
        max_chunk_seconds(safety=0)
    with pytest.raises(ValueError, match="safety"):
        max_chunk_seconds(safety=1.5)


# --- parse_silencedetect ---------------------------------------------------------


def test_parse_silencedetect_handles_trailing_silence() -> None:
    err = (
        "[silencedetect @ 0x1] silence_start: 1.5\n"
        "[silencedetect @ 0x1] silence_end: 2.25 | silence_duration: 0.75\n"
        "[silencedetect @ 0x1] silence_start: 9\n"
    )
    assert parse_silencedetect(err, 10) == [Silence(1.5, 2.25), Silence(9, 10)]


def test_parse_silencedetect_plain_pairs() -> None:
    err = (
        "silence_start: 1\n"
        "silence_end: 2 | silence_duration: 1\n"
        "silence_start: 5.5\n"
        "silence_end: 6.25 | silence_duration: 0.75\n"
    )
    assert parse_silencedetect(err) == [Silence(1, 2), Silence(5.5, 6.25)]


def test_parse_silencedetect_empty_and_unrelated_output() -> None:
    assert parse_silencedetect("") == []
    assert parse_silencedetect("ffmpeg version 7.0\nblah\n") == []


def test_parse_silencedetect_clamps_to_duration() -> None:
    err = "silence_start: 9\nsilence_end: 11 | silence_duration: 2\nsilence_start: 30\n"
    # duration=10 closes the trailing start and drops the interval past the end.
    assert parse_silencedetect(err, 10) == [Silence(9, 10)]


def test_parse_silencedetect_sorts_intervals() -> None:
    err = (
        "silence_start: 7\nsilence_end: 8 | silence_duration: 1\n"
        "silence_start: 1\nsilence_end: 2 | silence_duration: 1\n"
    )
    assert parse_silencedetect(err) == [Silence(1, 2), Silence(7, 8)]


# --- detect_silence (mocked ffmpeg) ---------------------------------------------


def test_detect_silence_command_and_parsing(fake_ffmpeg: FakeFFmpeg, tmp_path: Path) -> None:
    file = tmp_path / "audio.mp3"
    file.write_bytes(b"fake mp3")
    fake_ffmpeg.stderr = (
        "[silencedetect @ 0x1] silence_start: 0.5\n"
        "[silencedetect @ 0x1] silence_end: 1.75 | silence_duration: 1.25\n"
    )

    assert detect_silence(file, 10.0) == [Silence(0.5, 1.75)]

    command, timeout = fake_ffmpeg.calls[0]
    assert command[0] == "ffmpeg"
    assert command[-3:] == ["-f", "null", "-"]
    assert f"silencedetect=noise=-30dB:d={chunking.MIN_SILENCE_S}" in command
    assert str(file.resolve()) in command
    assert timeout == pytest.approx(30.0 + 0.2 * 10.0)


def test_detect_silence_failure_includes_stderr_tail(
    fake_ffmpeg: FakeFFmpeg, tmp_path: Path
) -> None:
    file = tmp_path / "audio.mp3"
    file.write_bytes(b"fake mp3")
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.stderr = "first line\nlast line"

    with pytest.raises(chunking.AudioExtractionError, match="last line"):
        detect_silence(file, 10.0)
