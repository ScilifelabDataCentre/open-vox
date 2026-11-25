"""Unit tests for ovox.chunking. No ffmpeg involved: planning and parsing are pure."""

import itertools
import math
import random
import subprocess
from pathlib import Path

import pytest

from ovox import chunking
from ovox.chunking import (
    MIN_TAIL_S,
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
        self.stdout = ""
        self.stderr = ""
        self.raises: Exception | None = None
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, timeout))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            command, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFFmpeg:
    fake = FakeFFmpeg()
    monkeypatch.setattr(chunking.audio, "run_ffmpeg", fake)
    return fake


def _plan(specs: list[ChunkSpec]) -> list[tuple[float, float, float]]:
    return [(s.start, s.end, s.overlap_before) for s in specs]


# --- chunk planning (pure) ---------------------------------------------------


def test_short_audio_is_one_chunk() -> None:
    assert plan_chunks(50, [], max_len=100) == [ChunkSpec(0, 0, 50)]


def test_audio_exactly_max_len_is_one_chunk() -> None:
    assert plan_chunks(100, [], max_len=100) == [ChunkSpec(0, 0, 100)]


def test_cuts_on_longest_silence_in_window_without_overlap() -> None:
    sil = [Silence(80, 80.5), Silence(90, 92), Silence(150, 150.4)]
    specs = plan_chunks(160, sil, max_len=100, search_window=25)
    assert _plan(specs) == [(0, 91, 0), (91, 160, 0)]


def test_hard_cut_with_overlap_when_no_silence() -> None:
    specs = plan_chunks(250, [], max_len=100, overlap=1.0)
    assert _plan(specs) == [(0, 100, 0), (99, 199, 1.0), (198, 250, 1.0)]


def test_hard_cuts_use_default_overlap() -> None:
    specs = plan_chunks(150, [], max_len=100)
    assert specs[1].overlap_before == chunking.DEFAULT_OVERLAP_S


def test_duration_zero_returns_no_chunks() -> None:
    assert plan_chunks(0, [], max_len=100) == []


def test_silence_cut_is_shared_by_nothing() -> None:
    # Chunk 1 is cut inside the silence and re-sends nothing; chunk 2 finds no
    # silence and is hard-cut, so chunk 3 repeats its last 5 s.
    specs = plan_chunks(200, [Silence(88, 90)], max_len=100, overlap=5.0)
    assert _plan(specs) == [(0, 89, 0), (89, 189, 0), (184, 200, 5.0)]


def test_short_silences_are_ignored() -> None:
    # 0.1 s < MIN_SILENCE_S (0.2): not worth a cut, so fall back to the hard cut.
    specs = plan_chunks(150, [Silence(90, 90.1)], max_len=100, overlap=1.0)
    assert _plan(specs) == [(0, 100, 0), (99, 150, 1.0)]


def test_silence_outside_the_search_window_is_ignored() -> None:
    specs = plan_chunks(150, [Silence(60, 65)], max_len=100, search_window=25, overlap=1.0)
    assert _plan(specs) == [(0, 100, 0), (99, 150, 1.0)]


def test_long_silence_across_the_planned_end_is_used() -> None:
    # A pause from 98 s to 250 s: cut inside it each time, never with overlap.
    specs = plan_chunks(250, [Silence(98, 250)], max_len=100, overlap=1.0)
    assert _plan(specs) == [(0, 99, 0), (99, 184, 0), (184, 250, 0)]


def test_never_cuts_before_half_of_max_len() -> None:
    # Short pauses every second used to give many 1 s chunks.
    sil = [Silence(i, i + 0.3) for i in range(1, 60)]
    specs = plan_chunks(60, sil, max_len=20, overlap=1.0)
    assert all(s.duration >= 10 for s in specs[:-1])
    assert len(specs) <= 6


def test_equal_silences_prefer_the_first() -> None:
    specs = plan_chunks(150, [Silence(80, 82), Silence(90, 92)], max_len=100, search_window=30)
    assert specs[0].end == 81


def test_silence_on_the_window_edges_counts() -> None:
    # Window is [75, 100]: silences touching either edge are clipped, not dropped.
    assert plan_chunks(150, [Silence(74, 76)], max_len=100, search_window=25)[0].end == 75.5
    assert plan_chunks(150, [Silence(99, 101)], max_len=100, search_window=25)[0].end == 99.5


def test_no_sliver_at_the_end() -> None:
    specs = plan_chunks(100.01, [], max_len=100, overlap=0.0)
    assert len(specs) == 2
    assert specs[-1].duration >= MIN_TAIL_S


def test_plan_invariants_on_random_inputs() -> None:
    rng = random.Random(1234)
    for _ in range(500):
        duration = rng.uniform(0.5, 3000)
        max_len = rng.uniform(5, 600)
        overlap = rng.uniform(0, max_len / 2)
        starts = sorted(rng.uniform(0, duration) for _ in range(rng.randint(0, 50)))
        silences = [Silence(t, t + rng.uniform(0.05, 20)) for t in starts]
        specs = plan_chunks(duration, silences, max_len=max_len, overlap=overlap)

        assert specs[0].start == 0 and specs[0].overlap_before == 0
        assert specs[-1].end == pytest.approx(duration)
        assert all(s.duration <= max_len + 1e-9 for s in specs)
        for a, b in itertools.pairwise(specs):
            assert b.start + b.overlap_before == pytest.approx(a.end)
            assert b.start > a.start
        if len(specs) > 1:
            assert specs[-1].duration >= MIN_TAIL_S - 1e-9
        assert len(specs) <= math.ceil(duration / (max_len / 2)) + 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_len": 0}, "max_len"),
        ({"max_bytes": 0}, "max_bytes and max_seconds"),
        ({"max_len": 100, "overlap": -1}, "overlap"),
        ({"max_len": 100, "overlap": 60}, "overlap"),
        ({"max_len": 100, "search_window": 0}, "search_window"),
    ],
)
def test_invalid_plan_chunks_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plan_chunks(250, [], **kwargs)  # type: ignore[arg-type]


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="duration"):
        plan_chunks(-1, [], max_len=100)


def test_overlap_of_exactly_half_is_allowed() -> None:
    specs = plan_chunks(250, [], max_len=100, overlap=50)
    assert specs[1].overlap_before == 50


# --- max_chunk_seconds ---------------------------------------------------------


def test_max_chunk_seconds_respects_both_limits() -> None:
    # 25 MiB at 64 kbps is ~3276.8 s, above the 10 000 s cap -> size limit wins.
    assert max_chunk_seconds(25 * 1024 * 1024, 10_000, safety=1) == pytest.approx(3276.8)
    assert max_chunk_seconds(25 * 1024 * 1024, 600, safety=1) == 600


def test_max_chunk_seconds_uses_the_configured_bitrate() -> None:
    # The size limit converts to seconds using the same bitrate the audio is
    # encoded at, so doubling the rate halves how many seconds fit.
    assert max_chunk_seconds(64_000, 10_000, safety=1, bitrate_bps=32_000) == pytest.approx(16.0)
    assert max_chunk_seconds(64_000, 10_000, safety=1, bitrate_bps=64_000) == pytest.approx(8.0)


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
    with pytest.raises(ValueError, match="bitrate_bps"):
        max_chunk_seconds(bitrate_bps=0)


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
    # duration=10 clamps the first interval and drops the one past the end.
    assert parse_silencedetect(err, 10) == [Silence(9, 10)]


def test_parse_silencedetect_clamps_negative_start_without_duration() -> None:
    err = "silence_start: -0.02\nsilence_end: 0.5 | silence_duration: 0.52\n"
    assert parse_silencedetect(err) == [Silence(0, 0.5)]


def test_parse_silencedetect_ignores_end_without_start() -> None:
    err = "silence_end: 2 | silence_duration: 1\nsilence_start: 5\nsilence_end: 6 | x\n"
    assert parse_silencedetect(err) == [Silence(5, 6)]


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
    # Runs on the source file, selecting the first audio stream (so it works on
    # video sources, and measures the same audio the chunks are cut from).
    assert command[command.index("-map") + 1] == "0:a:0"
    assert "-vn" in command
    assert command[-3:] == ["-f", "null", "-"]
    assert f"silencedetect=noise=-30dB:d={chunking.MIN_SILENCE_S}" in command
    assert str(file.resolve()) in command
    assert timeout == pytest.approx(30.0 + 0.2 * 10.0)


def test_detect_silence_probes_duration_when_missing(
    fake_ffmpeg: FakeFFmpeg, tmp_path: Path
) -> None:
    file = tmp_path / "audio.mp3"
    file.write_bytes(b"fake mp3")
    fake_ffmpeg.stdout = "12.5\n"
    fake_ffmpeg.stderr = "silence_start: 11\n"  # trailing silence, closed at the probed end

    assert detect_silence(file) == [Silence(11, 12.5)]
    assert [cmd[0] for cmd, _ in fake_ffmpeg.calls] == ["ffprobe", "ffmpeg"]
    assert fake_ffmpeg.calls[1][1] == pytest.approx(30.0 + 0.2 * 12.5)


def test_detect_silence_bad_probe_output(fake_ffmpeg: FakeFFmpeg, tmp_path: Path) -> None:
    fake_ffmpeg.stdout = "N/A\n"
    with pytest.raises(chunking.AudioProcessingError, match="duration"):
        detect_silence(tmp_path / "audio.mp3")


def test_detect_silence_failure_includes_stderr_tail(
    fake_ffmpeg: FakeFFmpeg, tmp_path: Path
) -> None:
    file = tmp_path / "audio.mp3"
    file.write_bytes(b"fake mp3")
    fake_ffmpeg.returncode = 1
    fake_ffmpeg.stderr = "first line\nlast line"

    with pytest.raises(chunking.AudioProcessingError, match="last line"):
        detect_silence(file, 10.0)


def test_detect_silence_propagates_run_ffmpeg_errors(
    fake_ffmpeg: FakeFFmpeg, tmp_path: Path
) -> None:
    fake_ffmpeg.raises = chunking.AudioProcessingError("ffmpeg timed out")
    with pytest.raises(chunking.AudioProcessingError, match="timed out"):
        detect_silence(tmp_path / "audio.mp3", 10.0)
