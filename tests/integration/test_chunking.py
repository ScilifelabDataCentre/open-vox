"""Integration tests for ovox.chunking: real ffmpeg silencedetect on real files."""

import itertools
from pathlib import Path

import pytest

from ovox.audio import extract_audio, workdir
from ovox.chunking import detect_silence, plan_chunks


def test_detect_silence_finds_the_middle_gap(media_dir: Path) -> None:
    silences = detect_silence(media_dir / "tone_with_silence.wav")

    assert len(silences) == 1
    assert 0.85 <= silences[0].start <= 1.1
    assert 1.9 <= silences[0].end <= 2.15


def test_detect_silence_clamps_trailing_silence(media_dir: Path) -> None:
    silences = detect_silence(media_dir / "tone_with_trailing_silence.wav")

    assert len(silences) == 1
    assert 1.85 <= silences[0].start <= 2.15
    # The silence runs to the end of the file and must not extend past it.
    assert 3.7 <= silences[0].end <= 4.000001


def test_detect_silence_finds_nothing_in_continuous_audio(media_dir: Path) -> None:
    assert detect_silence(media_dir / "tone_only.wav") == []


def test_plan_chunks_from_real_silence(media_dir: Path) -> None:
    silences = detect_silence(media_dir / "tone_with_silence.wav")
    specs = plan_chunks(3.0, silences, max_len=2.0)

    assert len(specs) == 2
    assert specs[0].start == 0
    # The cut lands on the real silence around 1.5 s, not at the hard cut at 2 s.
    assert 1.35 <= specs[0].end <= 1.65
    assert specs[0].overlap_before == 0
    assert specs[1].start == specs[0].end
    assert specs[1].end == 3.0


def test_extracted_mp3_is_cut_on_its_pauses(media_dir: Path) -> None:
    # The real pipeline runs silencedetect on the extracted MP3, whose encoder
    # padding shifts times slightly. 54 s with a pause every 9 s, max 20 s chunks.
    with workdir() as d:
        mp3 = extract_audio(media_dir / "speech_like.wav", d)
        silences = detect_silence(mp3)
        duration = 54.0
        specs = plan_chunks(duration, silences, max_len=20.0, search_window=10.0)

    assert len(specs) == 3
    assert specs[-1].end == pytest.approx(duration)
    for a, b in itertools.pairwise(specs):
        assert b.overlap_before == 0  # every cut found a pause
        # Pauses are at 8-9, 17-18, 26-27, 35-36, 44-45 s.
        assert any(lo <= a.end <= lo + 1 for lo in (8, 17, 26, 35, 44))
