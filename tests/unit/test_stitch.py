"""Unit tests for ovox.stitch: merging, dedupe, segments and chunk result files.

No ffmpeg and no HTTP: everything here is pure logic over reply dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ovox.api import FatalSttError
from ovox.plan import ChunkEntry, ChunkPlan, Setting
from ovox.stitch import (
    build_text,
    chunk_transcript_name,
    load_chunk_transcript,
    load_replies,
    merge_texts,
    pending_chunks,
    transcript_file,
    write_chunk_transcript,
)

NAME_A = "meeting_chunk_00_0-300.mp3"
NAME_B = "meeting_chunk_01_300-600.mp3"  # silence cut: overlap_before 0
NAME_C = "meeting_chunk_02_600-750.mp3"


def make_plan(
    entries: list[tuple[float, float, float]] | None = None,
    *,
    overlap_flags: list[bool] | None = None,
) -> ChunkPlan:
    """A three-chunk plan; overlap_flags[i] says chunk i follows a hard cut."""
    if entries is None:
        entries = [(0.0, 300.0, 0.0), (300.0, 600.0, 0.0), (600.0, 750.0, 1.0)]
    flags = overlap_flags if overlap_flags is not None else [False, False, True]
    chunks = [
        ChunkEntry(
            index=i,
            start=start,
            end=end,
            duration=end - start,
            estimated_size=8000,
            name=names[i] if (names := [NAME_A, NAME_B, NAME_C]) else f"c{i}",
            reason="cut",
            overlap_before=overlap,
        )
        for i, ((start, end, overlap), flag) in enumerate(zip(entries, flags, strict=True))
    ]
    return ChunkPlan(
        input={"file": "/tmp/meeting.mp4", "path": "/tmp/meeting.mp4", "model": "m"},
        settings={
            "max_chunk_size_bytes": Setting(25 * 1024 * 1024, "default"),
            "max_chunk_duration_s": Setting(600.0, "default"),
            "safety": Setting(0.8, "default"),
            "sample_rate": Setting(16000, "default"),
            "channels": Setting(1, "default"),
            "bitrate_kbps": Setting(64, "default"),
        },
        chunks=chunks,
        warnings=[],
        totals={"num_chunks": len(chunks)},
        split_method="silence",
    )


# --- merge_texts ---------------------------------------------------------------


def test_merge_texts_joins_when_no_overlap() -> None:
    assert merge_texts("first part ends here", "and continues") == (
        "first part ends here and continues"
    )


def test_merge_texts_removes_overlap_after_hard_cut() -> None:
    prev = "the quick brown fox jumps over the lazy dog"
    cur = "the lazy dog then ran away"
    # The 3-word overlap is dropped; the joined text has it once.
    assert merge_texts(prev, cur) == ("the quick brown fox jumps over the lazy dog then ran away")


def test_merge_texts_ignores_punctuation_and_case() -> None:
    prev = "Hello, world. Hello,"
    cur = "hello world, hello, again."
    assert merge_texts(prev, cur) == "Hello, world. Hello, again."


def test_merge_texts_prefers_the_longest_match() -> None:
    prev = "one two three four five"
    cur = "three four five six"
    assert merge_texts(prev, cur) == "one two three four five six"


def test_merge_texts_dedupes_a_short_overlap_by_default() -> None:
    # The default min match is one word: even a 2-word overlap is dropped.
    prev = "one two three"
    cur = "two three four"
    assert merge_texts(prev, cur) == "one two three four"


def test_merge_texts_min_match_words_disables_short_dedupe() -> None:
    # With a higher minimum, a 2-word overlap is not specific enough.
    prev = "one two three"
    cur = "two three four"
    assert merge_texts(prev, cur, min_match_words=3) == "one two three two three four"


def test_merge_texts_empty_sides() -> None:
    assert merge_texts("", "abc") == "abc"
    assert merge_texts("abc", "") == "abc"
    assert merge_texts("", "") == ""


def test_merge_texts_respects_max_overlap_words() -> None:
    words = [f"w{i}" for i in range(20)]
    prev = " ".join(words[:20])
    # The alignment gives exactly a 15-word overlap (w5..w19): above the
    # default MAX_OVERLAP_WORDS (12), so the default call does not dedupe...
    cur = " ".join(words[5:20]) + " tail"
    assert merge_texts(prev, cur) == (prev + " " + cur)
    # ...but with a raised limit the overlap is dropped.
    assert merge_texts(prev, cur, max_overlap_words=20) == (prev + " tail")


def test_merge_texts_drops_the_previous_side_of_the_overlap() -> None:
    """The repeated run is dropped from the *next* chunk's text, never from
    the previous one: the next chunk heard the cut word whole, so its copy is
    what stays. Here the next chunk re-speaks the overlap with different
    spelling — that spelling is what survives."""
    prev = "one two three"
    cur = "ONE TWO THREE four"  # the overlap, re-transcribed
    assert merge_texts(prev, cur) == "one two three four"
    # The previous text is untouched; only the start of the current text is
    # dropped.
    assert merge_texts(prev, "three four") == "one two three four"


def test_merge_texts_keeps_a_garbled_last_word_before_the_cut() -> None:
    """When the hard cut garbles the previous chunk's last word, no run
    matches and nothing is dropped: the garbled word is kept (an imperfect
    transcript is acceptable; dropping it would risk eating real words)."""
    prev = "the lazy dog was cut here asd"  # "asd": the model's garbled tail
    cur = "the lazy dog was cut here and ran"
    assert merge_texts(prev, cur) == prev + " " + cur


# --- build_text ----------------------------------------------------------------


def test_build_text_merges_only_after_hard_cuts() -> None:
    plan = make_plan()
    replies = [
        {"text": "alpha beta gamma"},
        {"text": "gamma delta"},  # silence cut: joined without dedupe
        {"text": "delta epsilon zeta"},  # hard cut: "delta" dropped
    ]
    assert build_text(plan, replies) == "alpha beta gamma gamma delta epsilon zeta"


def test_build_text_requires_one_reply_per_chunk() -> None:
    plan = make_plan()
    with pytest.raises(ValueError, match="expected 3 replies"):
        build_text(plan, [{"text": "x"}])


def test_build_text_rejects_reply_without_text() -> None:
    plan = make_plan()
    with pytest.raises(FatalSttError, match="no 'text' field"):
        build_text(plan, [{"text": "a"}, {"other": 1}, {"text": "c"}])


# --- per-chunk transcript files -----------------------------------------------------


def test_chunk_transcript_name_from_chunk_name() -> None:
    assert chunk_transcript_name(NAME_A) == "chunk_00.json"
    assert chunk_transcript_name(NAME_B) == "chunk_01.json"
    assert chunk_transcript_name("a_chunk_10_500-900.mp3") == "chunk_10.json"


def test_chunk_transcript_name_rejects_other_names() -> None:
    with pytest.raises(ValueError, match="not a chunk file name"):
        chunk_transcript_name("ovox-chunk-manifest.json")


def test_transcript_round_trip(tmp_path: Path) -> None:
    payload = {"schema": 1, "model": "m", "reply": {"text": "hello"}}
    out = write_chunk_transcript(tmp_path, NAME_B, payload)
    assert out == transcript_file(tmp_path, NAME_B) == tmp_path / "chunk_01.json"

    loaded = load_chunk_transcript(tmp_path, NAME_B)
    assert loaded == payload
    # No leftover temp file (the rename is atomic and clean).
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_missing_is_none(tmp_path: Path) -> None:
    assert load_chunk_transcript(tmp_path, NAME_B) is None


def test_load_corrupt_is_none(tmp_path: Path) -> None:
    (tmp_path / "chunk_01.json").write_text("{not json")
    assert load_chunk_transcript(tmp_path, NAME_B) is None


def test_load_non_object_is_none(tmp_path: Path) -> None:
    (tmp_path / "chunk_01.json").write_text("[1, 2]")
    assert load_chunk_transcript(tmp_path, NAME_B) is None


def test_load_replies_and_pending(tmp_path: Path) -> None:
    plan = make_plan()
    assert load_replies(tmp_path, plan) == [None, None, None]
    assert [i for i, _, _ in pending_chunks(tmp_path, plan, [None, None, None])] == [0, 1, 2]

    write_chunk_transcript(tmp_path, NAME_A, {"reply": {"text": "a"}})
    write_chunk_transcript(tmp_path, NAME_C, {"reply": {"text": "c"}})
    replies = load_replies(tmp_path, plan)
    assert [r is None for r in replies] == [False, True, False]
    pending = pending_chunks(tmp_path, plan, replies)
    assert [i for i, _, _ in pending] == [1]
    assert pending[0][1].name == NAME_B
    assert pending[0][2] == tmp_path / NAME_B


def test_pending_requires_matching_lengths(tmp_path: Path) -> None:
    plan = make_plan()
    with pytest.raises(ValueError, match="expected 3 replies"):
        pending_chunks(tmp_path, plan, [None])


def test_write_overwrites_atomically(tmp_path: Path) -> None:
    write_chunk_transcript(tmp_path, NAME_A, {"reply": {"text": "old"}})
    write_chunk_transcript(tmp_path, NAME_A, {"reply": {"text": "new"}})
    loaded = load_chunk_transcript(tmp_path, NAME_A)
    assert loaded is not None
    assert json.loads((tmp_path / "chunk_00.json").read_text())["reply"]["text"] == "new"
