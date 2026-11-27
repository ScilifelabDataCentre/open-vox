"""Turning per-chunk STT replies into one transcript.

Each chunk's reply is stored as its own JSON file (``chunk_003.json``) by
:mod:`ovox.pipeline`; that file is the unit of resume, so the manifest never
changes after it is written. This module joins the replies into the final
text and reads what extra structure the endpoint chose to return.

The merge is text-only (see ``docs/design/transcription.md``): chunk texts are
joined in order, and after a hard cut the words the overlap repeats are
removed by matching the end of the previous text against the start of the next.
Most cuts land on silence and need no merge at all. When an endpoint does
return ``segments`` (some hosts do, even without being asked), they are kept
in the chunks' JSON untouched, but the global-timeline / segments-JSON output
is deferred until timestamps exist (the text merge removes overlap words that
a timeline built from raw segment times would still contain, and there is no
endpoint to test it against today).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ovox.api import FatalSttError
from ovox.plan import ChunkEntry, ChunkPlan

# How many trailing words are compared when deduplicating a hard-cut overlap.
# Longer matches are preferred (more specific, less chance of coincidental
# repetition); the search goes from the longest run down to one word, because
# the default 1 s overlap often transcribes to only two or three words, and a
# single repeated word at the boundary is still overlap.
MAX_OVERLAP_WORDS = 12
MIN_MATCH_WORDS = 1


def _normalize_word(word: str) -> str:
    """Lower-case a word and drop punctuation, so "Hello," and "hello" match.

    Diacritics are folded (NFKD, then combined marks dropped), so "café" and
    "cafe" match. The result can be empty (a pure-punctuation token); such
    tokens then match each other, which is fine for a dedupe heuristic.
    """
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in decomposed if ch.isalnum())


def _words(text: str) -> list[str]:
    """The words of *text*, whitespace-split (punctuation kept)."""
    return text.split()


def _match_key(word: str) -> str:
    return _normalize_word(word)


def _overlap_drop_length(
    prev_keys: list[str],
    cur_keys: list[str],
    *,
    max_overlap_words: int,
    min_match_words: int,
) -> int | None:
    """How many words at the start of *cur_keys* repeat the end of
    *prev_keys*, or None when no run matches.

    The longest matching run is preferred (it is the more specific one),
    bounded by *max_overlap_words* and at least *min_match_words*.
    """
    limit = min(max_overlap_words, len(prev_keys), len(cur_keys))
    for length in range(limit, min_match_words - 1, -1):
        if prev_keys[-length:] == cur_keys[:length]:
            return length
    return None


def merge_texts(
    previous: str,
    current: str,
    *,
    max_overlap_words: int = MAX_OVERLAP_WORDS,
    min_match_words: int = MIN_MATCH_WORDS,
) -> str:
    """Join two consecutive chunk texts, dropping the overlap repetition.

    After a hard cut the next chunk repeats the last few seconds of the
    previous one, so the model transcribes the same words twice. This looks
    for the longest run of words (at least *min_match_words*, at most
    *max_overlap_words*) whose lower-case punctuation-free form ends
    *previous* and starts *current*, and drops that run **from *current*** —
    the previous text is never changed. The next chunk is the side that heard
    the cut word whole, so its copy is what stays in the transcript; the
    previous chunk's copy of it was heard only in its (possibly clipped or
    garbled) tail. Words are compared by their normalized form, so
    punctuation and case differences do not defeat the match; the original
    spelling from *current* is what is dropped.

    With no matching run the texts are joined as-is (a silence cut, or a
    model that transcribed the overlap differently — a repeated word or two
    then remains, which is acceptable for the beta). In particular, when the
    previous chunk's last word is garbled by the cut and no run matches, the
    garbled word is kept: nothing is dropped, and both copies of the overlap
    may remain in the transcript.
    """
    prev_words = _words(previous)
    cur_words = _words(current)
    if not prev_words or not cur_words:
        return " ".join(t for t in (previous, current) if t).strip()
    drop = _overlap_drop_length(
        [_match_key(w) for w in prev_words],
        [_match_key(w) for w in cur_words],
        max_overlap_words=max_overlap_words,
        min_match_words=min_match_words,
    )
    if drop is None:
        return (previous + " " + current).strip()
    return (previous + " " + " ".join(cur_words[drop:])).strip()


def _dedupe_overlap(
    previous: str,
    current: str,
    *,
    max_overlap_words: int = MAX_OVERLAP_WORDS,
    min_match_words: int = MIN_MATCH_WORDS,
) -> str:
    """The part of *current* that is not a repeat of the end of *previous*.

    The overlap run is dropped when it matches; otherwise *current* is
    returned unchanged.
    """
    prev_words = _words(previous)
    cur_words = _words(current)
    if not prev_words or not cur_words:
        return current
    drop = _overlap_drop_length(
        [_match_key(w) for w in prev_words],
        [_match_key(w) for w in cur_words],
        max_overlap_words=max_overlap_words,
        min_match_words=min_match_words,
    )
    if drop is None:
        return current
    return " ".join(cur_words[drop:])


def build_text(plan: ChunkPlan, replies: Sequence[dict[str, Any]]) -> str:
    """Join the chunk replies, in plan order, into the transcript text.

    The overlap dedupe runs only on a chunk that follows a hard cut
    (``overlap_before > 0``); silence cuts share no audio, so their texts are
    joined directly.

    Args:
        plan: The chunk plan (provides the cut kinds, in order).
        replies: One reply dict per chunk, in the same order as ``plan.chunks``.

    Raises:
        FatalSttError: a reply is missing its ``text`` field.
    """
    if len(replies) != len(plan.chunks):
        raise ValueError(f"expected {len(plan.chunks)} replies, got {len(replies)}")
    parts: list[str] = []
    for entry, reply in zip(plan.chunks, replies, strict=True):
        text = reply.get("text")
        if not isinstance(text, str):
            raise FatalSttError(f"chunk {entry.name} has no 'text' field in its stored reply")
        if entry.overlap_before > 0 and parts:
            text = _dedupe_overlap(parts[-1], text)
        parts.append(text)
    return " ".join(p for p in parts if p).strip()


# --- Per-chunk transcript files -------------------------------------------------


def chunk_transcript_name(entry_name: str) -> str:
    """The transcript file name for a chunk: ``foo_chunk_01_5.2-98.3.mp3`` ->
    ``chunk_01.json`` (the index from the chunk file name, zero-padded to
    two digits like the chunk name itself)."""
    m = re.search(r"_chunk_(\d+)_", entry_name)
    if not m:
        raise ValueError(f"not a chunk file name: {entry_name!r}")
    return f"chunk_{int(m.group(1)):02d}.json"


def transcript_file(workdir: Path, entry_name: str) -> Path:
    return Path(workdir) / chunk_transcript_name(entry_name)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via ``.tmp`` + ``os.replace`` so a kill mid-write cannot leave
    a half-written file that a later resume would trust."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_chunk_transcript(workdir: Path, entry_name: str, payload: dict[str, Any]) -> Path:
    """Store one chunk's reply (and the settings it was asked with) next to the
    chunks, written safely."""
    out = transcript_file(workdir, entry_name)
    _write_json_atomic(out, payload)
    return out


def load_chunk_transcript(workdir: Path, entry_name: str) -> dict[str, Any] | None:
    """Read a stored chunk reply, or None when it does not exist.

    A file that is missing, truncated or not valid JSON is treated as not
    done: the chunk is re-transcribed and the file atomically rewritten.
    (The atomic rename already means a kill cannot leave a half-written file;
    this check is the belt-and-braces for manual tampering.)
    """
    path = transcript_file(workdir, entry_name)
    if not path.exists():
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_replies(workdir: Path, plan: ChunkPlan) -> list[dict[str, Any] | None]:
    """One entry per planned chunk: its stored result payload, or None when
    the chunk has no (valid) transcript file."""
    return [load_chunk_transcript(workdir, entry.name) for entry in plan.chunks]


def replies_from_payloads(
    payloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The replies out of stored result payloads.

    A result file stores the reply inside an envelope (``{"reply": {...}}``
    plus the model/endpoint it was asked with); a bare reply dict is accepted
    too. The merge and segments code only ever look at the reply itself.
    """
    replies: list[dict[str, Any]] = []
    for payload in payloads:
        inner = payload.get("reply")
        replies.append(inner if isinstance(inner, dict) else payload)
    return replies


def pending_chunks(
    workdir: Path,
    plan: ChunkPlan,
    replies: Sequence[dict[str, Any] | None],
) -> list[tuple[int, ChunkEntry, Path]]:
    """The chunks without a (valid) transcript file, as (index, entry, audio path).

    Progress is tracked by the presence of these files, so the manifest never
    changes after it is written and a resume is just "transcribe the chunks
    whose file is missing".
    """
    if len(replies) != len(plan.chunks):
        raise ValueError(f"expected {len(plan.chunks)} replies, got {len(replies)}")
    return [
        (entry.index, entry, Path(workdir) / entry.name)
        for entry, reply in zip(plan.chunks, replies, strict=True)
        if reply is None
    ]
