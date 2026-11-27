# tests/integration/test_live_transcribe.py
"""Live end-to-end tests against the real STT endpoint, checked against the
expected transcripts committed with the short speech clips.

These are opt-in: they need a reachable endpoint with a valid key in the
environment (or the repo's ``.env`` — run ``ovox check`` first), so they are
marked ``live`` and excluded from the default run (see ``addopts`` in
pyproject.toml):

    pytest -m live                                  real transcription of the committed clips
    OVOX_TEST_MEDIA_DIR=~/media pytest -m "live and local_media"
                                                    same, for your own long/private files

The committed clips (``tests/data/audio/*.mp3``) each have a sibling
``.txt`` with the expected transcript; the comparison is word-based, so
minor punctuation or case differences do not fail the run.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ovox.config import AudioConfig, ChunkConfig, Config, SttConfig, TranscribeConfig
from ovox.media import validate_input
from ovox.pipeline import transcribe_file

pytestmark = [pytest.mark.live]

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"  # tests/data (shared with the local_media tests)
AUDIO_DIR = DATA_DIR / "audio"

COMMITTED_SPEECH = sorted(AUDIO_DIR.glob("*.mp3"))

LIVE_REASON = (
    "live STT endpoint tests need a reachable STT_ENDPOINT and a valid STT_API_KEY "
    "in the environment or .env (run 'ovox check'); excluded from the default run "
    "(see pyproject.toml addopts)"
)
NO_ENDPOINT = not (os.environ.get("STT_ENDPOINT") and os.environ.get("STT_API_KEY"))


def _configs() -> tuple[ChunkConfig, AudioConfig, SttConfig, TranscribeConfig]:
    config = Config.from_env()
    return config.chunk, config.audio, config.stt, config.transcribe


def _words(text: str) -> list[str]:
    # Unicode \w: keeps å/ä/ö; drops punctuation; keeps "it's" as one word
    return re.findall(r"\w+(?:'\w+)*", text.lower())


def _expected_text(expected: Path) -> str:
    text = expected.read_text(encoding="utf-8").strip()
    if not _words(text):
        pytest.fail(f"{expected} is empty; nothing to compare against")
    return text


def _assert_transcript_matches(
    actual: str, expected: str, *, min_recall: float = 0.8, max_inserted_run: int = 3
) -> None:
    exp, act = _words(expected), _words(actual)
    if len(exp) <= 2:
        pytest.fail(f"expected transcript too short to check: {exp!r}")
    sm = difflib.SequenceMatcher(a=exp, b=act, autojunk=False)  # autojunk breaks >200 words
    matched = sum(block.size for block in sm.get_matching_blocks())
    problems = []
    if matched / len(exp) < min_recall:
        problems.append(f"only {matched}/{len(exp)} expected words found in order")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        inserted = (j2 - j1) - (i2 - i1) if tag in ("insert", "replace") else 0
        if inserted > max_inserted_run:
            problems.append(f"extra text not in the speech: {' '.join(act[j1:j2])!r}")
    if problems:
        raise AssertionError(
            "\n  ".join(problems) + f"\n  expected: {expected}\n  actual:   {actual}"
        )


@pytest.fixture
def live_configs() -> Iterator[tuple[ChunkConfig, AudioConfig, SttConfig, TranscribeConfig]]:
    if NO_ENDPOINT:
        pytest.skip(LIVE_REASON)
    # The STT settings (endpoint, key, model, concurrency, retries) come from
    # the developer's environment / .env — the live endpoint is the point.
    # The chunk and audio settings keep the deterministic integration-test
    # values pinned by this folder's autouse conftest fixture.
    yield _configs()


def _transcribe(
    media: Path,
    tmp_path: Path,
    configs: tuple[ChunkConfig, AudioConfig, SttConfig, TranscribeConfig],
) -> dict[str, Any]:
    chunk_config, audio_config, stt, transcribe_config = configs
    return transcribe_file(
        media,
        chunk_config=chunk_config,
        audio_config=audio_config,
        stt=stt,
        transcribe_config=transcribe_config,
        workdir=tmp_path / "wd",
        output=tmp_path / f"{media.stem}.txt",
    )


def _assert_chunked_properly(media: Path, result: dict[str, Any]) -> None:
    """The local pipeline still works for this media: it probed the audio
    (validate_input raises for files without an audio stream) and transcribed
    every chunk it planned."""
    assert validate_input(media).audio_streams >= 1
    assert result["chunks_total"] >= 1
    assert result["chunks_transcribed"] == result["chunks_total"]
    assert result["text_file"].exists()


@pytest.mark.parametrize(
    "media",
    COMMITTED_SPEECH,
    ids=lambda p: str(p.relative_to(DATA_DIR)),
)
def test_live_transcribe_committed_clip(media: Path, tmp_path: Path, live_configs) -> None:
    """Real transcription of the committed clip, checked against its .txt.

    The clip is a few seconds long, well under the pinned 25 MiB / 600 s
    limits, so it is one chunk and the merged transcript is the endpoint's
    text for the whole clip.
    """
    expected_path = media.with_suffix(".txt")
    if not expected_path.is_file():
        pytest.skip(f"no expected transcript {expected_path} for {media}")
    expected = _expected_text(expected_path)
    result = _transcribe(media, tmp_path, live_configs)
    _assert_chunked_properly(media, result)
    assert result["chunks_total"] == 1, f"expected one chunk for {media}, got {result}"
    actual = result["text_file"].read_text(encoding="utf-8")
    print(repr(actual))
    _assert_transcript_matches(actual, expected)


# --- Your own long or private files (opt-in) ------------------------------------

LOCAL_MEDIA_DIR = os.environ.get("OVOX_TEST_MEDIA_DIR")
LOCAL_MEDIA_REASON = "OVOX_TEST_MEDIA_DIR not set"
NO_LOCAL_FILES_REASON = (
    f"no media files with an expected .txt in OVOX_TEST_MEDIA_DIR={LOCAL_MEDIA_DIR}"
)


def _local_media_pairs() -> list[tuple[Path, Path]]:
    """(media, expected .txt) pairs under $OVOX_TEST_MEDIA_DIR, including
    subfolders.

    A media file participates only when it has a sibling ``.txt`` with the
    same stem (its expected transcript); anything else in the folder is
    ignored.
    """
    if not LOCAL_MEDIA_DIR:
        return []
    root = Path(LOCAL_MEDIA_DIR).expanduser()
    if not root.is_dir():
        raise pytest.UsageError(f"OVOX_TEST_MEDIA_DIR is not a directory: {root}")
    pairs: list[tuple[Path, Path]] = []
    for media in sorted(root.rglob("*")):
        if not media.is_file() or media.name.startswith(".") or media.suffix.lower() == ".txt":
            continue
        expected = media.with_suffix(".txt")
        if expected.is_file():
            pairs.append((media, expected))
    return pairs


LOCAL_MEDIA_PAIRS = _local_media_pairs()


def _local_media_ids(value: object) -> str:
    """A test id per parameter value, relative to the media dir root."""
    if isinstance(value, Path):
        return str(value.relative_to(Path(LOCAL_MEDIA_DIR or "").expanduser()))
    return "none"


@pytest.mark.local_media
@pytest.mark.skipif(not LOCAL_MEDIA_DIR, reason=LOCAL_MEDIA_REASON)
@pytest.mark.skipif(bool(LOCAL_MEDIA_DIR) and not LOCAL_MEDIA_PAIRS, reason=NO_LOCAL_FILES_REASON)
@pytest.mark.parametrize(
    "media, expected_path",
    LOCAL_MEDIA_PAIRS or [(None, None)],
    ids=_local_media_ids,
)
def test_live_transcribe_local_media(
    media: Path, expected_path: Path, tmp_path: Path, live_configs
) -> None:
    """Real transcription of your own long files: the closest thing to a full
    test of real use (extraction, chunking, many concurrent requests, the
    merge)."""
    expected = _expected_text(expected_path)
    result = _transcribe(media, tmp_path, live_configs)
    _assert_chunked_properly(media, result)
    actual = result["text_file"].read_text(encoding="utf-8")
    print(repr(actual))
    _assert_transcript_matches(actual, expected)
