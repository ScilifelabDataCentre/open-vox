"""Chunk planning for transcription.

Splits an extracted MP3 into pieces that fit the STT endpoint's size and
duration limits, preferring cuts on silence. Planning and parsing are pure
functions so they can be unit tested without ffmpeg; only :func:`detect_silence`
runs ffmpeg (with the ``silencedetect`` filter).
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ovox import audio
from ovox.audio import MP3_BITRATE_BPS, AudioExtractionError

# Sensible defaults; every function taking them can be pointed at an
# endpoint's actual limits.
DEFAULT_MAX_CHUNK_SIZE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_CHUNK_SECONDS = 600.0
DEFAULT_SAFETY = 0.8
DEFAULT_OVERLAP_S = 1.0
DEFAULT_SEARCH_WINDOW_S = 30.0
# Silence shorter than this is not worth cutting on.
MIN_SILENCE_S = 0.2
# ffmpeg reports silencedetect times with one decimal place.
SILENCE_DETECT_TIMEOUT_PER_SECOND = 0.2
SILENCE_DETECT_TIMEOUT_BASE_S = 30.0

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Silence:
    """A silence interval, in seconds from the start of the audio."""

    start: float
    end: float


@dataclass(frozen=True)
class ChunkSpec:
    """One chunk to send to the STT endpoint.

    Args:
        start: Where the chunk begins, in seconds.
        overlap_before: How many seconds at the start of the chunk repeat the
            tail of the previous chunk (0 for the first chunk).
        end: Where the chunk ends, in seconds.
    """

    start: float
    overlap_before: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def max_chunk_seconds(
    max_bytes: int = DEFAULT_MAX_CHUNK_SIZE_BYTES,
    max_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    safety: float = DEFAULT_SAFETY,
) -> float:
    """Largest chunk length in seconds allowed by the size and duration limits.

    The MP3 bitrate is known at extraction time (see ovox.audio), so the byte
    limit converts to seconds exactly. ``safety`` keeps a margin below the
    endpoint's hard limits.
    """
    if max_bytes <= 0 or max_seconds <= 0:
        raise ValueError("max_bytes and max_seconds must be positive")
    if not 0 < safety <= 1:
        raise ValueError("safety must be in (0, 1]")
    from_bytes = max_bytes * 8.0 / MP3_BITRATE_BPS  # bytes -> bits -> seconds
    return min(from_bytes, max_seconds) * safety


def parse_silencedetect(stderr: str, duration: float | None = None) -> list[Silence]:
    """Parse ffmpeg's ``silencedetect`` log into silence intervals.

    ``duration`` clamps any interval and closes a trailing ``silence_start``
    that ffmpeg never finished reporting (silence running to the end of the
    file). Intervals are sorted and returned with non-negative starts.
    """
    starts: list[float] = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(stderr)]
    ends: list[float] = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(stderr)]

    silences = [Silence(start, end) for start, end in zip(starts, ends, strict=False)]
    if len(starts) > len(ends) and duration is not None:
        silences.append(Silence(starts[-1], duration))

    if duration is not None:
        silences = [
            Silence(max(0.0, s.start), min(duration, s.end))
            for s in silences
            if min(duration, s.end) > max(0.0, s.start)
        ]
    return sorted(silences, key=lambda s: s.start)


def plan_chunks(
    duration: float,
    silences: list[Silence],
    max_len: float | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_CHUNK_SIZE_BYTES,
    max_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    safety: float = DEFAULT_SAFETY,
    overlap: float = 0.0,
    search_window: float = DEFAULT_SEARCH_WINDOW_S,
) -> list[ChunkSpec]:
    """Split *duration* seconds of audio into chunks no longer than the limits.

    Cuts are made at the center of the longest silence inside a search window
    near the planned end of each chunk, so cuts land where the speech is
    already quiet and nothing is shared between the two resulting chunks.
    With no usable silence, the next chunk starts ``overlap`` seconds before
    the previous one ended, so words are not split across the STT endpoint's
    boundary. A chunk's ``overlap_before`` is exactly how much of its head is
    repeated audio from the previous chunk (0 for the first chunk).

    Args:
        duration: Total audio duration in seconds.
        silences: Silence intervals from :func:`detect_silence`.
        max_len: Hard cap on chunk length in seconds. Falls back to
            ``max_chunk_seconds(max_bytes, max_seconds, safety)``.
        max_bytes, max_seconds, safety: Endpoint limits used when *max_len*
            is not given.
        overlap: Seconds of repeated audio at the start of hard-cut chunks.
            Must be smaller than half of the chunk length.
        search_window: How far back from the planned end to look for a silence
            cut.

    Returns:
        The chunks covering the whole file, in order.
    """
    if duration <= 0:
        return []
    if max_len is None:
        max_len = max_chunk_seconds(max_bytes, max_seconds, safety)
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap > max_len / 2:
        raise ValueError(f"overlap must be at most half of max_len ({max_len / 2:g}s)")

    specs: list[ChunkSpec] = []
    start = 0.0
    ov = 0.0
    while start < duration:
        planned_end = min(start + max_len, duration)
        end, next_ov = _cut_point(start, planned_end, duration, silences, overlap, search_window)
        specs.append(ChunkSpec(start, ov, end))
        start = end - next_ov
        ov = next_ov
    return specs


def detect_silence(
    audio_file: str | os.PathLike[str],
    duration: float | None = None,
    *,
    noise: str = "-30dB",
    min_silence: float = MIN_SILENCE_S,
) -> list[Silence]:
    """Run ffmpeg's ``silencedetect`` over *audio_file* and return its silences.

    Args:
        audio_file: The extracted MP3 (see ovox.audio.extract_audio).
        duration: Clamp/clamp-assist for the parser; normally the duration
            from ovox.media's ffprobe.
        noise: A ffmpeg volume threshold for what counts as silence.
        min_silence: Ignore silences shorter than this many seconds.
    """
    if duration is None:
        duration = _probe_duration(audio_file)
    timeout = SILENCE_DETECT_TIMEOUT_BASE_S + SILENCE_DETECT_TIMEOUT_PER_SECOND * (duration or 0.0)
    result = audio.run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(Path(audio_file).resolve()),
            "-af",
            f"silencedetect=noise={noise}:d={min_silence}",
            "-f",
            "null",
            "-",
        ],
        timeout,
    )
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed while detecting silences in: {audio_file}\n"
            f"{audio.stderr_tail(result.stderr)}"
        )
    return parse_silencedetect(result.stderr, duration)


def _probe_duration(audio_file: str | os.PathLike[str]) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(Path(audio_file).resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioExtractionError(f"ffprobe failed while reading duration of: {audio_file}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioExtractionError(
            f"Could not read duration from ffprobe output for: {audio_file}"
        ) from exc


def _cut_point(
    start: float,
    planned_end: float,
    duration: float,
    silences: list[Silence],
    overlap: float,
    search_window: float,
) -> tuple[float, float]:
    """Choose where the current chunk ends.

    Returns:
        ``(end, next_overlap)``: the end of the current chunk and how much of
        the next chunk's head repeats it (0 when a silence was used, since
        nothing is shared then).
    """
    if planned_end >= duration:
        return duration, 0.0

    candidates = [
        ((s.start + s.end) / 2.0, s.end - s.start)
        for s in silences
        if s.start >= start
        and (s.end - s.start) >= MIN_SILENCE_S
        and planned_end - search_window <= (s.start + s.end) / 2.0 <= planned_end
    ]
    if candidates:
        center, _ = max(candidates, key=lambda c: c[1])
        return center, 0.0
    return planned_end, min(overlap, planned_end - start)
