"""Chunk planning for transcription.

Plans how an extracted MP3 is split into pieces that fit the STT endpoint's
size and duration limits, preferring cuts on silence. Planning and parsing are
pure functions so they can be unit tested without ffmpeg; only
:func:`detect_silence` runs ffmpeg (with the ``silencedetect`` filter).
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ovox import audio
from ovox.audio import DEFAULT_BITRATE_BPS, AudioExtractionError

# Sensible defaults; every function taking them can be pointed at an
# endpoint's actual limits.
DEFAULT_MAX_CHUNK_SIZE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_CHUNK_SECONDS = 600.0
DEFAULT_SAFETY = 0.8
DEFAULT_OVERLAP_S = 1.0
DEFAULT_SEARCH_WINDOW_S = 30.0
# Silence shorter than this is not worth cutting on.
MIN_SILENCE_S = 0.2
# Never leave a last chunk shorter than this: STT endpoints may reject very
# short audio or invent text for it.
MIN_TAIL_S = 1.0
# ffmpeg decodes far faster than real time; scale the timeout with the length.
SILENCE_DETECT_TIMEOUT_PER_SECOND = 0.2
SILENCE_DETECT_TIMEOUT_BASE_S = 30.0
PROBE_TIMEOUT_S = 30.0

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
            tail of the previous chunk (0 for the first chunk and after a
            silence cut).
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
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
) -> float:
    """Largest chunk length in seconds allowed by the size and duration limits.

    The MP3 bitrate is known at extraction time (see ovox.audio), so the byte
    limit converts to seconds exactly. ``safety`` keeps a margin below the
    endpoint's hard limits.

    Args:
        max_bytes, max_seconds: The endpoint's size and duration limits.
        safety: Fraction of the limits to plan chunks at, in (0, 1].
        bitrate_bps: The CBR bitrate the audio is encoded at. Pass the same
            value used to extract the audio (see :class:`ovox.config.AudioConfig`)
            so the size estimate matches the real file size.
    """
    if max_bytes <= 0 or max_seconds <= 0:
        raise ValueError("max_bytes and max_seconds must be positive")
    if not 0 < safety <= 1:
        raise ValueError("safety must be in (0, 1]")
    if bitrate_bps <= 0:
        raise ValueError("bitrate_bps must be positive")
    from_bytes = max_bytes * 8.0 / bitrate_bps  # bytes -> bits -> seconds
    return min(from_bytes, max_seconds) * safety


def parse_silencedetect(stderr: str, duration: float | None = None) -> list[Silence]:
    """Parse ffmpeg's ``silencedetect`` log into silence intervals.

    Starts are clamped to 0 (ffmpeg can report a slightly negative start at the
    beginning of a file). ``duration`` clamps ends and closes a trailing
    ``silence_start`` that ffmpeg never finished reporting (silence running to
    the end of the file). Empty intervals are dropped; the result is sorted.
    """
    silences: list[Silence] = []
    pending: float | None = None
    for line in stderr.splitlines():
        if m := _SILENCE_START_RE.search(line):
            pending = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END_RE.search(line)) and pending is not None:
            silences.append(Silence(pending, float(m.group(1))))
            pending = None
    if pending is not None and duration is not None:
        silences.append(Silence(pending, duration))

    if duration is not None:
        silences = [Silence(s.start, min(duration, s.end)) for s in silences]
    return sorted((s for s in silences if s.end > s.start), key=lambda s: s.start)


def plan_chunks(
    duration: float,
    silences: list[Silence],
    max_len: float | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_CHUNK_SIZE_BYTES,
    max_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    safety: float = DEFAULT_SAFETY,
    overlap: float = DEFAULT_OVERLAP_S,
    search_window: float = DEFAULT_SEARCH_WINDOW_S,
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
) -> list[ChunkSpec]:
    """Split *duration* seconds of audio into chunks no longer than the limits.

    Each chunk ends at the center of the longest silence in a search window
    just before its planned end, so nothing is shared between the two chunks.
    Silences are clipped to the window, so a long pause that runs across the
    planned end still counts. A chunk is never cut before half of ``max_len``,
    which keeps short pauses from producing many tiny chunks. With no usable
    silence, the chunk is hard-cut at its planned end and the next chunk starts
    ``overlap`` seconds earlier, so a word split by the cut is heard whole once.
    The last chunk is never shorter than ``MIN_TAIL_S`` (unless the whole
    file is).

    Args:
        duration: Total audio duration in seconds.
        silences: Silence intervals from :func:`detect_silence`.
        max_len: Hard cap on chunk length in seconds. Falls back to
            ``max_chunk_seconds(max_bytes, max_seconds, safety)``.
        max_bytes, max_seconds, safety: Endpoint limits used when *max_len*
            is not given.
        bitrate_bps: CBR bitrate used to convert the size limit to seconds;
            pass the same value the audio is encoded at.
        overlap: Seconds of repeated audio at the start of hard-cut chunks.
            At most half of the chunk length.
        search_window: How far back from the planned end to look for silence.

    Returns:
        The chunks covering the whole file, in order (empty for zero duration).
    """
    if duration < 0:
        raise ValueError("duration must be >= 0")
    if duration == 0:
        return []
    if max_len is None:
        max_len = max_chunk_seconds(max_bytes, max_seconds, safety, bitrate_bps)
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap > max_len / 2:
        raise ValueError(f"overlap must be at most half of max_len ({max_len / 2:g}s)")
    if search_window <= 0:
        raise ValueError("search_window must be positive")

    specs: list[ChunkSpec] = []
    start = 0.0
    ov = 0.0
    while start < duration:
        end, next_ov = _cut_point(start, max_len, duration, silences, overlap, search_window)
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
        duration: Length of the audio in seconds. Used to close and clamp
            silence running to the end of the file, and to scale the timeout.
            Probed with ffprobe when not given.
        noise: ffmpeg volume threshold below which audio counts as silence.
        min_silence: Ignore silences shorter than this many seconds.
    """
    if duration is None:
        duration = _probe_duration(audio_file)
    timeout = SILENCE_DETECT_TIMEOUT_BASE_S + SILENCE_DETECT_TIMEOUT_PER_SECOND * duration
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
    # Goes through audio.run_ffmpeg so it has a timeout and can be faked in tests.
    result = audio.run_ffmpeg(
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
        PROBE_TIMEOUT_S,
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
    max_len: float,
    duration: float,
    silences: list[Silence],
    overlap: float,
    search_window: float,
) -> tuple[float, float]:
    """Choose where the chunk starting at *start* ends.

    Returns:
        ``(end, next_overlap)``: the end of the current chunk and how much of
        the next chunk's head repeats it (0 when a silence was used, since
        nothing is shared then).
    """
    planned_end = start + max_len
    if planned_end >= duration:
        return duration, 0.0
    # Pull the cut earlier rather than leave a sliver at the end, as long as
    # the chunk still makes progress after the overlap.
    if duration - planned_end < MIN_TAIL_S and duration - MIN_TAIL_S - overlap > start:
        planned_end = duration - MIN_TAIL_S

    window_start = max(start + max_len / 2, planned_end - search_window)
    best: Silence | None = None
    for s in silences:
        clipped = Silence(max(s.start, window_start), min(s.end, planned_end))
        length = clipped.end - clipped.start
        if length >= MIN_SILENCE_S and (best is None or length > best.end - best.start):
            best = clipped
    if best is not None:
        return (best.start + best.end) / 2.0, 0.0
    return planned_end, overlap
