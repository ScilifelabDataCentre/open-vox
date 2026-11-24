"""Chunk planning and writing.

This is the one place that decides *where* the chunks go. :func:`build_plan`
does the work (extract audio, detect silence, plan the cuts) and returns a
:class:`ChunkPlan`; :func:`write_chunks` turns that plan into files. The
``ovox chunk`` command (``--dry-run``, ``--plan-only`` and a real run) all call
the same :func:`build_plan`, and real transcription will reuse it, so the plan
a preview prints is always an accurate description of what would be uploaded.

Planning is pure given its inputs (see :mod:`ovox.chunking`); the ffmpeg steps
(extraction, silence detection, cutting) are the only I/O.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ovox import audio, chunking
from ovox.audio import DEFAULT_BITRATE_BPS
from ovox.chunking import ChunkSpec, Silence, detect_silence, plan_chunks
from ovox.config import AudioConfig, ChunkConfig
from ovox.media import MediaInfo

# The normalized format every chunk is written in, and what real transcription
# uploads. Changing it changes how many minutes fit under a size limit.
CHUNK_FORMAT = "mp3"

# Where each setting came from, so the plan is self-documenting.
SOURCE_FLAG = "flag"
SOURCE_CONFIG = "config"  # resolved from .env / environment, or its default
SOURCE_DEFAULT = "default"  # a built-in value with no env/flag override


@dataclass(frozen=True)
class Setting:
    """A setting value plus where it came from (flag, env var, or default)."""

    value: object
    source: str
    origin: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "source": self.source, "origin": self.origin}


@dataclass
class ChunkEntry:
    """One planned (and, after writing, actual) chunk.

    ``start``, ``end`` and ``duration`` describe the chunk as *planned*;
    ``actual_size`` and ``actual_duration`` describe the file as *written*
    (both ``None`` until the chunk has been cut). ``actual_duration`` is
    measured by decoding the written file, so the manifest matches what is on
    disk rather than what an MP3 header claims.
    """

    index: int
    start: float
    end: float
    duration: float
    estimated_size: int
    name: str
    reason: str
    actual_size: int | None = None
    actual_duration: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "estimated_size": self.estimated_size,
            "actual_size": self.actual_size,
            "actual_duration": self.actual_duration,
            "name": self.name,
            "reason": self.reason,
        }


@dataclass
class ChunkPlan:
    """The full description of how a file will be split into chunks."""

    input: dict[str, object]
    settings: dict[str, Setting]
    chunks: list[ChunkEntry]
    warnings: list[str]
    totals: dict[str, object]
    split_method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.input,
            "settings": {name: s.to_dict() for name, s in self.settings.items()},
            "split_method": self.split_method,
            "chunks": [c.to_dict() for c in self.chunks],
            "warnings": self.warnings,
            "totals": self.totals,
        }


def estimate_mp3_bytes(duration: float, bitrate_bps: int = DEFAULT_BITRATE_BPS) -> int:
    """Estimated size of *duration* seconds of MP3 at *bitrate_bps*, in bytes.

    CBR MP3 size is predictable from duration, which the planner relies on.
    """
    return math.ceil(duration * bitrate_bps / 8.0)


def chunk_name(prefix: str, index: int, start: float, end: float) -> str:
    """Build a consistent, sortable chunk file name.

    Example: ``lecture_chunk_003_754.5-910.2.mp3``. Times are rounded to 0.1 s
    and the index pad adapts to the total width so names sort in chunk order.
    """
    return f"{prefix}_chunk_{index:02d}_{round(start, 1):g}-{round(end, 1):g}.{CHUNK_FORMAT}"


def build_plan(
    info: MediaInfo,
    chunk_config: ChunkConfig,
    audio_config: AudioConfig,
    *,
    max_bytes: int | None = None,
    max_seconds: float | None = None,
    overlap: float | None = None,
    search_window: float | None = None,
    safety: float | None = None,
    split_method: str | None = None,
) -> ChunkPlan:
    """Decide where the chunks of *info* go and return the plan.

    This is the single function used by the ``ovox chunk`` preview, the real
    ``ovox chunk`` run, and (in due course) real transcription, so a preview is
    always an accurate description of what would be uploaded.

    Args:
        info: Validated media info (see :func:`ovox.media.validate_input`).
        chunk_config: Per-target limits from config/env.
        audio_config: The upload format (sample rate, channels, bitrate). The
            same values are used to encode the audio and to estimate its size,
            so the plan and the files agree.
        max_bytes, max_seconds, overlap, search_window, safety: Optional
            overrides (normally from CLI flags) that win over config.
        split_method: ``"silence"`` (default) or ``"fixed"``. ``"fixed"`` hard
            cuts at the maximum length and never looks for silence.

    Raises:
        ovox.media.MediaError: ffmpeg/ffprobe failed.
        ValueError: invalid arguments or no usable audio.
    """
    if split_method is None:
        split_method = "silence"
    if split_method not in ("silence", "fixed"):
        raise ValueError(f"split_method must be 'silence' or 'fixed', got: {split_method!r}")

    with audio.workdir() as d:
        mp3 = audio.extract_audio(
            info.path,
            d,
            duration=info.duration,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            bitrate_bps=audio_config.bitrate_bps,
        )
        duration = info.duration
        if duration is None:
            duration = chunking._probe_duration(mp3)
        if duration <= 0:
            raise ValueError(f"No audio to split: {info.path}")

        silences = detect_silence(mp3, duration) if split_method == "silence" else []
        specs = plan_chunks(
            duration,
            silences,
            max_bytes=max_bytes if max_bytes is not None else chunk_config.max_chunk_size_bytes,
            max_seconds=max_seconds
            if max_seconds is not None
            else chunk_config.max_chunk_duration_s,
            overlap=overlap if overlap is not None else chunking.DEFAULT_OVERLAP_S,
            search_window=search_window
            if search_window is not None
            else chunking.DEFAULT_SEARCH_WINDOW_S,
            safety=safety if safety is not None else chunk_config.safety,
            bitrate_bps=audio_config.bitrate_bps,
        )

    prefix = info.path.stem
    entries = _entries(specs, prefix, silences, audio_config.bitrate_bps)
    settings = _settings(
        chunk_config,
        audio_config,
        max_bytes,
        max_seconds,
        overlap,
        search_window,
        safety,
        split_method,
    )
    warnings = _warnings(info, entries, silences, split_method)

    return ChunkPlan(
        input=_input_dict(info, duration),
        settings=settings,
        chunks=entries,
        warnings=warnings,
        totals={
            "num_chunks": len(entries),
            "total_audio_length": duration,
            "total_estimated_size": sum(e.estimated_size for e in entries),
        },
        split_method=split_method,
    )


def write_chunks(plan: ChunkPlan, output_dir: Path, audio_config: AudioConfig) -> list[Path]:
    """Write the chunks of *plan* into *output_dir* and record their size and length.

    Cuts each planned range from the original source file with ffmpeg (see
    :func:`ovox.audio.cut_audio`), re-encoding to the format in *audio_config*,
    so the files written are exactly the ones a real transcription would upload
    and match the plan's size estimates. Each written chunk is then measured by
    decoding (``actual_duration``), so the manifest records what is actually on
    disk rather than what the plan assumed. Returns the written paths in plan
    order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(str(plan.input["file"]))
    paths: list[Path] = []
    for entry in plan.chunks:
        out = output_dir / entry.name
        audio.cut_audio(
            source,
            out,
            entry.start,
            entry.end,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            bitrate_bps=audio_config.bitrate_bps,
        )
        entry.actual_size = out.stat().st_size
        entry.actual_duration = audio.decoded_duration(out, sample_rate=audio_config.sample_rate)
        paths.append(out)

    # Check the real size after writing: the estimate is based on the CBR rate,
    # but container overhead and variable-bitrate drift can push a chunk over.
    limit = cast(int, plan.settings["max_chunk_size_bytes"].value)
    for entry in plan.chunks:
        if entry.actual_size is not None and entry.actual_size > limit:
            plan.warnings.append(
                f"chunk {entry.index} is over the API size limit "
                f"({entry.actual_size} > {limit} bytes); it may be rejected."
            )
    return paths


def plan_manifest_text(plan: ChunkPlan) -> str:
    """Render a plan as the JSON written to the manifest (and shown by ``--json``).

    The same structure is used whether or not the run wrote files; which chunks
    were actually written is told by ``actual_size`` (null when nothing was cut).
    """
    return json.dumps(plan.to_dict(), indent=2)


# --- internals ------------------------------------------------------------------


def _input_dict(info: MediaInfo, duration: float) -> dict[str, object]:
    return {
        "file": str(info.path),
        "format": info.format_name,
        "codec": info.audio_codec,
        "duration": duration,
        "sample_rate": info.audio_sample_rate,
        "channels": info.audio_channels,
        "audio_track": info.audio_track_index,
    }


def _entries(
    specs: list[ChunkSpec], prefix: str, silences: list[Silence], bitrate_bps: int
) -> list[ChunkEntry]:
    entries: list[ChunkEntry] = []
    for i, spec in enumerate(specs):
        entries.append(
            ChunkEntry(
                index=i,
                start=spec.start,
                end=spec.end,
                duration=spec.duration,
                estimated_size=estimate_mp3_bytes(spec.duration, bitrate_bps),
                name=chunk_name(prefix, i, spec.start, spec.end),
                reason=_reason(i, spec, specs, silences),
            )
        )
    return entries


def _reason(index: int, spec: ChunkSpec, specs: list[ChunkSpec], silences: list[Silence]) -> str:
    if index == len(specs) - 1:
        return "end of audio"
    if index > 0:
        # A non-first chunk re-sends its head only after a hard cut.
        return "hard cut (no silence found)" if spec.overlap_before > 0 else "cut on silence"
    # Chunk 0 never has overlap, so decide by whether its end sits in a silence.
    if any(s.start < spec.end < s.end for s in silences):
        return "cut on silence"
    return "hard cut (no silence found)"


def _settings(
    config: ChunkConfig,
    audio_config: AudioConfig,
    max_bytes: int | None,
    max_seconds: float | None,
    overlap: float | None,
    search_window: float | None,
    safety: float | None,
    split_method: str,
) -> dict[str, Setting]:
    # Precedence: a CLI flag (non-None override) wins, else the value resolved
    # from config/env, else the built-in default. The origin is the env name so
    # the user can point at the setting that produced a non-flag value.
    def pick(
        override: int | float | None,
        resolved: int | float,
        default: int | float,
        env_name: str,
        flag: str,
    ) -> Setting:
        if override is not None:
            return Setting(override, SOURCE_FLAG, flag)
        source = SOURCE_CONFIG if resolved != default else SOURCE_DEFAULT
        return Setting(resolved, source, env_name)

    return {
        "max_chunk_size_bytes": pick(
            max_bytes,
            config.max_chunk_size_bytes,
            chunking.DEFAULT_MAX_CHUNK_SIZE_BYTES,
            "CHUNK_MAX_SIZE_BYTES",
            "--max-size-bytes",
        ),
        "max_chunk_duration_s": pick(
            max_seconds,
            config.max_chunk_duration_s,
            chunking.DEFAULT_MAX_CHUNK_SECONDS,
            "CHUNK_MAX_DURATION_S",
            "--max-duration-s",
        ),
        "overlap_s": pick(
            overlap,
            chunking.DEFAULT_OVERLAP_S,
            chunking.DEFAULT_OVERLAP_S,
            "CHUNK_OVERLAP_S",
            "--overlap-s",
        ),
        "search_window_s": pick(
            search_window,
            chunking.DEFAULT_SEARCH_WINDOW_S,
            chunking.DEFAULT_SEARCH_WINDOW_S,
            "CHUNK_SEARCH_WINDOW_S",
            "--search-window-s",
        ),
        "safety": pick(
            safety,
            config.safety,
            chunking.DEFAULT_SAFETY,
            "CHUNK_SAFETY",
            "--safety-factor",
        ),
        "split_method": Setting(
            split_method, SOURCE_FLAG if split_method == "fixed" else SOURCE_DEFAULT
        ),
        # The upload format. No CLI override exists; it comes from config/env
        # or the built-in default. It determines both the encoded audio and the
        # size estimates, so it is reported for traceability.
        "sample_rate": Setting(
            audio_config.sample_rate,
            SOURCE_CONFIG
            if audio_config.sample_rate != audio.DEFAULT_SAMPLE_RATE
            else SOURCE_DEFAULT,
            "AUDIO_SAMPLE_RATE",
        ),
        "channels": Setting(
            audio_config.channels,
            SOURCE_CONFIG if audio_config.channels != audio.DEFAULT_CHANNELS else SOURCE_DEFAULT,
            "AUDIO_CHANNELS",
        ),
        "bitrate_kbps": Setting(
            audio_config.bitrate_bps // 1000,
            SOURCE_CONFIG
            if audio_config.bitrate_bps != audio.DEFAULT_BITRATE_BPS
            else SOURCE_DEFAULT,
            "AUDIO_BITRATE_KBPS",
        ),
    }


def _warnings(
    info: MediaInfo,
    entries: list[ChunkEntry],
    silences: list[Silence],
    split_method: str,
) -> list[str]:
    warnings: list[str] = []
    if split_method == "silence" and not silences:
        warnings.append(
            f"No silence found in {info.path.name}; all cuts are at the maximum length."
        )
    if len(entries) == 1:
        warnings.append("No split was needed; the input fits the limits in a single chunk.")
    if not entries:
        warnings.append("No chunks to write (input has no usable audio).")
    return warnings
