"""Command-line interface for ovox.

Usage::

    ovox --version
    ovox version [--json]
    ovox check [--json]
    ovox chunk <file> --output-dir <dir> [--dry-run | --plan-only] [--json]
    ovox transcribe <file> [-o OUT.txt] [--dry-run] [--resume] [--json]

New commands are added as subcommands in ``build_parser`` and dispatched from
``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import platform
import re
import signal
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ovox import __version__
from ovox.chunking import (
    DEFAULT_MAX_CHUNK_SECONDS,
    DEFAULT_MAX_CHUNK_SIZE_BYTES,
    DEFAULT_OVERLAP_S,
    DEFAULT_SAFETY,
    DEFAULT_SEARCH_WINDOW_S,
)

PROG = "ovox"

# Byte multipliers for size suffixes. Decimal by default (K=1000, M=10**6, ...);
# an "i" (KiB, MiB, ...) selects the binary table (1024-based).
_BYTES_DECIMAL = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15, "E": 10**18}
_BYTES_BINARY = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5, "E": 1024**6}


def _human_bytes(value: int) -> str:
    """Format a byte count as a short human size (powers of 1024): 26214400 -> '25 MiB'."""
    if value < 1024:
        return f"{value} B"
    v = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        v /= 1024
        if v < 1024:
            return f"{v:g} {unit}"
    return f"{v:g} TiB"


def _size_default(value: int) -> str:
    """Format a byte limit for help text, e.g. ``26214400 = 25 MiB``."""
    human = _human_bytes(value)
    return f"{value} = {human}" if human != f"{value} B" else f"{value}"


def _parse_bytes(value: str) -> int:
    """Parse a byte size with an optional K/M/G/T/P/E (decimal) or KiB/MiB/... (binary) suffix.

    ``25M`` -> 25000000, ``25MiB`` -> 26214400, ``26214400`` -> 26214400.
    """
    m = re.fullmatch(r"\s*(\d+)([KMGTPE]i?B?)?\s*", value, re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid byte size {value!r}; use bytes or a suffix like 25M or 25MiB"
        )
    number = int(m.group(1))
    unit = (m.group(2) or "").upper()
    if not unit:
        return number
    # An "i" (KiB, MiB, ...) means binary (1024-based) units; otherwise decimal.
    table = _BYTES_BINARY if "I" in unit else _BYTES_DECIMAL
    return number * table[unit[0]]


def version_info() -> dict[str, str]:
    """Return version details for ovox and its runtime environment."""
    return {
        "ovox": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def _version_string() -> str:
    return f"{PROG} {__version__} (Python {platform.python_version()})"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Open Vox: transcribe audio and video files to text.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=_version_string(),
        help="show the ovox version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    version_parser = subparsers.add_parser(
        "version",
        help="show version information for ovox and its environment",
        description="Show version information for ovox and its environment.",
    )
    version_parser.add_argument(
        "--json",
        action="store_true",
        help="print version information as JSON",
    )
    version_parser.set_defaults(handler=_cmd_version)

    # Add check command
    check_parser = subparsers.add_parser(
        "check",
        help="check STT API connection and authentication",
        description="Check if the STT API is reachable and credentials are valid.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="print result as JSON",
    )
    check_parser.set_defaults(handler=_cmd_check)

    # Add chunk command. allow_abbrev=False keeps flags unambiguous (so e.g. the
    # old --safety cannot prefix-match --safety-factor).
    chunk_parser = subparsers.add_parser(
        "chunk",
        allow_abbrev=False,
        help="split a file into chunks that fit the STT endpoint's limits",
        description=(
            "Split an audio or video file into audio chunks (mono 16 kHz MP3, the format "
            "sent for transcription) that fit the configured size and duration limits. "
            "Cuts are placed at silences where possible. Use --dry-run to preview."
        ),
    )
    chunk_parser.add_argument("file", metavar="FILE", help="audio or video file to split")

    output_group = chunk_parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help=(
            "directory for the chunk files and manifest (required unless --dry-run). "
            "Existing chunk files with the same name are overwritten; the directory is "
            "created if missing"
        ),
    )
    mode_group = output_group.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="print the chunk plan without writing any files",
    )
    mode_group.add_argument(
        "--plan-only",
        action="store_true",
        help="write the manifest, without writing the chunk files",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="print the plan/result as JSON instead of text",
    )

    limits_group = chunk_parser.add_argument_group("limits")
    limits_group.add_argument(
        "--max-size-bytes",
        type=_parse_bytes,
        metavar="BYTES",
        help=(
            "maximum chunk file size, with an optional K/M/... suffix, e.g. 25M "
            f"(env: CHUNK_MAX_SIZE_BYTES, default: {_size_default(DEFAULT_MAX_CHUNK_SIZE_BYTES)})"
        ),
    )
    limits_group.add_argument(
        "--max-duration-s",
        type=float,
        metavar="SECONDS",
        help=(
            "maximum chunk duration in seconds "
            f"(env: CHUNK_MAX_DURATION_S, default: {DEFAULT_MAX_CHUNK_SECONDS:g})"
        ),
    )
    limits_group.add_argument(
        "--safety-factor",
        type=float,
        metavar="FACTOR",
        help=(
            "plan chunks at this fraction of the limits, "
            f"0 < FACTOR <= 1 (default: {DEFAULT_SAFETY})"
        ),
    )

    splitting_group = chunk_parser.add_argument_group("splitting")
    splitting_group.add_argument(
        "--split-method",
        choices=("silence", "fixed"),
        help="cut at silences (default) or at the maximum length",
    )
    splitting_group.add_argument(
        "--search-window-s",
        type=float,
        metavar="SECONDS",
        help=(
            "how far before the maximum length to look for a silence "
            f"(default: {DEFAULT_SEARCH_WINDOW_S:g})"
        ),
    )
    splitting_group.add_argument(
        "--overlap-s",
        type=float,
        metavar="SECONDS",
        help=(
            "overlap added where a cut is not at a silence, i.e. at the end of a hard cut "
            f"and on every cut with --split-method fixed (default: {DEFAULT_OVERLAP_S:g})"
        ),
    )
    chunk_parser.set_defaults(handler=_cmd_chunk)

    # Add transcribe command: the direct-mode end-to-end run.
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        allow_abbrev=False,
        help="transcribe an audio or video file to text",
        description=(
            "Transcribe an audio or video file with the configured STT endpoint. "
            "The file is split into chunks that fit the endpoint's limits, each chunk "
            "is sent with retries, and the results are merged into a .txt transcript "
            "next to the input (or -o). Chunk audio, the manifest and the per-chunk "
            "results are kept in a work folder next to the input, so an interrupted "
            "run can be resumed with --resume (or simply re-run)."
        ),
    )
    transcribe_parser.add_argument("file", metavar="FILE", help="audio or video file to transcribe")
    transcribe_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="TXT",
        help="write the transcript to this .txt file (default: next to the input)",
    )
    transcribe_parser.add_argument(
        "--workdir",
        type=Path,
        metavar="DIR",
        help="directory for the chunk files, manifest and chunk results "
        "(default: a hidden folder next to the input)",
    )
    transcribe_mode = transcribe_parser.add_mutually_exclusive_group()
    transcribe_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="run the local steps (validate, cut chunks) and print the plan, "
        "without making any STT requests",
    )
    transcribe_mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse the chunks a previous run cut (check that the source and "
        "settings still match) and only transcribe the chunks without a result",
    )
    transcribe_parser.add_argument(
        "--json",
        action="store_true",
        help="print the result (or the plan, with --dry-run) as JSON; "
        "progress goes to stderr in both modes",
    )
    transcribe_parser.set_defaults(handler=_cmd_transcribe)

    return parser


def _cmd_version(args: argparse.Namespace) -> int:
    info = version_info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        width = max(len(key) for key in info)
        for key, value in info.items():
            print(f"{key:<{width}}  {value}")
    return 0


def print_check_result(result: Mapping[str, Any], as_json: bool = False) -> None:
    """Print the result of ``STTClient.check_connection`` to stdout.

    In JSON mode the result is printed as-is. In text mode the endpoint and
    configured model are always shown, followed by the outcome and duration.
    """
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(result.get("message", ""))
    error = result.get("error")
    if error is not None:
        print(f"Error: {error}")
    print(f"Endpoint: {result.get('endpoint', '')}")
    print(f"Model (configured): {result.get('model', '')}")
    print(f"Duration: {result.get('duration_ms', 0)} ms")


def _cmd_check(args: argparse.Namespace) -> int:
    """Handle the check command to verify STT API connection and authentication.

    Args:
        args: The parsed command line arguments.

    Returns:
        An integer exit code (0 for success, non-zero for failure).
    """

    # Import here to avoid circular imports
    from ovox.api import STTClient

    async def _check_async() -> int:
        try:
            client = STTClient()
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1

        async with client:
            result = await client.check_connection()
            print_check_result(result, as_json=args.json)
            return 0 if result["success"] else 1

    # Run the async check function and return the exit code
    exit_code: int = asyncio.run(_check_async())
    return exit_code


def print_chunk_plan(plan: Any, *, mode: str, as_json: bool) -> None:
    """Print a chunk plan to stdout (JSON, or a human-readable table).

    *plan* is an :class:`ovox.plan.ChunkPlan`. Its ``chunks`` are
    :class:`ChunkEntry` objects, its ``settings`` are :class:`Setting` objects,
    and its ``input``/``totals`` are plain dicts. *mode* is ``"dry-run"``,
    ``"plan-only"`` or ``"real"`` and only affects the closing note.
    """
    from ovox.plan import plan_manifest_text

    if as_json:
        print(plan_manifest_text(plan))
        return

    i = plan.input
    rate = i["sample_rate"] if i["sample_rate"] is not None else "?"
    ch = i["channels"] if i["channels"] is not None else "?"
    print(f"File: {i['file']}")
    print(
        f"  format={i['format']} codec={i['codec']} duration={i['duration']:.1f}s "
        f"rate={rate}Hz channels={ch} audio_track={i['audio_track']}"
    )
    s = plan.settings
    print(
        "Settings: "
        f"max_size={s['max_chunk_size_bytes'].value}B [{s['max_chunk_size_bytes'].source}] "
        f"max_duration={s['max_chunk_duration_s'].value}s [{s['max_chunk_duration_s'].source}] "
        f"overlap={s['overlap_s'].value}s [{s['overlap_s'].source}] "
        f"search_window={s['search_window_s'].value}s "
        f"safety={s['safety'].value} method={plan.split_method}"
    )
    print(
        "Audio: "
        f"rate={s['sample_rate'].value}Hz channels={s['channels'].value} "
        f"bitrate={s['bitrate_kbps'].value}kbps [{s['bitrate_kbps'].source}]"
    )
    print(f"{'idx':>4}  {'start':>8}  {'end':>8}  {'dur':>7}  {'size':>10}  name")
    for c in plan.chunks:
        size = c.actual_size if c.actual_size is not None else c.estimated_size
        row = f"{c.index:>4}  {c.start:8.1f}  {c.end:8.1f}  {c.duration:7.1f}  {size:>10}  "
        print(row + c.name)
        print(f"{'':>4}  reason: {c.reason}")
    t = plan.totals
    print(f"Totals: {t['num_chunks']} chunk(s), {t['total_audio_length']:.1f}s of audio")
    for w in plan.warnings:
        print(f"Warning: {w}")
    if mode == "dry-run":
        print("Dry run: nothing was written.")
    elif mode == "plan-only":
        print("Plan only: wrote the manifest, no chunk files.")


def _cmd_chunk(args: argparse.Namespace) -> int:
    """Handle the chunk command.

    By default it writes the chunks and a manifest. ``--dry-run`` only plans and
    prints, writing nothing. ``--plan-only`` writes just the manifest.
    """
    from ovox.config import AudioConfig, ChunkConfig
    from ovox.media import MediaError, validate_input
    from ovox.plan import build_plan, plan_manifest_text, write_chunks

    # --output-dir is only needed when something is written.
    if args.output_dir is None and not args.dry_run:
        print("Error: --output-dir is required unless --dry-run is given", file=sys.stderr)
        return 1

    try:
        info = validate_input(args.file)
        config = ChunkConfig.from_env()
        audio_config = AudioConfig.from_env()
        plan = build_plan(
            info,
            config,
            audio_config,
            max_bytes=args.max_size_bytes,
            max_seconds=args.max_duration_s,
            overlap=args.overlap_s,
            search_window=args.search_window_s,
            safety=args.safety_factor,
            split_method=args.split_method,
        )
    except MediaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Catches both invalid input and invalid deployment settings (e.g. a
        # bitrate that is not valid for the sample rate), so the user sees the
        # actionable message, not a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        # A dry run is safe by convention: it only tests and prints, writing nothing.
        mode = "dry-run"
    else:
        out_dir = args.output_dir
        assert out_dir is not None  # guaranteed: enforced at the top of the handler
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.plan_only:
            try:
                write_chunks(plan, out_dir, audio_config)
            except MediaError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        # The manifest reflects the plan (and the written sizes, if any chunks were cut).
        (out_dir / "ovox-chunk-manifest.json").write_text(plan_manifest_text(plan))
        mode = "plan-only" if args.plan_only else "real"

    print_chunk_plan(plan, mode=mode, as_json=args.json)
    return 0


def print_transcribe_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    """Print the outcome of a transcription run to stdout.

    JSON mode prints the result dict as-is. Text mode prints the output files
    and a short summary (including per-stage timings, which is what the
    direct-mode benchmarks are built from).
    """
    if as_json:
        # default=str renders the output paths; the text modes show them as-is.
        print(json.dumps(result, indent=2, default=str))
        return
    print(f"Transcript: {result['text_file']}")
    segments_file = result.get("segments_file")
    if segments_file is not None:
        print(f"Segments:   {segments_file}")
    print(
        f"Chunks:   {result['chunks_transcribed']} transcribed this run, "
        f"{result['chunks_total']} total"
    )
    if result.get("resumed"):
        print(f"Workdir:  {result['workdir']} (reused)")
    else:
        print(f"Workdir:  {result['workdir']}")
    timings: Mapping[str, float] = result.get("timings", {})
    if timings:
        parts = [f"{name}={seconds:.2f}s" for name, seconds in timings.items()]
        total = sum(timings.values())
        print(f"Timings:  {' '.join(parts)} (total {total:.2f}s)")


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle the transcribe command (direct mode, end to end).

    ``--dry-run`` runs the local stages (validate, cut) and prints the plan,
    making no STT requests; a later real run resumes from the same work
    folder. A normal run sends the missing chunks to the endpoint and writes
    the transcript. Failures are actionable messages with non-zero exit codes:
    authentication errors point at ``ovox check``, and an interrupted run says
    how many chunks are done and how to resume.
    """
    from ovox.config import AudioConfig, ChunkConfig, SttConfig, TranscribeConfig
    from ovox.media import MediaError
    from ovox.pipeline import PipelineError, WorkdirError, prepare_for_dry_run, transcribe_file

    def _configs() -> tuple[ChunkConfig, AudioConfig, SttConfig, TranscribeConfig]:
        return (
            ChunkConfig.from_env(),
            AudioConfig.from_env(),
            SttConfig.from_env(),
            TranscribeConfig.from_env(),
        )

    try:
        if args.dry_run:
            chunk_config, audio_config, stt, _ = _configs()
            result = prepare_for_dry_run(
                args.file,
                chunk_config=chunk_config,
                audio_config=audio_config,
                stt=stt,
                workdir=args.workdir,
                resume=args.resume,
            )
            if args.json:
                print(json.dumps(result["plan"], indent=2))
            else:
                # mode="real" keeps the plan printer's closing note off: unlike
                # ovox chunk --dry-run, this dry run *does* write the chunks
                # and the manifest (into the work folder, for the resume).
                print_chunk_plan_from_dict(result["plan"], mode="real")
                print(
                    f"Chunks already transcribed: {result['chunks_transcribed']}"
                    f"/{result['chunks_transcribed'] + result['chunks_pending']}"
                )
                print(f"Workdir: {result['workdir']}")
                print("Dry run: no STT requests were made. Run without --dry-run to transcribe.")
            return 0
        chunk_config, audio_config, stt, transcribe_config = _configs()
        result = transcribe_file(
            args.file,
            chunk_config=chunk_config,
            audio_config=audio_config,
            stt=stt,
            transcribe_config=transcribe_config,
            workdir=args.workdir,
            output=args.output,
            resume=args.resume,
        )
    except MediaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (WorkdirError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Invalid deployment settings (e.g. an impossible STT_MAX_CONCURRENCY).
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_transcribe_result(result, as_json=args.json)
    return 0


def print_chunk_plan_from_dict(plan: Mapping[str, Any], *, mode: str) -> None:
    """Print a plan dict (as read from the manifest) the way a live plan is
    printed, so a dry run of ``ovox transcribe`` matches ``ovox chunk``."""
    from ovox.plan import ChunkEntry, ChunkPlan, Setting

    entries = [
        ChunkEntry(
            index=c["index"],
            start=c["start"],
            end=c["end"],
            duration=c["duration"],
            estimated_size=c["estimated_size"],
            name=c["name"],
            reason=c.get("reason", ""),
            overlap_before=c.get("overlap_before", 0.0),
            actual_size=c.get("actual_size"),
            actual_duration=c.get("actual_duration"),
        )
        for c in plan["chunks"]
    ]
    settings = {
        name: Setting(
            value=s.get("value"), source=str(s.get("source", "")), origin=str(s.get("origin", ""))
        )
        for name, s in plan["settings"].items()
    }
    live = ChunkPlan(
        input=dict(plan["input"]),
        settings=settings,
        chunks=entries,
        warnings=list(plan.get("warnings", [])),
        totals=dict(plan.get("totals", {})),
        split_method=str(plan.get("split_method", "silence")),
    )
    print_chunk_plan(live, mode=mode, as_json=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ovox CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    # SIGTERM (e.g. from a job manager) is turned into SystemExit so the
    # cleanup that runs through the normal exit path (work folders, temp
    # files, in-flight requests) also runs when the process is killed.
    def _term(signum: int, _frame: Any) -> None:
        raise SystemExit(143)

    with contextlib.suppress(ValueError, OSError):
        # Not on the main thread (tests) or the platform has no SIGTERM.
        signal.signal(signal.SIGTERM, _term)

    try:
        result: int = handler(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        print("Interrupted (signal).", file=sys.stderr)
        return code
    return result


if __name__ == "__main__":
    sys.exit(main())
