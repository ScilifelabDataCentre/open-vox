"""Command-line interface for ovox.

Usage::

    ovox --version
    ovox version [--json]
    ovox check [--json]
    ovox chunk <file> --output-dir <dir> [--dry-run | --plan-only] [--json]

New commands (e.g. ``transcribe``) are added as subcommands in
``build_parser`` and dispatched from ``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ovox CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        result: int = handler(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return result


if __name__ == "__main__":
    sys.exit(main())
