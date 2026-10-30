"""Command-line interface for ovox.

Usage::

    ovox --version
    ovox version [--json]

New commands (e.g. ``transcribe``) are added as subcommands in
``build_parser`` and dispatched from ``main``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Sequence

from ovox import __version__

PROG = "ovox"


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


def _cmd_check(args: argparse.Namespace) -> int:
    """Handle the check command to verify STT API connection and authentication.

    Args:
        args: The parsed command line arguments.

    Returns:
        An integer exit code (0 for success, non-zero for failure).
    """

    # Import here to avoid circular imports
    from ovox.api import STTClient

    async def _check_async():
        async with STTClient() as client:
            result = await client.check_connection()

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(result["message"])

            return 0 if result["success"] else 1

    # Run the async check function
    return asyncio.run(_check_async())


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
