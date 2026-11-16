"""Audio extraction.

Converts the first audio stream of a media file to a compact MP3 suitable
for sending to an STT endpoint: mono, 16 kHz, 32 kbps (about 14 MB per hour
of audio). Everything runs through ffmpeg; no other runtime dependencies.
"""

import contextlib
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from ovox.media import MediaError

# Extraction normally runs well under realtime. Allow 2x realtime plus a
# fixed overhead, with a floor for short files and a default when the input
# duration is unknown.
TIMEOUT_BASE_S = 60.0
TIMEOUT_PER_SECOND_OF_AUDIO = 2.0
TIMEOUT_MIN_S = 120.0
TIMEOUT_DEFAULT_S = 600.0

OUTPUT_NAME = "audio.mp3"
STDERR_TAIL_LINES = 5
# Mono MP3 at 32 kbps, about 14 MB per hour of audio. Shared with ovox.chunking.
MP3_BITRATE_BPS = 32_000


class AudioExtractionError(MediaError):
    """Audio could not be extracted from the input file."""


@contextlib.contextmanager
def workdir() -> Iterator[Path]:
    """Yield a temporary directory that is removed on exit.

    Cleanup runs on success, on exceptions and on Ctrl-C, because
    KeyboardInterrupt and SystemExit propagate through the context manager.
    """
    with tempfile.TemporaryDirectory(prefix="ovox-") as name:
        yield Path(name)


def extract_audio(source: Path, dest_dir: Path, *, duration: float | None = None) -> Path:
    """Extract the first audio stream of *source* as mono 16 kHz 32 kbps MP3.

    The result is written to ``dest_dir/audio.mp3`` and its path is returned.

    Args:
        source: A validated input file (see ovox.media).
        dest_dir: Directory to write the MP3 into, typically from workdir().
        duration: Input duration in seconds, used to scale the ffmpeg timeout.

    Raises:
        AudioExtractionError: ffmpeg failed, timed out, or produced no audio.
    """
    output = dest_dir / OUTPUT_NAME
    timeout = timeout_for(duration)
    try:
        result = run_ffmpeg(_ffmpeg_command(source, output), timeout)
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(
            f"ffmpeg timed out after {timeout:.0f}s extracting audio from: {source}"
        ) from exc

    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed while extracting audio from: {source}\n{stderr_tail(result.stderr)}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise AudioExtractionError(f"ffmpeg produced no audio from: {source}")
    return output


def _ffmpeg_command(source: Path, output: Path) -> list[str]:
    # An absolute path keeps names starting with "-" from being read as options.
    return [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{MP3_BITRATE_BPS // 1000}k",
        str(output),
    ]


def timeout_for(duration: float | None) -> float:
    if duration is None:
        return TIMEOUT_DEFAULT_S
    return max(TIMEOUT_MIN_S, TIMEOUT_BASE_S + TIMEOUT_PER_SECOND_OF_AUDIO * duration)


def run_ffmpeg(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg, making sure the child process is killed on timeout or interruption."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return subprocess.CompletedProcess(command, proc.returncode, stdout=out, stderr=err)


def stderr_tail(stderr: str, lines: int = STDERR_TAIL_LINES) -> str:
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(tail) if tail else "no error output"
