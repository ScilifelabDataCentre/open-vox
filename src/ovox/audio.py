"""Audio extraction.

Converts the first audio stream of a media file to a compact, constant-bitrate
MP3 suitable for sending to an STT endpoint. The sample rate, channel count and
bitrate are deployment settings (see :mod:`ovox.config`: ``AUDIO_SAMPLE_RATE``,
``AUDIO_CHANNELS``, ``AUDIO_BITRATE_KBPS``); the built-in defaults are mono,
16 kHz, 64 kbps (about 28.8 MB per hour of audio). Everything runs through
ffmpeg; no other runtime dependencies.
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

# Built-in defaults for the upload format. They are deployment settings
# (``AUDIO_SAMPLE_RATE``, ``AUDIO_CHANNELS``, ``AUDIO_BITRATE_KBPS``); the
# config layer resolves them and passes the result to these functions, so a
# run never mixes the encoded rate with a different planning rate.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_BITRATE_BPS = 64_000  # 64 kbps, about 28.8 MB per hour of audio


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


def extract_audio(
    source: Path,
    dest_dir: Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
    duration: float | None = None,
) -> Path:
    """Extract the first audio stream of *source* as a constant-bitrate MP3.

    The audio is converted to *channels* channel(s) at *sample_rate* Hz and
    encoded as CBR MP3 at *bitrate_bps*. The result is written to
    ``dest_dir/audio.mp3`` and its path is returned.

    Args:
        source: A validated input file (see ovox.media).
        dest_dir: Directory to write the MP3 into, typically from workdir().
        sample_rate, channels, bitrate_bps: The upload format. Pass the values
            resolved from config (see :class:`ovox.config.AudioConfig`) so the
            encoding and the chunk-size planning agree.
        duration: Input duration in seconds, used to scale the ffmpeg timeout.

    Raises:
        AudioExtractionError: ffmpeg failed, timed out, or produced no audio.
    """
    output = dest_dir / OUTPUT_NAME
    timeout = timeout_for(duration)
    try:
        command = _ffmpeg_command(source, output, sample_rate, channels, bitrate_bps)
        result = run_ffmpeg(command, timeout)
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


def _ffmpeg_command(
    source: Path, output: Path, sample_rate: int, channels: int, bitrate_bps: int
) -> list[str]:
    # Constant bitrate (CBR), not variable: the size formula in ovox.plan and
    # ovox.chunking only works because size is predictable from duration.
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
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_bps // 1000}k",
        str(output),
    ]


def cut_audio(
    source: Path,
    output: Path,
    start: float,
    end: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bitrate_bps: int = DEFAULT_BITRATE_BPS,
) -> Path:
    """Cut the range ``[start, end)`` of *source* into *output* as a CBR MP3.

    Used by ``ovox chunk`` to write each planned chunk (and, in real
    transcription, to produce the same chunks that get uploaded). Each chunk is
    produced by decoding the requested range from the original *source* and
    re-encoding it to the upload format (mono, *sample_rate* Hz, CBR MP3 at
    *bitrate_bps*). The cut is positioned with an input seek (``-ss`` before
    ``-i``) and an exact ``-t`` duration, so it lands at the planned sample
    rather than on an MP3 frame boundary.

    This is deliberately not a stream copy (``-c copy``). Stream-copying an MP3
    can only cut on whole frames, and because the source's trailing
    bit-reservoir and encoder padding hold no decodable samples once you cut
    into them, a copy near the end of the file decodes to *less* audio than its
    header claims (measured ~0.25 s short at 16 kHz). That loss differs between
    ffmpeg versions and would corrupt transcript timestamps and resume. Cutting
    from the *source* (not an intermediate MP3) re-encodes once, so it avoids a
    second generation of lossy compression; 16 kHz mono MP3 is fast.

    Args:
        source: The original media file (see ovox.media).
        output: Where to write the chunk.
        start: Start time in seconds.
        end: End time in seconds.
        sample_rate, channels, bitrate_bps: The upload format. Pass the values
            resolved from config (see :class:`ovox.config.AudioConfig`) so the
            chunk matches what transcription will upload.

    Raises:
        AudioExtractionError: ffmpeg failed, timed out, or produced no audio.
    """
    duration = end - start
    if duration <= 0:
        raise AudioExtractionError(f"Invalid chunk range: {start}s to {end}s")
    # Input seek (-ss before -i) is an O(1) seek, so cutting many chunks of a
    # long file is fast; -t then trims to the exact range. Re-encoding makes the
    # cut sample-accurate on every ffmpeg version (see the docstring).
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source.resolve()),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_bps // 1000}k",
        str(output),
    ]
    timeout = timeout_for(duration)
    try:
        result = run_ffmpeg(command, timeout)
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(
            f"ffmpeg timed out after {timeout:.0f}s cutting: {source}"
        ) from exc
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed while cutting {start}s-{end}s from: {source}\n"
            f"{stderr_tail(result.stderr)}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise AudioExtractionError(f"ffmpeg produced no audio cutting: {source}")
    return output


def decoded_duration(path: str | Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Measured length of *path* in seconds, by decoding and counting samples.

    ffprobe's reported (header) duration can disagree with the audio that
    actually decodes: for an MP3 the header includes encoder padding and the
    trailing bit-reservoir, so it can overstate the length (and that
    overstatement differs between ffmpeg versions). This fully decodes the file
    to raw PCM and counts the samples, which is the ground truth for how long
    the chunk really is. Decoding to a fixed *sample_rate* is fine because
    resampling preserves duration.

    Args:
        path: An audio file.
        sample_rate: The rate to decode to; only used to convert the sample
            count back to seconds, so the result is independent of it.

    Raises:
        AudioExtractionError: ffmpeg failed, timed out, or produced no audio.
    """
    target = Path(path)
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as handle:
        pcm = handle.name
    try:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(target.resolve()),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            pcm,
        ]
        try:
            result = run_ffmpeg(command, TIMEOUT_DEFAULT_S)
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"ffmpeg timed out measuring the duration of: {target}"
            ) from exc
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg failed measuring the duration of: {target}\n{stderr_tail(result.stderr)}"
            )
        # s16le mono: two bytes per sample.
        return (Path(pcm).stat().st_size // 2) / sample_rate
    finally:
        Path(pcm).unlink(missing_ok=True)


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
