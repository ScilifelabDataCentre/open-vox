"""Integration tests for ovox.media: real ffprobe on real files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ovox.media import InputValidationError, check_ffmpeg, validate_input

# --- Valid input --------------------------------------------------------------


def test_ffmpeg_is_available() -> None:
    check_ffmpeg()


@pytest.mark.parametrize("name", ["tone.wav", "tone.mp3"])
def test_audio_file_is_valid(media_dir: Path, name: str) -> None:
    info = validate_input(media_dir / name)

    assert info.audio_streams == 1
    assert info.has_video is False
    assert info.duration == pytest.approx(1.0, abs=0.1)


def test_video_with_audio_is_valid(media_dir: Path) -> None:
    info = validate_input(media_dir / "video_with_audio.mp4")

    assert info.audio_streams == 1
    assert info.has_video is True
    assert "mp4" in info.format_name


# --- Invalid input ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("missing.wav", "File not found", id="missing"),
        pytest.param("empty.wav", "File is empty", id="empty"),
        pytest.param("corrupt.wav", "Unsupported or corrupt", id="corrupt"),
        pytest.param("notes.txt", "Unsupported or corrupt", id="not-media"),
        pytest.param("video_no_audio.mp4", "Video file has no audio stream", id="video-no-audio"),
    ],
)
def test_invalid_input(media_dir: Path, name: str, expected: str) -> None:
    with pytest.raises(InputValidationError, match=expected):
        validate_input(media_dir / name)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="file permissions are not enforced for root (or on this platform)",
)
def test_unreadable_file(media_dir: Path, tmp_path: Path) -> None:
    locked = tmp_path / "locked.wav"
    locked.write_bytes((media_dir / "tone.wav").read_bytes())
    locked.chmod(0o000)
    try:
        with pytest.raises(InputValidationError, match="not readable"):
            validate_input(locked)
    finally:
        locked.chmod(0o600)  # let pytest clean up tmp_path


def test_path_starting_with_dash(
    media_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path like '-x.wav' must not be read by ffprobe as an option."""
    (tmp_path / "-x.wav").write_bytes((media_dir / "tone.wav").read_bytes())
    monkeypatch.chdir(tmp_path)

    assert validate_input("-x.wav").audio_streams == 1


# --- Your own media (opt-in) ----------------------------------------------------

LOCAL_MEDIA_DIR = os.environ.get("OVOX_TEST_MEDIA_DIR")


@pytest.mark.local_media
@pytest.mark.skipif(not LOCAL_MEDIA_DIR, reason="OVOX_TEST_MEDIA_DIR not set")
def test_local_media_files_are_valid() -> None:
    """Validate every file in $OVOX_TEST_MEDIA_DIR; all are expected to contain audio."""
    assert LOCAL_MEDIA_DIR is not None
    files = sorted(p for p in Path(LOCAL_MEDIA_DIR).expanduser().iterdir() if p.is_file())
    assert files, f"No files found in {LOCAL_MEDIA_DIR}"

    failures = {}
    for path in files:
        try:
            validate_input(path)
        except InputValidationError as exc:
            failures[path.name] = str(exc)

    assert not failures, failures
