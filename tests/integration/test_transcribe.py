"""Integration tests for ``ovox transcribe``: real ffmpeg, real files, and a
local fake STT server (standard library http.server) instead of the network.

The fake server answers the OpenAI-compatible transcriptions route and lets
each test program its behaviour (success, 429 + Retry-After, 401, invalid
JSON, one always-failing chunk) while recording every request it receives.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ovox.config import AudioConfig, ChunkConfig, SttConfig, TranscribeConfig
from ovox.pipeline import PipelineError, WorkdirError, prepare_for_dry_run, transcribe_file

# --- The fake STT server ---------------------------------------------------------


def _parse_multipart(body: bytes, boundary: bytes) -> dict[str, object]:
    """A just-enough multipart parser: field names -> value, files -> (name, bytes)."""
    parts: dict[str, object] = {}
    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n-")
        if not part or b"Content-Disposition" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head)
        if not name:
            continue
        key = name.group(1).decode()
        filename = re.search(rb'filename="([^"]*)"', head)
        if filename:
            parts[key] = (filename.group(1).decode(), content)
        else:
            parts[key] = content.decode()
    return parts


class _FakeSttHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep the test output quiet
        pass

    def do_POST(self) -> None:
        server: _FakeSttServer = self.server.fake  # type: ignore[union-attr]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        boundary = self.headers.get("Content-Type", "").split("boundary=", 1)[-1]

        record = {
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "model": None,
            "file_name": None,
            "file_size": 0,
        }
        if boundary:
            fields = _parse_multipart(body, boundary.encode())
            record["model"] = fields.get("model")
            file_field = fields.get("file")
            if isinstance(file_field, tuple):
                record["file_name"], record["file_bytes"] = file_field
                record["file_size"] = len(record["file_bytes"])
        with server.lock:
            server.requests.append(record)
            # Per-chunk attempt count: every chunk POSTs to the same path, so
            # counting per path would make the behaviour racy across chunks.
            key = str(record["file_name"])
            attempt = server.attempts.get(key, 0) + 1
            server.attempts[key] = attempt

        status, payload, headers = server.behavior(attempt, record)
        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
        elif isinstance(payload, dict):
            data = json.dumps(payload).encode()
        else:
            data = str(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", headers.pop("Content-Type", "application/json"))
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _FakeSttServer:
    """A threaded fake STT endpoint with a programmable per-request behaviour."""

    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSttHandler)
        self.httpd.daemon_threads = True
        # The handler's self.server is the httpd; reach this wrapper through it.
        self.httpd.fake = self  # type: ignore[attr-defined]
        self.lock = threading.Lock()
        self.requests: list[dict[str, object]] = []
        self.attempts: dict[str, int] = {}
        self.behavior = _default_behavior

    @property
    def endpoint(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1/audio/transcriptions"

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def file_names(self) -> list[str]:
        return [str(r["file_name"]) for r in self.requests]


def _default_behavior(
    attempt: int, record: dict[str, object]
) -> tuple[int, object, dict[str, str]]:
    return 200, {"text": f"text of {record['file_name']}"}, {}


@pytest.fixture
def server() -> _FakeSttServer:
    srv = _FakeSttServer()
    srv.start()
    yield srv
    srv.stop()


# --- helpers ---------------------------------------------------------------------


def stt_config(server: _FakeSttServer, model: str = "voxtral-1") -> SttConfig:
    return SttConfig(endpoint=server.endpoint, api_key="test-key", model=model)


def default_configs() -> tuple[ChunkConfig, AudioConfig, TranscribeConfig]:
    # Short chunks: tone_only.wav (3 s) becomes two chunks with a hard cut
    # (overlap 1 s = half the 2 s cap) and a second cut at the end of audio.
    return (
        ChunkConfig(max_chunk_size_bytes=26214400, max_chunk_duration_s=2.0, safety=1.0),
        AudioConfig(sample_rate=16000, channels=1, bitrate_bps=64_000),
        TranscribeConfig(max_concurrency=2, max_retries=2, backoff_base_s=0.01, backoff_cap_s=0.05),
    )


def run(
    media_dir: Path,
    tmp_path: Path,
    server: _FakeSttServer,
    *,
    name: str = "tone_only.wav",
    behavior: object = None,
    **kwargs: object,
) -> dict:
    if behavior is not None:
        server.behavior = behavior  # type: ignore[method-assign]
    chunk_config, audio_config, transcribe_config = default_configs()
    return transcribe_file(
        media_dir / name,
        chunk_config=chunk_config,
        audio_config=audio_config,
        stt=stt_config(server),
        transcribe_config=transcribe_config,
        workdir=tmp_path / "wd",
        **kwargs,  # type: ignore[arg-type]
    )


# --- end to end --------------------------------------------------------------------


def test_transcribe_end_to_end_with_hard_cut_overlap(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    """Two chunks of tone_only.wav: the second follows a hard cut (1 s overlap)
    and the fake model returns overlapping words for it, which the merge must
    drop — proving the whole chain: cut -> upload -> retry-free success -> merge.
    """

    def behavior(attempt: int, record: dict[str, object]) -> tuple[int, object, dict[str, str]]:
        file_name = str(record["file_name"])
        m = re.search(r"_chunk_(\d+)_", file_name)
        index = int(m.group(1)) if m else -1
        if index == 0:
            return 200, {"text": "alpha beta"}, {}
        return 200, {"text": "beta gamma"}, {}

    server.behavior = behavior  # type: ignore[method-assign]
    result = run(media_dir, tmp_path, server)

    assert result["chunks_total"] == 2
    assert result["chunks_transcribed"] == 2
    assert result["segments_file"] is None
    # The overlap word ("beta") appears exactly once in the merged text.
    assert result["text_file"].read_text() == "alpha beta gamma\n"
    # The requests were plain: model + file, bearer auth, an MP3 named after
    # the chunk.
    for record in server.requests:
        assert record["model"] == "voxtral-1"
        assert record["auth"] == "Bearer test-key"
        assert str(record["file_name"]).endswith(".mp3")
        assert record["file_size"] > 0  # a real MP3 was uploaded, not an empty part
    # The manifest records the source identity for resume checks.
    manifest = json.loads((tmp_path / "wd" / "ovox-chunk-manifest.json").read_text())
    assert manifest["input"]["size"] == (media_dir / "tone_only.wav").stat().st_size
    assert manifest["input"]["model"] == "voxtral-1"


def test_video_with_audio_end_to_end(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    result = run(media_dir, tmp_path, server, name="video_with_audio.mp4")
    assert result["chunks_total"] == 1
    # The uploaded file is the video's single chunk (audio extracted from video).
    names = server.file_names()
    assert len(names) == 1
    assert names[0].startswith("video_with_audio_chunk_")
    assert names[0].endswith(".mp3")
    assert result["text_file"].read_text() == f"text of {names[0]}\n"


def test_dry_run_makes_no_stt_requests(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    chunk_config, audio_config, _ = default_configs()
    result = prepare_for_dry_run(
        media_dir / "tone_only.wav",
        chunk_config=chunk_config,
        audio_config=audio_config,
        stt=stt_config(server),
        workdir=tmp_path / "wd",
    )
    assert result["chunks_pending"] == 2
    assert server.requests == []  # nothing was sent
    workdir = tmp_path / "wd"
    assert (workdir / "ovox-chunk-manifest.json").exists()
    assert len(list(workdir.glob("*_chunk_*.mp3"))) == 2
    assert not any(p.name.endswith(".txt") for p in workdir.parent.iterdir())


def test_dry_run_then_real_run_resumes(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    chunk_config, audio_config, _ = default_configs()
    prepare_for_dry_run(
        media_dir / "tone_only.wav",
        chunk_config=chunk_config,
        audio_config=audio_config,
        stt=stt_config(server),
        workdir=tmp_path / "wd",
    )
    assert server.requests == []

    result = run(media_dir, tmp_path, server, resume=True)
    # The chunks were already cut by the dry run: a resume skips the cut and
    # sends all (not-yet-transcribed) chunks.
    assert result["resumed"] is True
    assert result["chunks_transcribed"] == 2
    assert len(server.requests) == 2


# --- retries and failures ------------------------------------------------------------


def test_retry_on_429_with_retry_after(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    """Chunk 0 gets a 429 (with Retry-After) once, then succeeds; the run
    finishes with all chunks transcribed."""

    def behavior(attempt: int, record: dict[str, object]) -> tuple[int, object, dict[str, str]]:
        name = str(record["file_name"])
        if "_chunk_00_" in name and attempt == 1:
            return 429, {"error": "slow down"}, {"Retry-After": "0"}
        return 200, {"text": f"text of {name}"}, {}

    result = run(media_dir, tmp_path, server, behavior=behavior)

    assert result["chunks_transcribed"] == 2
    # Chunk 0 was retried after the 429; the other chunk needed one attempt.
    chunk0 = next(n for n in server.attempts if "_chunk_00_" in n)
    assert server.attempts[chunk0] == 2
    others = [n for n in server.attempts if n != chunk0]
    assert all(server.attempts[n] == 1 for n in others)


def test_401_fails_fast_and_points_at_check(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    def behavior(attempt: int, record: dict[str, object]) -> tuple[int, object, dict[str, str]]:
        return 401, {"error": "invalid key"}, {}

    with pytest.raises(PipelineError, match="ovox check"):
        run(media_dir, tmp_path, server, behavior=behavior)
    # 401 is fatal: no retries, so each chunk was attempted exactly once.
    assert all(attempts == 1 for attempts in server.attempts.values())


def test_invalid_json_is_fatal_and_not_retried(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    def behavior(attempt: int, record: dict[str, object]) -> tuple[int, object, dict[str, str]]:
        return 200, "this is not json", {}

    with pytest.raises(PipelineError, match="non-JSON"):
        run(media_dir, tmp_path, server, behavior=behavior)
    # Fatal on the first attempt for each chunk; no retry happened.
    assert all(attempts == 1 for attempts in server.attempts.values())


def test_partial_failure_keeps_results_and_resume_completes(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    """The last chunk always 500s: the run fails but the other chunks' results
    stay on disk; a second run (now healthy) transcribes only that chunk and
    produces the full transcript.

    tone_only.wav (3 s) with a 2 s max duration cuts into two chunks:
    [0, 2) hard-cut with 1 s overlap, then [1, 3) to the end of the audio.
    """
    # A previous test (sharing the session media dir) may have written the
    # transcript here; clear it so this assertion is about this run.
    (media_dir / "tone_only.txt").unlink(missing_ok=True)

    def failing(attempt: int, record: dict[str, object]) -> tuple[int, object, dict[str, str]]:
        if "_chunk_01_" in str(record["file_name"]):
            return 500, {"error": "overloaded"}, {}
        return 200, {"text": "ok"}, {}

    with pytest.raises(PipelineError, match="resume"):
        run(media_dir, tmp_path, server, behavior=failing)

    workdir = tmp_path / "wd"
    assert (workdir / "chunk_00.json").exists()  # the finished chunk survives
    assert not (workdir / "chunk_01.json").exists()  # the failed one does not
    assert not (media_dir / "tone_only.txt").exists()

    server.behavior = _default_behavior  # type: ignore[method-assign]
    server.requests.clear()
    result = run(media_dir, tmp_path, server, resume=True)

    assert result["resumed"] is True
    assert result["chunks_transcribed"] == 1
    # Only the missing chunk was sent this time.
    manifest = json.loads((workdir / "ovox-chunk-manifest.json").read_text())
    assert server.file_names() == [manifest["chunks"][1]["name"]]
    assert result["text_file"].exists()


def test_re_recorded_source_invalidates_the_workfolder(
    media_dir: Path, tmp_path: Path, server: _FakeSttServer
) -> None:
    """Same file name, different content: a --resume run must refuse to mix the
    old chunks with the new recording."""
    src = media_dir / "tone_only.wav"
    original = src.read_bytes()
    result0 = run(media_dir, tmp_path, server)
    assert result0["chunks_total"] == 2

    import time as _time

    _time.sleep(0.01)
    src.write_bytes(original[:200] + b"changed")
    try:
        with pytest.raises(WorkdirError, match="has changed since"):
            run(media_dir, tmp_path, server, resume=True)
    finally:
        src.write_bytes(original)


def test_resume_without_workfolder_fails(
    tmp_path: Path, server: _FakeSttServer, media_dir: Path
) -> None:
    chunk_config, audio_config, tc = default_configs()
    with pytest.raises(WorkdirError, match="No chunk manifest"):
        transcribe_file(
            media_dir / "tone_only.wav",
            chunk_config=chunk_config,
            audio_config=audio_config,
            stt=stt_config(server),
            transcribe_config=tc,
            workdir=tmp_path / "nope",
            resume=True,
        )
