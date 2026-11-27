"""Unit tests for the ``ovox transcribe`` command. The pipeline is mocked;
this covers argument wiring, output and exit codes."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

import ovox.pipeline as pipeline_mod
from ovox.cli import build_parser, main
from ovox.media import InputValidationError
from ovox.pipeline import PipelineError, WorkdirError


@pytest.fixture(autouse=True)
def stt_env_autouse(stt_env: dict[str, str]) -> None:
    """The transcribe command resolves its config (``SttConfig.from_env``)
    before it calls the pipeline, so the required STT settings must exist even
    though the pipeline is mocked here and no request is made. Reuse the shared
    ``stt_env`` fixture (tests/conftest.py) for every test so these do not
    depend on a local ``.env`` (git-ignored, absent in CI)."""


def result_dict(tmp_path: Path) -> dict:
    return {
        "text_file": tmp_path / "meeting.txt",
        "chunks_transcribed": 2,
        "chunks_total": 3,
        "resumed": False,
        "workdir": str(tmp_path / ".ovox-meeting-abc123"),
        "timings": {"probe": 0.05, "cut": 1.1, "stt": 12.3, "merge": 0.01},
    }


def plan_dict() -> dict:
    return {
        "input": {
            "file": "/x/meeting.mp4",
            "path": "/x/meeting.mp4",
            "model": "m",
            "format": "mp4",
            "codec": "aac",
            "duration": 90.0,
            "sample_rate": 44100,
            "channels": 2,
            "audio_track": 0,
        },
        "settings": {
            "max_chunk_size_bytes": {"value": 26214400, "source": "default", "origin": ""},
            "max_chunk_duration_s": {"value": 600.0, "source": "default", "origin": ""},
            "safety": {"value": 0.8, "source": "default", "origin": ""},
            "overlap_s": {"value": 1.0, "source": "default", "origin": ""},
            "search_window_s": {"value": 30.0, "source": "default", "origin": ""},
            "split_method": {"value": "silence", "source": "default", "origin": ""},
            "sample_rate": {"value": 16000, "source": "default", "origin": ""},
            "channels": {"value": 1, "source": "default", "origin": ""},
            "bitrate_kbps": {"value": 64, "source": "default", "origin": ""},
        },
        "chunks": [
            {
                "index": i,
                "start": i * 30.0,
                "end": (i + 1) * 30.0,
                "duration": 30.0,
                "overlap_before": 0.0 if i == 0 else 1.0,
                "estimated_size": 24000,
                "actual_size": 24000,
                "actual_duration": 30.0,
                "name": f"meeting_chunk_{i:02d}_{i * 30.0:g}-{(i + 1) * 30.0:g}.mp3",
                "reason": "cut on silence",
            }
            for i in range(3)
        ],
        "warnings": [],
        "totals": {"num_chunks": 3, "total_audio_length": 90.0},
        "split_method": "silence",
    }


def input_file(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fake")
    return path


def run_transcribe_cli(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, *args: str
) -> tuple[int, str, str, dict[str, object]]:
    """Run ``ovox transcribe`` with the pipeline faked; return (rc, out, err)."""
    file = input_file(tmp_path)
    calls: dict[str, object] = {}

    def fake_transcribe_file(path, **kwargs):
        calls["call"] = (path, kwargs)
        return result_dict(tmp_path)

    def fake_dry_run(path, **kwargs):
        calls["dry"] = (path, kwargs)
        return {
            "plan": plan_dict(),
            "workdir": str(tmp_path / "wd"),
            "chunks_transcribed": 1,
            "chunks_pending": 2,
            "resumed": False,
            "timings": {"probe": 0.1, "cut": 0.2},
        }

    monkeypatch.setattr(pipeline_mod, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(pipeline_mod, "prepare_for_dry_run", fake_dry_run)
    rc = main(["transcribe", str(file), *args])
    out, err = capsys.readouterr()
    calls["file"] = file
    return rc, out, err, calls


# --- parser ----------------------------------------------------------------------


def test_transcribe_command_in_parser() -> None:
    parser = build_parser()
    assert "transcribe" in parser._subparsers._group_actions[0].choices


def test_dry_run_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    file = input_file(tmp_path)
    with pytest.raises(SystemExit):
        main(["transcribe", str(file), "--dry-run", "--resume"])


# --- normal run --------------------------------------------------------------------


def test_transcribe_success_prints_summary(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, out, _err, calls = run_transcribe_cli(tmp_path, capsys, monkeypatch)

    assert rc == 0
    file, kwargs = calls["call"]
    assert Path(file) == calls["file"]
    assert kwargs["output"] is None
    assert kwargs["workdir"] is None
    assert kwargs["resume"] is False
    assert "Transcript:" in out
    assert "Chunks:   2 transcribed this run, 3 total" in out
    assert "Timings:" in out
    assert "stt=12.30s" in out


def test_transcribe_output_and_workdir_flags(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_file = tmp_path / "sub" / "notes.txt"
    wd = tmp_path / "mywork"
    rc, _out, _err, calls = run_transcribe_cli(
        tmp_path, capsys, monkeypatch, "-o", str(out_file), "--workdir", str(wd), "--resume"
    )

    assert rc == 0
    _file, kwargs = calls["call"]
    assert kwargs["output"] == out_file
    assert kwargs["workdir"] == wd
    assert kwargs["resume"] is True


def test_transcribe_json_output(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    rc, out, _err, _ = run_transcribe_cli(tmp_path, capsys, monkeypatch, "--json")

    assert rc == 0
    data = json.loads(out)
    assert data["chunks_transcribed"] == 2
    assert data["chunks_total"] == 3
    assert data["timings"]["stt"] == 12.3


# --- dry run -------------------------------------------------------------------------


def test_dry_run_makes_no_stt_calls(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, out, _err, calls = run_transcribe_cli(tmp_path, capsys, monkeypatch, "--dry-run")

    assert rc == 0
    assert "dry" in calls and "call" not in calls  # no real transcription
    assert "Chunks already transcribed: 1/3" in out
    assert "Dry run: no STT requests were made" in out
    # The plan table is printed (the same one ovox chunk shows).
    assert "meeting_chunk_01" in out


def test_dry_run_json_prints_the_plan(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, out, _err, _calls = run_transcribe_cli(tmp_path, capsys, monkeypatch, "--dry-run", "--json")

    assert rc == 0
    data = json.loads(out)
    assert data["totals"]["num_chunks"] == 3
    assert data["chunks"][0]["name"].startswith("meeting_chunk_00")
    assert data["input"]["model"] == "m"


# --- error handling -------------------------------------------------------------------


def _raiser(error: Exception):
    def raise_it(*args: object, **kwargs: object) -> None:
        raise error

    return raise_it


@pytest.mark.parametrize(
    ("error", "match"),
    [
        pytest.param(InputValidationError("File not found: x"), "File not found", id="media"),
        pytest.param(PipelineError("boom"), "boom", id="pipeline"),
        pytest.param(WorkdirError("no manifest"), "no manifest", id="workdir"),
        pytest.param(ValueError("bad setting"), "bad setting", id="config"),
    ],
)
def test_transcribe_errors_exit_nonzero(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    match: str,
) -> None:
    file = input_file(tmp_path)
    monkeypatch.setattr(pipeline_mod, "transcribe_file", _raiser(error))
    monkeypatch.setattr(pipeline_mod, "prepare_for_dry_run", _raiser(error))

    rc = main(["transcribe", str(file)])
    err = capsys.readouterr().err

    assert rc == 1
    assert match in err
    assert "Traceback" not in err


def test_transcribe_auth_error_points_at_check(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = input_file(tmp_path)
    error = PipelineError(
        "STT endpoint error: STT endpoint rejected the API key (HTTP 401). "
        "Run 'ovox check' to verify STT_ENDPOINT and STT_API_KEY."
    )
    monkeypatch.setattr(pipeline_mod, "transcribe_file", _raiser(error))
    monkeypatch.setattr(pipeline_mod, "prepare_for_dry_run", _raiser(error))

    rc = main(["transcribe", str(file)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "ovox check" in err


# --- SIGTERM -----------------------------------------------------------------------


@pytest.fixture
def clean_signals(monkeypatch: pytest.MonkeyPatch):
    """Restore the process' SIGINT/SIGTERM handlers after a test and reset the
    shared shutdown state the CLI's handlers act on."""
    import ovox.api as api_mod

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    api_mod._shutdown.reset()
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    api_mod._shutdown.reset()


def test_first_sigterm_with_no_requests_in_flight(
    tmp_path: Path, capsys, monkeypatch, clean_signals
) -> None:
    """With no requests in flight the first SIGTERM behaves like the default
    handler: its SystemExit(143) unwinds the run immediately, and the CLI
    reports it as a signal exit."""
    file = input_file(tmp_path)

    def fake_transcribe_file(path, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        # The handler's SystemExit(143) unwinds the run from here.
        raise AssertionError("should not reach this line")

    monkeypatch.setattr(pipeline_mod, "transcribe_file", fake_transcribe_file)

    rc = main(["transcribe", str(file)])
    err = capsys.readouterr().err

    assert rc == 143
    assert "Interrupted (signal)." in err


def test_second_interrupt_exits_at_once(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch, clean_signals
) -> None:
    """A second interrupt quits at once with the conventional code (143 for
    SIGTERM), whatever is still running. The first press arrives with a
    request in flight (the "waiting" path), the second ends the process.
    ``os._exit`` is faked (raising SystemExit instead of ending the test
    process); the real exit is covered by the subprocess tests in
    tests/integration/test_transcribe.py."""
    import ovox.api as api_mod

    file = input_file(tmp_path)
    exits: list[int] = []
    saw_waiting = threading.Event()  # the stage saw the interrupt with a request in flight
    done = threading.Event()  # the second press reached os._exit

    def fake_exit(code: int) -> None:
        exits.append(code)
        done.set()
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)

    def fake_transcribe_file(path, **kwargs):
        # Stand in for the STT stage with one request in flight. The two
        # interrupts are sent from a helper thread, one after the other, while
        # the "stage" (this thread) blocks between them — as the real main
        # thread does waiting on the pool. That makes press 1 and press 2
        # separate handler runs (no same-thread coalescing), and the stage
        # sees the first one with a request in flight (the "waiting" path).
        api_mod._shutdown.begin()

        def press() -> None:
            os.kill(os.getpid(), signal.SIGTERM)  # first: the "waiting" path
            time.sleep(1.0)  # let the first handler run before the second press
            os.kill(os.getpid(), signal.SIGTERM)  # second: exit at once

        threading.Thread(target=press, daemon=True).start()
        # Block as the main thread does (waiting on the pool); poll the
        # shutdown state so the first press is observed while a request is in
        # flight. The second press's SystemExit(143) unwinds the run from here.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if done.is_set():
                return
            if api_mod.interrupted() and api_mod._shutdown.requesting:
                saw_waiting.set()
            time.sleep(0.01)
        raise AssertionError("timed out waiting for the second interrupt to exit")

    monkeypatch.setattr(pipeline_mod, "transcribe_file", fake_transcribe_file)

    rc = main(["transcribe", str(file)])
    err = capsys.readouterr().err

    assert saw_waiting.is_set()  # the "waiting" path ran, not the default exit
    assert exits == [143]
    assert rc == 143
    assert "Interrupted (signal)." in err
    assert "Waiting for 1 in-flight request(s)" in err
