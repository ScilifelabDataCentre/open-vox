"""Tests for ovox CLI."""

import json
from collections.abc import Mapping
from unittest.mock import AsyncMock, patch

import pytest

from ovox.cli import build_parser, main

ENDPOINT = "https://api.example.com/v1/audio/transcriptions"
MODEL = "voxtral-1"


def success_result() -> dict:
    return {
        "success": True,
        "message": "Successfully connected to STT API",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "duration_ms": 142,
        "status_code": 200,
    }


def failure_result() -> dict:
    return {
        "success": False,
        "message": "Could not connect to the API endpoint. Check the URL and your network.",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "duration_ms": 0,
        "status_code": None,
        "error": "connection_error",
    }


def run_check(
    capsys: pytest.CaptureFixture[str], result: Mapping[str, object], *cli_args: str
) -> tuple[int, pytest.CaptureResult]:
    """Run ``ovox check`` with STTClient mocked to return *result*."""
    with patch("ovox.api.STTClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client
        mock_client.check_connection.return_value = result

        with patch.dict("os.environ", {"STT_ENDPOINT": ENDPOINT, "STT_API_KEY": "test-key"}):
            exit_code = main(["check", *cli_args])

        mock_client.check_connection.assert_called_once()
    captured = capsys.readouterr()
    return exit_code, captured


def test_build_parser_has_check_command():
    """Test that the argument parser includes the check command."""
    parser = build_parser()
    # Check if 'check' is in the list of subcommands
    assert "check" in parser._subparsers._group_actions[0].choices

    # Check the help text for the check command
    check_parser = parser._subparsers._group_actions[0].choices["check"]
    assert "STT API is reachable and credentials are valid" in check_parser.description


def test_check_command_success(capsys):
    """The check command exits 0 and prints the endpoint and configured model."""
    exit_code, captured = run_check(capsys, success_result())

    assert exit_code == 0
    # Verify the endpoint and model appear; do not compare whole sentences.
    assert ENDPOINT in captured.out
    assert MODEL in captured.out


def test_check_command_failure(capsys):
    """The check command exits 1 on failure and prints the endpoint and error."""
    exit_code, captured = run_check(capsys, failure_result())

    assert exit_code == 1
    # Endpoint and error are shown to aid debugging; no full-sentence comparison.
    assert ENDPOINT in captured.out
    assert "connection_error" in captured.out


def test_check_command_failure_http_error(capsys):
    """A non-2xx HTTP response exits non-zero and reports the status code."""
    result = {
        "success": False,
        "message": "Authentication failed. Check your API key.",
        "endpoint": ENDPOINT,
        "model": MODEL,
        "duration_ms": 30,
        "status_code": 401,
        "error": "unauthorized",
    }
    exit_code, captured = run_check(capsys, result)

    assert exit_code == 1
    assert ENDPOINT in captured.out


def test_check_command_missing_config(capsys, monkeypatch):
    """With STT_ENDPOINT / STT_API_KEY unset, check reports an actionable
    configuration error and exits non-zero (no raw traceback)."""
    monkeypatch.delenv("STT_ENDPOINT", raising=False)
    monkeypatch.delenv("STT_API_KEY", raising=False)

    result = main(["check"])

    assert result == 1
    captured = capsys.readouterr()
    assert "STT_ENDPOINT" in captured.err
    assert "traceback" not in captured.err.lower()


def test_check_command_json_output_success(capsys):
    """--json prints success, status_code, endpoint, model and duration_ms."""
    exit_code, captured = run_check(capsys, success_result(), "--json")

    assert exit_code == 0
    data = json.loads(captured.out)

    assert data["success"] is True
    assert data["status_code"] == 200
    assert data["endpoint"] == ENDPOINT
    assert data["model"] == MODEL
    assert isinstance(data["duration_ms"], int)
    assert data["duration_ms"] >= 0


def test_check_command_json_output_failure(capsys):
    """--json on failure has success false, an error type, and still includes
    endpoint and model."""
    exit_code, captured = run_check(capsys, failure_result(), "--json")

    assert exit_code == 1
    data = json.loads(captured.out)

    assert data["success"] is False
    assert data["status_code"] is None
    assert data["error"] == "connection_error"
    assert data["endpoint"] == ENDPOINT
    assert data["model"] == MODEL
    assert isinstance(data["duration_ms"], int)
    assert data["duration_ms"] >= 0
