"""Tests for the ovox STT API client."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ovox.api import STTClient

# --- Helpers and fixtures -----------------------------------------------------


def make_response(
    status_code: int, text: str = "", url: str = "https://example.test"
) -> httpx.Response:
    """Build a real httpx.Response, as AsyncClient.get would return it."""
    return httpx.Response(status_code, text=text, request=httpx.Request("GET", url))


@pytest.fixture
def mock_get() -> Iterator[AsyncMock]:
    """Replace httpx.AsyncClient.get so no real HTTP request is made."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock:
        yield mock


# --- Initialization -----------------------------------------------------------


def test_init_reads_settings_from_environment(
    stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_MODEL", "whisper-1")

    client = STTClient()

    assert client.endpoint == stt_env["STT_ENDPOINT"]
    assert client.api_key == stt_env["STT_API_KEY"]
    assert client.model == "whisper-1"


def test_init_arguments_override_environment(stt_env: dict[str, str]) -> None:
    client = STTClient(
        endpoint="https://custom-api.example.com/v1/audio/transcriptions",
        api_key="custom-key-123",
        model="custom-model",
    )

    assert client.endpoint == "https://custom-api.example.com/v1/audio/transcriptions"
    assert client.api_key == "custom-key-123"
    assert client.model == "custom-model"


# --- check_connection: success --------------------------------------------------


async def test_check_connection_success(stt_env: dict[str, str], mock_get: AsyncMock) -> None:
    mock_get.return_value = make_response(200, "{}")

    result = await STTClient().check_connection()

    assert result["success"] is True
    assert result["message"] == "Successfully connected to STT API"
    assert result["status_code"] == 200

    mock_get.assert_called_once()
    url = str(mock_get.call_args.args[0])
    headers = mock_get.call_args.kwargs["headers"]
    assert "https://api.example.com/v1/models" in url
    assert headers["Authorization"] == f"Bearer {stt_env['STT_API_KEY']}"


# --- check_connection: HTTP error responses -----------------------------------


@pytest.mark.parametrize(
    ("status_code", "body", "message", "error"),
    [
        pytest.param(
            401,
            "Unauthorized",
            "Authentication failed. Check your API key.",
            "unauthorized",
            id="401-unauthorized",
        ),
        pytest.param(
            404,
            "Not Found",
            "API endpoint not found. Check the URL.",
            "not_found",
            id="404-not-found",
        ),
        pytest.param(
            500,
            "Internal Server Error",
            "API returned error: 500",
            "Internal Server Error",
            id="500-server-error",
        ),
    ],
)
async def test_check_connection_http_error(
    stt_env: dict[str, str],
    mock_get: AsyncMock,
    status_code: int,
    body: str,
    message: str,
    error: str,
) -> None:
    mock_get.return_value = make_response(status_code, body, stt_env["STT_ENDPOINT"])

    result = await STTClient().check_connection()

    assert result["success"] is False
    assert result["message"] == message
    assert result["status_code"] == status_code
    assert result["error"] == error


# --- check_connection: exceptions ---------------------------------------------


@pytest.mark.parametrize(
    ("exception", "message", "error"),
    [
        pytest.param(
            httpx.TimeoutException("Timeout"),
            "Connection timed out. Check your network and API endpoint.",
            "timeout",
            id="timeout",
        ),
        pytest.param(
            httpx.ConnectError("Connection error"),
            "Could not connect to the API endpoint. Check the URL and your network.",
            "connection_error",
            id="connect-error",
        ),
        pytest.param(
            Exception("Unexpected error"),
            "Unexpected error: Unexpected error",
            "Unexpected error",
            id="unexpected",
        ),
    ],
)
async def test_check_connection_exception(
    stt_env: dict[str, str],
    mock_get: AsyncMock,
    exception: Exception,
    message: str,
    error: str,
) -> None:
    mock_get.side_effect = exception

    result = await STTClient().check_connection()

    assert result["success"] is False
    assert result["message"] == message
    assert result["error"] == error


# --- Context manager ----------------------------------------------------------


async def test_client_context_manager_closes_http_client(stt_env: dict[str, str]) -> None:
    with patch("httpx.AsyncClient.aclose", new_callable=AsyncMock) as mock_close:
        async with STTClient() as client:
            assert isinstance(client, STTClient)

    mock_close.assert_awaited_once()
