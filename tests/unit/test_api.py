"""Tests for the ovox STT API client."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ovox.api import STTClient
from ovox.config import SttConfig

ENDPOINT = "https://api.example.com/v1/audio/transcriptions"
API_KEY = "test-key"
MODEL = "voxtral-1"

# --- Helpers and fixtures -----------------------------------------------------


def make_response(
    status_code: int, text: str = "", url: str = "https://example.test"
) -> httpx.Response:
    """Build a real httpx.Response, as AsyncClient.get would return it."""
    return httpx.Response(status_code, text=text, request=httpx.Request("GET", url))


@pytest.fixture
def client() -> STTClient:
    """An STTClient built from explicit settings, so tests don't depend on the env."""
    return STTClient(endpoint=ENDPOINT, api_key=API_KEY, model=MODEL)


@pytest.fixture
def mock_get() -> Iterator[AsyncMock]:
    """Replace httpx.AsyncClient.get so no real HTTP request is made."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock:
        yield mock


def assert_common_fields(result: dict, client: STTClient, status_code: int | None) -> None:
    """Fields that must be present and correct on every check result."""
    assert result["endpoint"] == client.endpoint
    assert result["model"] == client.model
    assert isinstance(result["duration_ms"], int)
    assert result["duration_ms"] >= 0
    assert result["status_code"] == status_code


# --- Initialization -----------------------------------------------------------


def test_init_reads_settings_from_environment(
    stt_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STT_MODEL", "voxtral-1")

    client = STTClient()

    assert client.endpoint == stt_env["STT_ENDPOINT"]
    assert client.api_key == stt_env["STT_API_KEY"]
    assert client.model == "voxtral-1"


def test_init_arguments_override_environment(stt_env: dict[str, str]) -> None:
    client = STTClient(
        endpoint="https://custom-api.example.com/v1/audio/transcriptions",
        api_key="custom-key-123",
        model="custom-model",
    )

    assert client.endpoint == "https://custom-api.example.com/v1/audio/transcriptions"
    assert client.api_key == "custom-key-123"
    assert client.model == "custom-model"


def test_init_accepts_stt_config() -> None:
    stt = SttConfig(
        endpoint="https://stt.example.com/v1/audio/transcriptions",
        api_key="config-key",
        model="config-model",
    )

    client = STTClient(config=stt)

    assert client.endpoint == "https://stt.example.com/v1/audio/transcriptions"
    assert client.api_key == "config-key"
    assert client.model == "config-model"


# --- check_connection: success --------------------------------------------------


async def test_check_connection_success(client: STTClient, mock_get: AsyncMock) -> None:
    mock_get.return_value = make_response(200, "{}")

    result = await client.check_connection()

    assert result["success"] is True
    assert result["message"] == "Successfully connected to STT API"
    assert_common_fields(result, client, 200)
    assert "error" not in result

    mock_get.assert_called_once()
    url = str(mock_get.call_args.args[0])
    headers = mock_get.call_args.kwargs["headers"]
    assert "https://api.example.com/v1/models" in url
    assert headers["Authorization"] == f"Bearer {API_KEY}"


async def test_check_connection_200_html_is_not_a_success(
    client: STTClient, mock_get: AsyncMock
) -> None:
    """A 200 that is a web page (not the API) must not count as connected.

    Open WebUI-style gateways answer any unknown URL with 200 and the web UI's
    HTML, so a wrong STT_ENDPOINT would otherwise report success.
    """
    html = "<!doctype html><html><head><title>Open WebUI</title></head></html>"
    mock_get.return_value = httpx.Response(
        200,
        content=html.encode(),
        request=httpx.Request("GET", client.endpoint),
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    result = await client.check_connection()

    assert result["success"] is False
    assert result["error"] == "unexpected_response"
    # The HTTP status code was 200; it was the body that was wrong.
    assert_common_fields(result, client, 200)


# --- check_connection: HTTP error responses -----------------------------------


@pytest.mark.parametrize(
    ("status_code", "error"),
    [
        pytest.param(401, "unauthorized", id="401-unauthorized"),
        pytest.param(404, "not_found", id="404-not-found"),
        pytest.param(500, "http_error", id="500-server-error"),
    ],
)
async def test_check_connection_http_error(
    client: STTClient, mock_get: AsyncMock, status_code: int, error: str
) -> None:
    mock_get.return_value = make_response(status_code, "error body", client.endpoint)

    result = await client.check_connection()

    assert result["success"] is False
    assert result["error"] == error
    assert_common_fields(result, client, status_code)


# --- check_connection: exceptions ---------------------------------------------


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        pytest.param(
            httpx.TimeoutException("Timeout"),
            "timeout",
            id="timeout",
        ),
        pytest.param(
            httpx.ConnectError("Connection error"),
            "connection_error",
            id="connect-error",
        ),
        pytest.param(
            Exception("Unexpected error"),
            "unexpected_error",
            id="unexpected",
        ),
    ],
)
async def test_check_connection_exception(
    client: STTClient, mock_get: AsyncMock, exception: Exception, error: str
) -> None:
    mock_get.side_effect = exception

    result = await client.check_connection()

    assert result["success"] is False
    assert result["error"] == error
    # No HTTP response was received, so there is no status code to report.
    assert_common_fields(result, client, None)


# --- Context manager ----------------------------------------------------------


async def test_client_context_manager_closes_http_client(stt_env: dict[str, str]) -> None:
    with patch("httpx.AsyncClient.aclose", new_callable=AsyncMock) as mock_close:
        async with STTClient() as client:
            assert isinstance(client, STTClient)

    mock_close.assert_awaited_once()
