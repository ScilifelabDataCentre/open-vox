"""Tests for the ovox STT API client."""

from __future__ import annotations

import random
import signal
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ovox.api import (
    FatalSttError,
    STTClient,
    SttError,
    backoff_delay,
    transcribe_chunk,
)
from ovox.config import SttConfig, TranscribeConfig

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


# --- transcribe_file: the plain request -------------------------------------


def _post_response(
    status_code: int,
    body: str = "",
    url: str = ENDPOINT,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a real httpx.Response, as Client.send would return it."""
    merged = {"Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    return httpx.Response(
        status_code, text=body, request=httpx.Request("POST", url), headers=merged
    )


def test_transcribe_file_sends_plain_multipart_request(client: STTClient, tmp_path) -> None:
    """The request carries only model and file: no verbose_json or timestamps."""
    chunk = tmp_path / "meeting_chunk_01_5.2-98.3.mp3"
    chunk.write_bytes(b"fake mp3")

    def send(request: httpx.Request, *args: object, **kwargs: object) -> httpx.Response:
        request.read()  # the real send serializes the body from the (open) file
        return _post_response(200, '{"text": "hello world"}')

    with patch("httpx.Client.send", side_effect=send) as mock_send:
        reply = client.transcribe_file(chunk, chunk_name=chunk.name)

    assert reply == {"text": "hello world"}
    request = mock_send.call_args.args[0]
    assert request.url == ENDPOINT
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    # The multipart body carries a "model" field and a "file" part named
    # after the chunk; nothing else is asked for.
    body = request.content
    assert b'name="model"' in body and MODEL.encode() in body
    assert b'name="file"' in body
    assert chunk.name.encode() in body
    assert b"audio/mpeg" in body
    assert b"timestamp" not in body and b"verbose" not in body


def test_transcribe_file_keeps_extra_reply_fields(client: STTClient, tmp_path) -> None:
    """Whatever the endpoint returns besides text is kept untouched."""
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    body = '{"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}'
    with patch("httpx.Client.send", return_value=_post_response(200, body)):
        reply = client.transcribe_file(chunk)

    assert reply["text"] == "hi"
    assert reply["segments"] == [{"start": 0.0, "end": 1.0, "text": "hi"}]


def test_transcribe_file_401_is_fatal_and_points_at_check(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", return_value=_post_response(401, "nope")),
        pytest.raises(FatalSttError, match="ovox check") as excinfo,
    ):
        client.transcribe_file(chunk, chunk_name="c.mp3")
    assert excinfo.value.chunk_name == "c.mp3"


def test_transcribe_file_404_is_fatal(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", return_value=_post_response(404, "")),
        pytest.raises(FatalSttError),
    ):
        client.transcribe_file(chunk)


def test_transcribe_file_bad_json_is_fatal(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    html = _post_response(200, "<html>gateway</html>")
    html.headers["Content-Type"] = "text/html"
    with (
        patch("httpx.Client.send", return_value=html),
        pytest.raises(FatalSttError, match="non-JSON"),
    ):
        client.transcribe_file(chunk)


def test_transcribe_file_missing_text_is_fatal(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", return_value=_post_response(200, '{"error": "boom"}')),
        pytest.raises(FatalSttError, match="no 'text' field"),
    ):
        client.transcribe_file(chunk)


def test_transcribe_file_retryable_status_keeps_retry_after(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch(
            "httpx.Client.send",
            return_value=_post_response(429, "", headers={"Retry-After": "7"}),
        ),
        pytest.raises(SttError) as excinfo,
    ):
        client.transcribe_file(chunk)
    assert not isinstance(excinfo.value, FatalSttError)
    assert excinfo.value.retry_after == 7.0


def test_transcribe_file_timeout_is_transient(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", side_effect=httpx.ReadTimeout("slow")),
        pytest.raises(SttError, match="may still be working"),
    ):
        client.transcribe_file(chunk)


def test_transcribe_file_timeout_is_marked(client: STTClient, tmp_path) -> None:
    """A timeout is flagged as such, so the retry loop can budget it less."""
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", side_effect=httpx.ReadTimeout("slow")),
        pytest.raises(SttError) as excinfo,
    ):
        client.transcribe_file(chunk)
    assert excinfo.value.timed_out is True


def test_transcribe_file_network_error_is_not_marked_timeout(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", side_effect=httpx.ConnectError("down")),
        pytest.raises(SttError, match="Network error") as excinfo,
    ):
        client.transcribe_file(chunk)
    assert excinfo.value.timed_out is False


# --- backoff_delay -----------------------------------------------------------


def test_backoff_delay_is_capped_exponential_with_jitter() -> None:
    rng = random.Random(1)
    for attempt in range(1, 6):
        for _ in range(20):
            delay = backoff_delay(attempt, base=1.0, cap=60.0, rng=rng)
            assert 0.0 <= delay < min(60.0, 1.0 * 2**attempt)
    # Once the schedule exceeds the cap, every value is bounded by the cap.
    for _ in range(50):
        assert backoff_delay(10, base=1.0, cap=5.0, rng=rng) < 5.0


def test_backoff_delay_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="base must be positive"):
        backoff_delay(1, base=0.0, cap=1.0)
    with pytest.raises(ValueError, match="cap must be >= base"):
        backoff_delay(1, base=2.0, cap=1.0)


# --- transcribe_chunk: the retry loop -----------------------------------------


def _transcribe_config(**overrides: object) -> TranscribeConfig:
    values: dict[str, object] = {
        "max_concurrency": 2,
        "max_retries": 3,
        "backoff_base_s": 1.0,
        "backoff_cap_s": 8.0,
        "max_timeout_retries": 1,
    }
    values.update(overrides)
    return TranscribeConfig(**values)  # type: ignore[arg-type]


def test_transcribe_chunk_retries_then_succeeds(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    sleeps: list[float] = []
    with patch(
        "httpx.Client.send",
        side_effect=[
            httpx.ReadTimeout("slow"),
            _post_response(503, ""),
            _post_response(200, '{"text": "done"}'),
        ],
    ):
        reply = transcribe_chunk(
            client,
            chunk,
            "c.mp3",
            _transcribe_config(),
            sleep=sleeps.append,
            rng=random.Random(0),
        )

    assert reply == {"text": "done"}
    assert len(sleeps) == 2  # one backoff per failed attempt, none after success
    for delay in sleeps:
        assert 0.0 <= delay <= 16.0  # capped at 2 * cap


def test_transcribe_chunk_honours_retry_after(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    sleeps: list[float] = []
    # A Retry-After of 30 s exceeds 2 * cap (16 s), so it is capped.
    with patch(
        "httpx.Client.send",
        side_effect=[
            _post_response(429, "", headers={"Retry-After": "30"}),
            _post_response(200, '{"text": "ok"}'),
        ],
    ):
        transcribe_chunk(
            client,
            chunk,
            "c.mp3",
            _transcribe_config(),
            sleep=sleeps.append,
            rng=random.Random(0),
        )

    assert sleeps == [16.0]


def test_transcribe_chunk_gives_up_after_max_retries(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with (
        patch("httpx.Client.send", return_value=_post_response(500, "broken")),
        pytest.raises(SttError, match="failed after 4 attempt") as excinfo,
    ):
        transcribe_chunk(
            client, chunk, "c.mp3", _transcribe_config(max_retries=3), sleep=lambda _s: None
        )
    assert excinfo.value.chunk_name == "c.mp3"


def test_transcribe_chunk_401_fails_at_once(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    sleeps: list[float] = []
    config = _transcribe_config()
    with (
        patch("httpx.Client.send", return_value=_post_response(401, "bad key")),
        pytest.raises(FatalSttError),
    ):
        transcribe_chunk(client, chunk, "c.mp3", config, sleep=sleeps.append)
    assert sleeps == []  # no backoff for a fatal error


def test_transcribe_chunk_stop_event_cancels(client: STTClient, tmp_path) -> None:
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    stop = {"set": False}

    class Stopper:
        def is_set(self) -> bool:
            return stop["set"]

    calls = {"n": 0}

    def post(*args: object, **kwargs: object) -> httpx.Response:
        calls["n"] += 1
        stop["set"] = True  # a sibling chunk failed after this attempt
        return _post_response(500, "broken")

    with patch("httpx.Client.send", side_effect=post), pytest.raises(SttError, match="Stopped"):
        transcribe_chunk(
            client,
            chunk,
            "c.mp3",
            _transcribe_config(),
            sleep=lambda _s: None,
            stop_event=Stopper(),
        )
    # One request was made, then the stop event prevented the next attempt.
    assert calls["n"] == 1


def test_timeout_retries_use_their_own_smaller_budget(
    client: STTClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of timeouts spends only the timeout budget, so the 429/5xx budget
    is not consumed (a slow server should not burn the retries meant for a
    busy one), and the chunk fails after the smaller number of attempts."""
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    calls = {"n": 0}

    def send(*args: object, **kwargs: object) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    config = _transcribe_config(max_retries=3, max_timeout_retries=1)
    with (
        patch("httpx.Client.send", side_effect=send),
        pytest.raises(SttError, match="failed after 2 attempt"),
    ):
        transcribe_chunk(client, chunk, "c.mp3", config, sleep=lambda _s: None)

    # 1 initial attempt + 1 timeout retry, not 1 + 3.
    assert calls["n"] == 2


def test_timeouts_and_errors_have_separate_budgets(
    client: STTClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timeouts and 5xx failures each keep their own budget: the 500 does not
    spend a retry of the timeout budget, and vice versa."""
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    with patch(
        "httpx.Client.send",
        side_effect=[
            httpx.ReadTimeout("slow"),  # timeout 1/1
            _post_response(500, "broken"),  # 5xx 1/3
            _post_response(500, "broken"),  # 5xx 2/3
            _post_response(500, "broken"),  # 5xx 3/3
            _post_response(200, '{"text": "done"}'),
        ],
    ):
        config = _transcribe_config(max_retries=3, max_timeout_retries=1)
        reply = transcribe_chunk(
            client, chunk, "c.mp3", config, sleep=lambda _s: None, rng=random.Random(0)
        )

    assert reply == {"text": "done"}  # the 500s were retried, not aborted by the timeout


def test_transcribe_chunk_interrupt_stops_retries(
    client: STTClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt signal stops the retries: no request goes out after it,
    even though the budget is not spent."""
    import ovox.api as api_mod

    # A fresh shutdown state (the module singleton is shared; monkeypatch
    # restores it after the test).
    fresh = api_mod._ShutdownState()
    monkeypatch.setattr(api_mod, "_shutdown", fresh)

    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    calls = {"n": 0}

    def send(*args: object, **kwargs: object) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            fresh.abort(signal.SIGINT)  # an interrupt arrives after attempt 1
        return _post_response(500, "broken")

    with (
        patch("httpx.Client.send", side_effect=send),
        pytest.raises(SttError, match="interrupted"),
    ):
        transcribe_chunk(client, chunk, "c.mp3", _transcribe_config(), sleep=lambda _s: None)

    assert calls["n"] == 1  # the second attempt was not sent
