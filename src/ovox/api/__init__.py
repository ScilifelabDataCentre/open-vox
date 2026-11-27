"""API client for ovox.

The endpoint contract is an **OpenAI-compatible transcription API**:
``POST .../audio/transcriptions`` with a multipart form (``file`` and
``model``), bearer-token auth, and a JSON reply with a ``text`` field.
Everything endpoint-specific lives in :meth:`STTClient.transcribe_file`, the
function that sends one plain request — no ``verbose_json`` or timestamp
options, so the result does not depend on how a gateway (Open WebUI, ...)
happens to treat options its model does not support. Timestamps are a bonus
when an endpoint offers them: the reply is stored as-is and the pipeline
reads what it can (see :mod:`ovox.stitch`).

Retries live in :func:`transcribe_chunk`: transient failures (network errors,
HTTP 408/409/425/429/5xx, and read timeouts) are retried with capped
exponential backoff and full jitter, honouring ``Retry-After``. Timeouts get
a smaller retry budget than the other transient failures, because when the
read timeout fires the server may still be working on the chunk and a retry
re-sends the same audio on top of it. Authentication errors (401/403) fail at
once and point at ``ovox check``; other 4xx and invalid responses fail without
retrying.
"""

from __future__ import annotations

import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from ovox.config import SttConfig, TranscribeConfig

# Self-hosted STT endpoints routinely take many seconds (or minutes) to process a
# chunk, so httpx's 5s default is far too aggressive. Generous per-request
# timeouts keep real transcription calls from failing on slow hardware.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 600.0
REQUEST_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT_S,
    read=READ_TIMEOUT_S,
    write=READ_TIMEOUT_S,
    pool=CONNECT_TIMEOUT_S,
)


# --- Errors -----------------------------------------------------------------


class SttError(Exception):
    """A chunk could not be transcribed (or the endpoint is misconfigured).

    Attributes:
        chunk_name: The chunk the failure is about, or ``""`` when it is not
            about one specific chunk.
        retry_after: Seconds the server asked to wait (``Retry-After``), when
            the failure carried one. The retry loop uses it in place of the
            computed backoff (capped).
        timed_out: Set when the failure was a read timeout: the request left
            but no answer came back in time, so the server may still be
            processing the chunk. The retry loop budgets fewer retries for
            these than for other transient failures.
    """

    def __init__(
        self,
        message: str,
        *,
        chunk_name: str = "",
        retry_after: float | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.chunk_name = chunk_name
        self.retry_after = retry_after
        self.timed_out = timed_out


class FatalSttError(SttError):
    """A failure that will not succeed on retry: authentication, an invalid
    request (other 4xx), or a response that is not what ovox expects."""


# Statuses a self-hosted endpoint may return while it is busy or the request
# arrived at an unlucky moment; all safe to retry (5xx handled separately).
RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def _is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUS or 500 <= status < 600


def _failure_suffix(chunk_name: str) -> str:
    return f" for {chunk_name}" if chunk_name else ""


# --- Interruptible shutdown -----------------------------------------------------


# How long the first interrupt waits for in-flight requests to finish before
# the process quits at once (the user's second Ctrl-C does the same sooner).
COUNTDOWN_S = 5.0


class _ShutdownState:
    """Process-wide state for the clean Ctrl-C / SIGTERM shutdown.

    The STT stage runs its requests in worker threads, and the thread pool
    waits for the running requests when the process exits — with a 600 s read
    timeout each of them could hold the shutdown up for minutes. The shutdown
    is bounded in two ways: a request that is in flight is counted, so the
    CLI can say how many are still running, and a watchdog ends the process
    at once (``os._exit``) if any of them outlasts a short countdown. In
    between, the transcription stage checks :meth:`interrupted` between
    chunks and between retries, so no chunk that has not started is sent and
    no retry goes out after the interrupt. A request that finishes around the
    interrupt still returns, so its chunk result is kept for the resume; the
    ones that are cut short by the watchdog are redone on resume anyway (the
    finished transcripts are already safely on disk, written atomically).
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requesting = 0  # in-flight requests (the main thread never sends any)
        self.cancel = threading.Event()  # set when the interrupt arrived
        self._countdown_at = 0.0  # monotonic deadline set by the first interrupt
        self._signum: int | None = None  # the signal that started the shutdown

    def begin(self) -> None:
        with self.lock:
            self.requesting += 1

    def end(self) -> None:
        with self.lock:
            self.requesting -= 1

    def interrupted(self) -> bool:
        """True once an interrupt arrived. The stage checks it between chunks
        and between retries, so nothing not yet started is sent."""
        return self.cancel.is_set()

    def abort(self, signum: int | None = None) -> int:
        """Register the first interrupt: tell the run to stop and arm the
        countdown watchdog. Returns the number of in-flight requests — 0 when
        there was nothing to wait for. *signum* is the signal that arrived; it
        fixes the conventional exit code (128+signum).

        The watchdog is armed even when no request is in flight at the moment
        of the interrupt: a request can start counting right after (a worker
        that passed its stop check a moment earlier), and it then still gets
        the countdown bound. When nothing is running by the expiry, the
        watchdog simply does nothing.
        """
        with self.lock:
            if self.cancel.is_set():
                return 0
            self.cancel.set()
            self._signum = signum
            n = self.requesting
        self._countdown_at = time.monotonic() + COUNTDOWN_S
        threading.Thread(target=self._quit_if_still_running, daemon=True).start()
        return n

    def exit_code(self) -> int:
        """The conventional exit code for the interrupt that started the
        shutdown: 130 for SIGINT, 143 for SIGTERM."""
        return 128 + (self._signum or signal.SIGINT)

    def _quit_if_still_running(self) -> None:
        """The countdown backstop: if the in-flight requests have not finished
        within :data:`COUNTDOWN_S`, end the process at once instead of waiting
        out the read timeout. ``os._exit`` is deliberate — the remaining work
        is just the process ending, and every chunk that finished is already
        safely on disk."""
        time.sleep(max(0.0, self._countdown_at - time.monotonic()))
        with self.lock:
            n = self.requesting
        if n and self.cancel.is_set():
            print(
                f"{n} request(s) still running after the countdown; quitting at once. "
                "Unfinished chunks are redone on resume.",
                file=sys.stderr,
            )
            os._exit(self.exit_code())

    def reset(self) -> None:
        """Back to the pristine state (used by tests)."""
        with self.lock:
            self.requesting = 0
        self.cancel.clear()
        self._countdown_at = 0.0
        self._signum = None


_shutdown = _ShutdownState()


def interrupted() -> bool:
    """True once an interrupt signal (Ctrl-C, SIGTERM) arrived.

    Checked by the transcription stage between chunks (so no chunk not yet
    started is sent) and between retries (so no new request goes out after
    the interrupt).
    """
    return _shutdown.interrupted()


# --- Response handling -------------------------------------------------------


def _classify_http_error(status: int, body: str, chunk_name: str) -> FatalSttError:
    """Turn a non-2xx response into the right error type with an actionable message."""
    detail = body.strip()
    if len(detail) > 500:
        detail = detail[:500] + "..."
    detail = f" {detail}" if detail else ""
    if status in (401, 403):
        return FatalSttError(
            "STT endpoint rejected the API key "
            f"(HTTP {status}). Run 'ovox check' to verify STT_ENDPOINT and STT_API_KEY." + detail,
            chunk_name=chunk_name,
        )
    return FatalSttError(
        f"STT endpoint returned HTTP {status}{_failure_suffix(chunk_name)}." + detail,
        chunk_name=chunk_name,
    )


def _parse_reply(payload: Any, chunk_name: str) -> dict[str, Any]:
    """Validate a transcription reply and return it as a dict.

    The reply must be a JSON object with a ``text`` string. Anything else is a
    fatal, non-retriable error: retrying will not fix a body the endpoint
    chose to send. Extra fields (``segments``, ``words``, ...) are kept as-is.
    """
    if not isinstance(payload, dict):
        raise FatalSttError(
            f"STT endpoint returned a response that is not a JSON object"
            f"{_failure_suffix(chunk_name)}."
        )
    text = payload.get("text")
    if not isinstance(text, str):
        raise FatalSttError(
            f"STT endpoint response has no 'text' field{_failure_suffix(chunk_name)}: "
            "the endpoint may be returning an error page or an unsupported format."
        )
    return payload


# --- Backoff ------------------------------------------------------------------


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random | None = None) -> float:
    """Capped exponential backoff with full jitter for the retry after *attempt*.

    ``attempt`` is the number of failed attempts so far (1 after the first
    failure). The delay is a uniform random value in ``[0, min(cap,
    base * 2**attempt))``; the jitter spreads the retries from parallel chunk
    requests so they do not hit the endpoint at the same moment. *rng* is
    injectable so tests get deterministic delays.
    """
    if base <= 0:
        raise ValueError("base must be positive")
    if cap < base:
        raise ValueError(f"cap must be >= base, got: {cap} < {base}")
    upper = min(cap, base * (2.0**attempt))
    rand = rng or random
    return rand.uniform(0.0, upper)


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    """The server's Retry-After header in seconds, or None if absent/invalid.

    Only the ``seconds`` form is used (HTTP-date is rare on API servers); a
    non-numeric value is ignored rather than trusted blindly.
    """
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# --- Client -------------------------------------------------------------------


class STTClient:
    """Client for interacting with the STT API."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config: SttConfig | None = None,
    ) -> None:
        """Initialise the STT client.

        Args:
            endpoint: The STT API endpoint URL. If not provided, uses the configured value.
            api_key: The API key for authentication. If not provided, uses the configured value.
            model: The model to use for transcription. If not provided, uses the configured value.
            config: STT settings to use. If not provided, they are read from the
                environment (and .env). Individual arguments override them.
        """
        if config is None:
            if endpoint is None or api_key is None or model is None:
                config = SttConfig.from_env()
            else:
                config = SttConfig(endpoint=endpoint, api_key=api_key, model=model)

        self.endpoint: str = endpoint or config.endpoint
        self.api_key: str = api_key or config.api_key
        self.model: str = model or config.model
        # check_connection is async; the chunk transcription calls run in a
        # thread pool and use the synchronous client. A sync httpx.Client's
        # connection pool is thread-safe, so one instance serves all threads.
        self.client: httpx.AsyncClient = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        self.http: httpx.Client = httpx.Client(timeout=REQUEST_TIMEOUT)

    async def __aenter__(self) -> STTClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close both HTTP clients (use from an async context)."""
        await self.client.aclose()
        self.http.close()

    def close_http(self) -> None:
        """Close the synchronous HTTP client (chunk transcription).

        The transcription pipeline runs in a thread pool with no event loop,
        so it closes the sync client this way; the async client it created is
        never used there and is dropped with the process.
        """
        self.http.close()

    async def check_connection(self) -> dict[str, Any]:
        """Make a cheap authenticated call to check if the API is reachable
        and credentials are valid.

        Returns:
            A dictionary with connection status and details.
            {
                "success": bool,
                "message": str,
                "endpoint": str,
                "model": str,
                "duration_ms": int,
                "status_code": int | None,  # None when no HTTP response was received
                "error": str  # only on failure, e.g. "connection_error", "timeout"
            }
        """
        started = time.monotonic()

        def _elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # Present on every result, success or failure, to aid debugging.
        base: dict[str, Any] = {
            "success": False,
            "endpoint": self.endpoint,
            "model": self.model,
            "status_code": None,
        }

        try:
            # Use a cheap API call to check connection - getting models list or health check
            # For OpenAI-compatible APIs, we can use the models endpoint
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Try to get models list (a cheap authenticated call)
            response = await self.client.get(
                f"{self.endpoint.rsplit('/audio/transcriptions', 1)[0]}/models", headers=headers
            )

            # Raises httpx.HTTPStatusError for any non-2xx status, handled below.
            response.raise_for_status()
            base["status_code"] = response.status_code

            # A 2xx alone does not prove the URL is right: servers like Open
            # WebUI answer *any* unknown URL with 200 and a web page. A working
            # STT API answers with JSON, so the body must parse as JSON too.
            content_type = response.headers.get("Content-Type", "").lower()
            try:
                # A non-JSON body raises ValueError (json.JSONDecodeError).
                response.json()
            except ValueError:
                kind = "HTML" if "text/html" in content_type else "a non-JSON response"
                return {
                    **base,
                    "message": (
                        f"Endpoint returned {kind} instead of JSON. The URL may point at a web "
                        "page rather than the STT API. Check STT_ENDPOINT."
                    ),
                    "error": "unexpected_response",
                    "duration_ms": _elapsed_ms(),
                }

            return {
                **base,
                "success": True,
                "message": "Successfully connected to STT API",
                "duration_ms": _elapsed_ms(),
            }

        except httpx.TimeoutException:
            return {
                **base,
                "message": "Connection timed out. Check your network and API endpoint.",
                "error": "timeout",
                "duration_ms": _elapsed_ms(),
            }
        except httpx.ConnectError:
            return {
                **base,
                "message": "Could not connect to the API endpoint. Check the URL and your network.",
                "error": "connection_error",
                "duration_ms": _elapsed_ms(),
            }
        except httpx.HTTPStatusError as e:
            base["status_code"] = e.response.status_code
            if e.response.status_code == 401:
                base["message"] = "Authentication failed. Check your API key."
                base["error"] = "unauthorized"
            elif e.response.status_code == 404:
                base["message"] = "API endpoint not found. Check the URL."
                base["error"] = "not_found"
            else:
                base["message"] = f"API returned error: {e.response.status_code}"
                base["error"] = "http_error"
            base["duration_ms"] = _elapsed_ms()
            return base
        except Exception as e:
            return {
                **base,
                "message": f"Unexpected error: {e!s}",
                "error": "unexpected_error",
                "duration_ms": _elapsed_ms(),
            }

    def transcribe_file(self, path: str | Path, *, chunk_name: str = "") -> dict[str, Any]:
        """Send one chunk to the STT endpoint and return the parsed reply.

        The request is deliberately plain: just ``model`` and ``file``. Some
        endpoints (Voxtral on vLLM, Open WebUI's handling of unsupported
        options) do not honour ``response_format``/``timestamp_granularities``,
        so ovox does not ask for anything beyond the text. Whatever extra
        fields the endpoint does return (for example ``segments`` or
        ``words``) are kept in the reply untouched.

        The in-flight request is counted for the interrupt shutdown (see
        :class:`_ShutdownState`): if the run is interrupted while it is
        running, the count is what the "waiting for N requests" note and the
        countdown backstop are based on. A response that completes around the
        interrupt is still returned, so its chunk result is kept for the
        resume.

        Args:
            path: The chunk audio file to upload.
            chunk_name: The chunk's file name, used only in error messages.

        Returns:
            The full reply as a dict; it always contains a ``text`` string.

        Raises:
            FatalSttError: auth failure (401/403), other 4xx, or a reply that
                is not a JSON object with a ``text`` field.
            SttError: a transient failure (timeout, network error, retryable
                HTTP status) — the caller's retry loop handles those.
        """
        audio_path = Path(path)
        _shutdown.begin()
        try:
            try:
                # The file must stay open through the send: the multipart
                # body is serialized lazily from it, at send time.
                with audio_path.open("rb") as f:
                    response = self.http.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data={"model": self.model},
                        files={"file": (audio_path.name, f, "audio/mpeg")},
                    )
            except httpx.TimeoutException as exc:
                raise SttError(
                    f"STT request timed out{_failure_suffix(chunk_name)} (read timeout: "
                    f"{READ_TIMEOUT_S:.0f}s); the server may still be working on it.",
                    chunk_name=chunk_name,
                    timed_out=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise SttError(
                    f"Network error talking to the STT endpoint"
                    f"{_failure_suffix(chunk_name)}: {exc}",
                    chunk_name=chunk_name,
                ) from exc
        finally:
            _shutdown.end()

        if response.status_code < 200 or response.status_code >= 300:
            retry_after = _retry_after_seconds(response.headers)
            if _is_retryable_status(response.status_code):
                raise SttError(
                    f"STT endpoint returned HTTP {response.status_code}"
                    f"{_failure_suffix(chunk_name)}.",
                    chunk_name=chunk_name,
                    retry_after=retry_after,
                )
            raise _classify_http_error(response.status_code, response.text, chunk_name)
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise FatalSttError(
                f"STT endpoint returned a non-JSON response{_failure_suffix(chunk_name)}."
            ) from exc
        return _parse_reply(payload, chunk_name)


# --- Retried transcription of one chunk ----------------------------------------


def transcribe_chunk(
    client: STTClient,
    path: str | Path,
    chunk_name: str,
    config: TranscribeConfig,
    *,
    sleep: Any = time.sleep,
    rng: random.Random | None = None,
    stop_event: Any = None,
) -> dict[str, Any]:
    """Transcribe one chunk file, retrying transient failures.

    Network errors and retryable HTTP statuses (408/409/425/429/5xx) are
    retried at most ``config.max_retries`` times; read timeouts are retried at
    most ``config.max_timeout_retries`` times, and less often by default: when
    the read timeout fires the server may still be processing the chunk, so a
    retry re-sends the same audio while the first attempt may still be
    running. Between attempts it sleeps for capped exponential backoff with
    full jitter, or at least for the server's ``Retry-After`` seconds when it
    sent one (capped at twice the backoff cap, so a bad header cannot stall
    the run). A :class:`FatalSttError` (auth, other 4xx, invalid reply) is
    re-raised at once. *stop_event* (an object with ``is_set()``) is checked
    before each attempt and before each sleep: once set, the call raises a
    :class:`SttError` without sending another request, so a run stops cleanly
    after another chunk has failed for good. An interrupt signal stops the
    retries the same way (:func:`interrupted`), so no new request goes out
    after Ctrl-C.

    Args:
        client: The endpoint client (provides the synchronous HTTP client).
        path: The chunk audio file.
        chunk_name: The chunk's file name, for error messages.
        config: Retry settings (max_retries, max_timeout_retries, backoff
            base and cap).
        sleep: Injectable so tests run instantly.
        rng: Injectable randomness for the jitter.
        stop_event: Optional stop signal, checked between attempts.

    Returns:
        The parsed reply (always with a ``text`` string).

    Raises:
        FatalSttError: a non-retriable endpoint failure.
        SttError: the chunk could not be transcribed (naming the chunk and the
            last error), or the stop event was set.
    """
    general_budget = config.max_retries
    timeout_budget = config.max_timeout_retries
    attempt = 0
    last_error: SttError | None = None

    def _stop_requested() -> bool:
        return (stop_event is not None and stop_event.is_set()) or interrupted()

    def _stopped() -> SttError:
        return SttError(
            f"Stopped: another chunk failed or the run was interrupted, so "
            f"{chunk_name} was not transcribed.",
            chunk_name=chunk_name,
        )

    while True:
        if _stop_requested():
            raise _stopped()
        attempt += 1
        try:
            return client.transcribe_file(path, chunk_name=chunk_name)
        except FatalSttError:
            raise
        except SttError as exc:
            last_error = exc
            # Each kind of transient failure has its own retry budget, so a
            # stream of timeouts cannot spend the retries meant for 429s (and
            # vice versa), and a timeout never runs past its smaller budget.
            if exc.timed_out:
                timeout_budget -= 1
            else:
                general_budget -= 1
            if timeout_budget < 0 or general_budget < 0:
                break
            delay = backoff_delay(attempt, config.backoff_base_s, config.backoff_cap_s, rng=rng)
            if exc.retry_after is not None:
                delay = max(delay, min(exc.retry_after, config.backoff_cap_s * 2))
            if _stop_requested():
                raise _stopped() from exc
            sleep(delay)
    assert last_error is not None  # the loop only breaks after a failure
    raise SttError(
        f"Chunk {chunk_name} failed after {attempt} attempt(s): {last_error}",
        chunk_name=chunk_name,
    ) from last_error
