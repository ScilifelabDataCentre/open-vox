"""API client for ovox."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ovox.config import SttConfig

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
        self.client: httpx.AsyncClient = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

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
        """Close the HTTP client."""
        await self.client.aclose()

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

    async def transcribe(self, audio_data: bytes, **kwargs: Any) -> dict[str, Any]:
        """Transcribe audio data.

        Args:
            audio_data: The audio data as bytes.
            **kwargs: Additional arguments to pass to the API.
                Common ones include:
                - language: str - The language of the audio
                - prompt: str - Optional prompt to guide the model
                - response_format: str - The format of the response (json, text, srt, etc.)
                - temperature: float - The sampling temperature

        Returns:
            The transcription result as a dictionary.
        """
        # This method will be implemented in a later step
        # For now, return a placeholder response with proper typing
        return {"text": "", "segments": [], "language": "", "duration": 0.0}
