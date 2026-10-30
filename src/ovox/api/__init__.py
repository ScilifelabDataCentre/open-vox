"""API client for ovox."""

from __future__ import annotations

from typing import Any

import httpx

from ovox.config import Config


class STTClient:
    """Client for interacting with the STT API."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:

        """Initialize the STT client.

        Args:
            endpoint: The STT API endpoint URL. If not provided, uses the configured value.
            api_key: The API key for authentication. If not provided, uses the configured value.
            model: The model to use for transcription. If not provided, uses the configured value.
        """
        # Get config if needed
        if endpoint is None or api_key is None or model is None:
            config = Config.from_env()

        self.endpoint = endpoint or config.stt_endpoint
        self.api_key = api_key or config.stt_api_key
        self.model = model or config.stt_model
        self.client = httpx.AsyncClient()

    async def __aenter__(self) -> STTClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
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
                "status_code": int (if applicable),
                "error": str (if applicable)
            }
        """
        try:
            # Use a cheap API call to check connection - getting models list or health check
            # For OpenAI-compatible APIs, we can use the models endpoint
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Try to get models list (a cheap authenticated call)
            response = await self.client.get(
                f"{self.endpoint.rsplit('/audio/transcriptions', 1)[0]}/models",
                headers=headers
            )

            # Raises httpx.HTTPStatusError for any non-2xx status, handled below.
            response.raise_for_status()

            return {
                "success": True,
                "message": "Successfully connected to STT API",
                "status_code": response.status_code
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "message": "Connection timed out. Check your network and API endpoint.",
                "error": "timeout"
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "message": "Could not connect to the API endpoint. Check the URL and your network.",
                "error": "connection_error"
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return {
                    "success": False,
                    "message": "Authentication failed. Check your API key.",
                    "status_code": e.response.status_code,
                    "error": "unauthorized"
                }
            elif e.response.status_code == 404:
                return {
                    "success": False,
                    "message": "API endpoint not found. Check the URL.",
                    "status_code": e.response.status_code,
                    "error": "not_found"
                }
            else:
                return {
                    "success": False,
                    "message": f"API returned error: {e.response.status_code}",
                    "status_code": e.response.status_code,
                    "error": e.response.text
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Unexpected error: {e!s}",
                "error": str(e)
            }

    async def transcribe(self, audio_data: bytes, **kwargs) -> dict[str, Any]:
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
        pass
