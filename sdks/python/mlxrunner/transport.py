"""
Transport layer for MLXR Python SDK.

Handles communication via Unix Domain Socket and HTTP.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx

from .exceptions import (
    MLXRConnectionError,
    MLXRPermissionError,
    MLXRStreamError,
    MLXRTimeoutError,
    raise_for_status,
)

logger = logging.getLogger(__name__)


class UDSTransport(httpx.HTTPTransport):
    """Custom HTTPTransport for Unix Domain Socket connections."""

    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        self.socket_path = socket_path
        super().__init__(uds=socket_path, **kwargs)


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Build common HTTP headers."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _check_socket_path(socket_path: str) -> None:
    """Check if socket exists and is accessible."""
    if not os.path.exists(socket_path):
        raise MLXRConnectionError(
            f"Socket not found at {socket_path}. Is the daemon running?"
        )

    # Check read/write permissions
    if not os.access(socket_path, os.R_OK | os.W_OK):
        raise MLXRPermissionError(
            f"Permission denied for socket at {socket_path}. "
            f"Check file permissions and ownership."
        )


class BaseTransport:
    """Base transport for MLXR communication."""

    DEFAULT_SOCKET_PATH = os.path.expanduser(
        "~/Library/Application Support/MLXRunner/run/mlxrunner.sock"
    )
    DEFAULT_HTTP_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        socket_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.socket_path = socket_path or self.DEFAULT_SOCKET_PATH
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

        # Build headers once
        headers = _build_headers(api_key)

        # Set up client based on connection type
        if base_url:
            # HTTP connection
            self.client = httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        else:
            # Unix Domain Socket connection
            _check_socket_path(self.socket_path)
            self.client = httpx.Client(
                transport=UDSTransport(self.socket_path),
                base_url="http://localhost",
                headers=headers,
                timeout=timeout,
            )

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a request to the daemon."""
        try:
            response = self.client.request(
                method,
                path,
                json=json_data,
                params=params,
                **kwargs,
            )

            # Check for errors
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", response.text)
                except Exception:
                    error_message = response.text
                    error_data = None

                raise_for_status(response.status_code, error_message, error_data)
            else:
                # Parse response
                return {} if response.status_code == 204 else response.json()

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}") from e
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}") from e

    def _stream(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream responses from the daemon (SSE)."""
        try:
            with self.client.stream(
                method,
                path,
                json=json_data,
                params=params,
                **kwargs,
            ) as response:
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_message = error_data.get("error", {}).get("message", response.text)
                    except Exception:
                        error_message = response.text
                    raise_for_status(response.status_code, error_message)

                # Parse SSE stream
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # Handle SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Malformed SSE data line encountered: {data!r}")
                            continue
                    # Handle plain JSON lines (Ollama format)
                    elif line.startswith("{"):
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise MLXRStreamError(f"Failed to parse JSON: {e}") from e

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Stream timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}") from e
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}") from e

    def get(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """GET request."""
        return self._request("GET", path, **kwargs)

    def post(
        self, path: str, json_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """POST request."""
        return self._request("POST", path, json_data=json_data, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """DELETE request."""
        return self._request("DELETE", path, **kwargs)

    def stream(
        self, path: str, json_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        """Streaming POST request."""
        return self._stream("POST", path, json_data=json_data, **kwargs)

    def close(self) -> None:
        """Close the client."""
        self.client.close()

    def __enter__(self) -> "BaseTransport":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncTransport:
    """Async transport for MLXR communication."""

    DEFAULT_SOCKET_PATH = os.path.expanduser(
        "~/Library/Application Support/MLXRunner/run/mlxrunner.sock"
    )
    DEFAULT_HTTP_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        socket_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.socket_path = socket_path or self.DEFAULT_SOCKET_PATH
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

        # Build headers once
        headers = _build_headers(api_key)

        # Set up async client based on connection type
        if base_url:
            # HTTP connection
            self.client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        else:
            # Unix Domain Socket connection
            _check_socket_path(self.socket_path)

            # Use UDS transport for async client
            transport = httpx.AsyncHTTPTransport(uds=self.socket_path)
            self.client = httpx.AsyncClient(
                transport=transport,
                base_url="http://localhost",
                headers=headers,
                timeout=timeout,
            )

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make an async request to the daemon."""
        try:
            response = await self.client.request(
                method,
                path,
                json=json_data,
                params=params,
                **kwargs,
            )

            # Check for errors
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", response.text)
                except Exception:
                    error_message = response.text
                    error_data = None

                raise_for_status(response.status_code, error_message, error_data)
            else:
                # Parse response
                return {} if response.status_code == 204 else response.json()

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}") from e
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}") from e

    async def _stream(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream responses from the daemon (SSE)."""
        try:
            async with self.client.stream(
                method,
                path,
                json=json_data,
                params=params,
                **kwargs,
            ) as response:
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_message = error_data.get("error", {}).get("message", response.text)
                    except Exception:
                        error_message = response.text
                    raise_for_status(response.status_code, error_message)

                # Parse SSE stream
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # Handle SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed SSE data line encountered: {data!r}")
                            continue
                    # Handle plain JSON lines (Ollama format)
                    elif line.startswith("{"):
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise MLXRStreamError(f"Failed to parse JSON: {e}") from e

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Stream timed out: {e}") from e
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}") from e
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}") from e

    async def get(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Async GET request."""
        return await self._request("GET", path, **kwargs)

    async def post(
        self, path: str, json_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Async POST request."""
        return await self._request("POST", path, json_data=json_data, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Async DELETE request."""
        return await self._request("DELETE", path, **kwargs)

    def stream(
        self, path: str, json_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async streaming POST request."""
        return self._stream("POST", path, json_data=json_data, **kwargs)

    async def close(self) -> None:
        """Close the async client."""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncTransport":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
