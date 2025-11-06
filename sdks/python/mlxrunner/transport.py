"""
Transport layer for MLXR Python SDK.

Handles communication via Unix Domain Socket and HTTP.
"""

import json
import os
import socket
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union
from urllib.parse import urljoin, urlparse

import httpx

from .exceptions import (
    MLXRConnectionError,
    MLXRStreamError,
    MLXRTimeoutError,
    raise_for_status,
)


class UDSTransport(httpx.HTTPTransport):
    """Custom HTTPTransport for Unix Domain Socket connections."""

    def __init__(self, socket_path: str, **kwargs: Any) -> None:
        self.socket_path = socket_path
        super().__init__(uds=socket_path, **kwargs)


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

        # Set up client based on connection type
        if base_url:
            # HTTP connection
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self.client = httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        else:
            # Unix Domain Socket connection
            if not os.path.exists(self.socket_path):
                raise MLXRConnectionError(
                    f"Socket not found at {self.socket_path}. Is the daemon running?"
                )
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
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

                raise_for_status(response.status_code, error_message, error_data if 'error_data' in locals() else None)

            # Parse response
            if response.status_code == 204:
                return {}

            return response.json()

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Request timed out: {e}")
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}")
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}")

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
                        except json.JSONDecodeError:
                            # Handle non-JSON lines (like event: message)
                            continue
                    # Handle plain JSON lines (Ollama format)
                    elif line.startswith("{"):
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise MLXRStreamError(f"Failed to parse JSON: {e}")

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Stream timed out: {e}")
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}")
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}")

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

        # Set up async client based on connection type
        if base_url:
            # HTTP connection
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self.client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        else:
            # Unix Domain Socket connection
            if not os.path.exists(self.socket_path):
                raise MLXRConnectionError(
                    f"Socket not found at {self.socket_path}. Is the daemon running?"
                )
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

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
                raise_for_status(response.status_code, error_message)

            # Parse response
            if response.status_code == 204:
                return {}

            return response.json()

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Request timed out: {e}")
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}")
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}")

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
                            continue
                    # Handle plain JSON lines (Ollama format)
                    elif line.startswith("{"):
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise MLXRStreamError(f"Failed to parse JSON: {e}")

        except httpx.TimeoutException as e:
            raise MLXRTimeoutError(f"Stream timed out: {e}")
        except httpx.ConnectError as e:
            raise MLXRConnectionError(f"Failed to connect: {e}")
        except httpx.HTTPError as e:
            raise MLXRConnectionError(f"HTTP error: {e}")

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
