"""
Exception classes for MLXR Python SDK.
"""

from typing import Any, Dict, Optional


class MLXRError(Exception):
    """Base exception for MLXR SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class MLXRConnectionError(MLXRError):
    """Error connecting to MLXR daemon."""

    def __init__(self, message: str = "Failed to connect to MLXR daemon") -> None:
        super().__init__(message)


class MLXRTimeoutError(MLXRError):
    """Request to MLXR daemon timed out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class MLXRAPIError(MLXRError):
    """Error from MLXR API."""

    pass


class MLXRAuthenticationError(MLXRAPIError):
    """Authentication error."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class MLXRPermissionError(MLXRAPIError):
    """Permission denied error."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(message, status_code=403)


class MLXRNotFoundError(MLXRAPIError):
    """Resource not found error."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class MLXRRateLimitError(MLXRAPIError):
    """Rate limit exceeded error."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)


class MLXRServerError(MLXRAPIError):
    """Server error."""

    def __init__(self, message: str = "Internal server error") -> None:
        super().__init__(message, status_code=500)


class MLXRValidationError(MLXRError):
    """Request validation error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class MLXRModelError(MLXRError):
    """Model-related error."""

    pass


class MLXRStreamError(MLXRError):
    """Streaming error."""

    pass


def raise_for_status(status_code: int, message: str, response: Optional[Dict[str, Any]] = None) -> None:
    """Raise appropriate exception based on status code."""
    if status_code == 401:
        raise MLXRAuthenticationError(message)
    elif status_code == 403:
        raise MLXRPermissionError(message)
    elif status_code == 404:
        raise MLXRNotFoundError(message)
    elif status_code == 422:
        raise MLXRValidationError(message)
    elif status_code == 429:
        raise MLXRRateLimitError(message)
    elif status_code >= 500:
        raise MLXRServerError(message)
    elif status_code >= 400:
        raise MLXRAPIError(message, status_code=status_code, response=response)
