"""
MLXR - High-performance LLM inference engine for Apple Silicon.

This is the Python SDK for MLXR, providing a Pythonic interface to the
native inference engine with both OpenAI and Ollama API compatibility.
"""

__version__ = "0.1.0"

from .async_client import AsyncMLXR
from .client import MLXR
from .exceptions import (
    MLXRAPIError,
    MLXRAuthenticationError,
    MLXRConnectionError,
    MLXRError,
    MLXRModelError,
    MLXRNotFoundError,
    MLXRPermissionError,
    MLXRRateLimitError,
    MLXRServerError,
    MLXRStreamError,
    MLXRTimeoutError,
    MLXRValidationError,
)
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Completion,
    EmbeddingResponse,
    Model,
    ModelList,
    OllamaChatResponse,
    OllamaGenerateResponse,
    OllamaModel,
    OllamaModelList,
)

__all__ = [
    # Version
    "__version__",
    # Main clients
    "MLXR",
    "AsyncMLXR",
    # Exceptions
    "MLXRError",
    "MLXRConnectionError",
    "MLXRTimeoutError",
    "MLXRAPIError",
    "MLXRAuthenticationError",
    "MLXRPermissionError",
    "MLXRNotFoundError",
    "MLXRRateLimitError",
    "MLXRServerError",
    "MLXRValidationError",
    "MLXRModelError",
    "MLXRStreamError",
    # OpenAI types
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatMessage",
    "Completion",
    "EmbeddingResponse",
    "Model",
    "ModelList",
    # Ollama types
    "OllamaChatResponse",
    "OllamaGenerateResponse",
    "OllamaModel",
    "OllamaModelList",
]
