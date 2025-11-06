"""
Main client for MLXR Python SDK.
"""

from typing import Optional

from .ollama_api import OllamaAPI
from .openai_api import OpenAIAPI
from .transport import BaseTransport


class MLXR:
    """
    Main MLXR client.

    Provides access to both OpenAI-compatible and Ollama-compatible APIs.

    Example:
        >>> client = MLXR()
        >>> response = client.chat.completions.create(
        ...     model="TinyLlama-1.1B",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)

    Attributes:
        chat: OpenAI chat API
        completions: OpenAI completions API
        embeddings: OpenAI embeddings API
        models: OpenAI models API
        ollama: Ollama-compatible API
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize MLXR client.

        Args:
            socket_path: Path to Unix Domain Socket (default: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock)
            base_url: HTTP base URL (e.g., "http://localhost:11434")
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.transport = BaseTransport(
            socket_path=socket_path,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Initialize OpenAI API
        openai_api = OpenAIAPI(self.transport)
        self.chat = openai_api.chat
        self.completions = openai_api.completions
        self.embeddings = openai_api.embeddings
        self.models = openai_api.models

        # Initialize Ollama API
        self.ollama = OllamaAPI(self.transport)

    def health(self) -> dict:
        """
        Check daemon health.

        Returns:
            Health status dict
        """
        return self.transport.get("/health")

    def metrics(self) -> dict:
        """
        Get daemon metrics.

        Returns:
            Metrics dict
        """
        return self.transport.get("/metrics")

    def close(self) -> None:
        """Close the client connection."""
        self.transport.close()

    def __enter__(self) -> "MLXR":
        return self

    def __exit__(self, *args) -> None:
        self.close()
