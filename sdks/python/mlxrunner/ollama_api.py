"""
Ollama-compatible API client for MLXR.

Provides Ollama API compatibility for generation, chat, and model management.
"""

from typing import Any, Dict, Iterator, List, Optional, Union

from .transport import BaseTransport
from .types import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaCopyRequest,
    OllamaCreateRequest,
    OllamaCreateResponse,
    OllamaDeleteRequest,
    OllamaEmbeddingRequest,
    OllamaEmbeddingResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaModel,
    OllamaModelList,
    OllamaProcessList,
    OllamaPullRequest,
    OllamaPullResponse,
    OllamaShowRequest,
    OllamaShowResponse,
)


class OllamaAPI:
    """
    Ollama-compatible API client.

    Provides access to generation, chat, embeddings, and model management.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport

    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        keep_alive: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[OllamaGenerateResponse, Iterator[OllamaGenerateResponse]]:
        """
        Generate a completion.

        Args:
            model: Model name
            prompt: Prompt to generate completion for
            images: Optional list of base64-encoded images
            format: Response format (e.g., "json")
            options: Model parameters (temperature, top_k, etc.)
            system: System prompt
            template: Prompt template
            context: Context from previous response
            stream: Whether to stream responses
            raw: Whether to use raw mode (no prompt formatting)
            keep_alive: How long to keep model loaded
            **kwargs: Additional parameters

        Returns:
            OllamaGenerateResponse or Iterator[OllamaGenerateResponse]
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }

        if images is not None:
            request_data["images"] = images
        if format is not None:
            request_data["format"] = format
        if options is not None:
            request_data["options"] = options
        if system is not None:
            request_data["system"] = system
        if template is not None:
            request_data["template"] = template
        if context is not None:
            request_data["context"] = context
        if keep_alive is not None:
            request_data["keep_alive"] = keep_alive

        request_data.update(kwargs)

        if stream:
            return self._generate_stream(request_data)
        else:
            return self._generate(request_data)

    def _generate(self, request_data: Dict[str, Any]) -> OllamaGenerateResponse:
        """Generate a non-streaming completion."""
        response = self.transport.post("/api/generate", json_data=request_data)
        return OllamaGenerateResponse(**response)

    def _generate_stream(
        self, request_data: Dict[str, Any]
    ) -> Iterator[OllamaGenerateResponse]:
        """Generate a streaming completion."""
        for chunk_data in self.transport.stream("/api/generate", json_data=request_data):
            yield OllamaGenerateResponse(**chunk_data)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[OllamaChatResponse, Iterator[OllamaChatResponse]]:
        """
        Chat with a model.

        Args:
            model: Model name
            messages: List of messages in the conversation
            format: Response format (e.g., "json")
            options: Model parameters (temperature, top_k, etc.)
            stream: Whether to stream responses
            keep_alive: How long to keep model loaded
            **kwargs: Additional parameters

        Returns:
            OllamaChatResponse or Iterator[OllamaChatResponse]
        """
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if format is not None:
            request_data["format"] = format
        if options is not None:
            request_data["options"] = options
        if keep_alive is not None:
            request_data["keep_alive"] = keep_alive

        request_data.update(kwargs)

        if stream:
            return self._chat_stream(request_data)
        else:
            return self._chat(request_data)

    def _chat(self, request_data: Dict[str, Any]) -> OllamaChatResponse:
        """Chat non-streaming."""
        response = self.transport.post("/api/chat", json_data=request_data)
        return OllamaChatResponse(**response)

    def _chat_stream(self, request_data: Dict[str, Any]) -> Iterator[OllamaChatResponse]:
        """Chat streaming."""
        for chunk_data in self.transport.stream("/api/chat", json_data=request_data):
            yield OllamaChatResponse(**chunk_data)

    def embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        **kwargs: Any,
    ) -> OllamaEmbeddingResponse:
        """
        Generate embeddings.

        Args:
            model: Model name
            prompt: Text to embed
            options: Model parameters
            keep_alive: How long to keep model loaded
            **kwargs: Additional parameters

        Returns:
            OllamaEmbeddingResponse
        """
        request_data = {
            "model": model,
            "prompt": prompt,
        }

        if options is not None:
            request_data["options"] = options
        if keep_alive is not None:
            request_data["keep_alive"] = keep_alive

        request_data.update(kwargs)

        response = self.transport.post("/api/embeddings", json_data=request_data)
        return OllamaEmbeddingResponse(**response)

    def pull(
        self,
        name: str,
        insecure: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[OllamaPullResponse, Iterator[OllamaPullResponse]]:
        """
        Pull a model from the registry.

        Args:
            name: Model name
            insecure: Allow insecure connections
            stream: Whether to stream progress
            **kwargs: Additional parameters

        Returns:
            OllamaPullResponse or Iterator[OllamaPullResponse]
        """
        request_data = {
            "name": name,
            "insecure": insecure,
            "stream": stream,
        }
        request_data.update(kwargs)

        if stream:
            return self._pull_stream(request_data)
        else:
            return self._pull(request_data)

    def _pull(self, request_data: Dict[str, Any]) -> OllamaPullResponse:
        """Pull non-streaming."""
        response = self.transport.post("/api/pull", json_data=request_data)
        return OllamaPullResponse(**response)

    def _pull_stream(self, request_data: Dict[str, Any]) -> Iterator[OllamaPullResponse]:
        """Pull streaming."""
        for chunk_data in self.transport.stream("/api/pull", json_data=request_data):
            yield OllamaPullResponse(**chunk_data)

    def create(
        self,
        name: str,
        modelfile: str,
        stream: bool = False,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[OllamaCreateResponse, Iterator[OllamaCreateResponse]]:
        """
        Create a model from a Modelfile.

        Args:
            name: Model name
            modelfile: Modelfile contents
            stream: Whether to stream progress
            path: Path to Modelfile
            **kwargs: Additional parameters

        Returns:
            OllamaCreateResponse or Iterator[OllamaCreateResponse]
        """
        request_data = {
            "name": name,
            "modelfile": modelfile,
            "stream": stream,
        }

        if path is not None:
            request_data["path"] = path

        request_data.update(kwargs)

        if stream:
            return self._create_stream(request_data)
        else:
            return self._create(request_data)

    def _create(self, request_data: Dict[str, Any]) -> OllamaCreateResponse:
        """Create non-streaming."""
        response = self.transport.post("/api/create", json_data=request_data)
        return OllamaCreateResponse(**response)

    def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> Iterator[OllamaCreateResponse]:
        """Create streaming."""
        for chunk_data in self.transport.stream("/api/create", json_data=request_data):
            yield OllamaCreateResponse(**chunk_data)

    def copy(self, source: str, destination: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Copy a model.

        Args:
            source: Source model name
            destination: Destination model name
            **kwargs: Additional parameters

        Returns:
            Empty dict on success
        """
        request_data = {"source": source, "destination": destination}
        request_data.update(kwargs)
        return self.transport.post("/api/copy", json_data=request_data)

    def delete(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Delete a model.

        Args:
            name: Model name
            **kwargs: Additional parameters

        Returns:
            Empty dict on success
        """
        request_data = {"name": name}
        request_data.update(kwargs)
        return self.transport.delete("/api/delete", json_data=request_data)

    def show(self, name: str, **kwargs: Any) -> OllamaShowResponse:
        """
        Show model information.

        Args:
            name: Model name
            **kwargs: Additional parameters

        Returns:
            OllamaShowResponse
        """
        request_data = {"name": name}
        request_data.update(kwargs)
        response = self.transport.post("/api/show", json_data=request_data)
        return OllamaShowResponse(**response)

    def list(self) -> OllamaModelList:
        """
        List local models.

        Returns:
            OllamaModelList
        """
        response = self.transport.get("/api/tags")
        return OllamaModelList(**response)

    def ps(self) -> OllamaProcessList:
        """
        List running models.

        Returns:
            OllamaProcessList
        """
        response = self.transport.get("/api/ps")
        return OllamaProcessList(**response)
