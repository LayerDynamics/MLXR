"""
Async client for MLXR Python SDK.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .transport import AsyncTransport
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    EmbeddingResponse,
    Model,
    ModelList,
    OllamaChatResponse,
    OllamaCreateResponse,
    OllamaEmbeddingResponse,
    OllamaGenerateResponse,
    OllamaModelList,
    OllamaProcessList,
    OllamaPullResponse,
    OllamaShowResponse,
)


class AsyncChatCompletions:
    """Async OpenAI chat completions API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.transport = transport

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create an async chat completion."""
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
        }

        if stop is not None:
            request_data["stop"] = stop
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if presence_penalty != 0.0:
            request_data["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            request_data["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            request_data["logit_bias"] = logit_bias
        if user is not None:
            request_data["user"] = user

        request_data.update(kwargs)

        if stream:
            return self._create_stream(request_data)
        else:
            return await self._create(request_data)

    async def _create(self, request_data: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        response = await self.transport.post("/v1/chat/completions", json_data=request_data)
        return ChatCompletion(**response)

    async def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create a streaming chat completion."""
        async for chunk_data in self.transport.stream(
            "/v1/chat/completions", json_data=request_data
        ):
            yield ChatCompletionChunk(**chunk_data)


class AsyncChat:
    """Async OpenAI chat API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.completions = AsyncChatCompletions(transport)


class AsyncCompletions:
    """Async OpenAI text completions API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.transport = transport

    async def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        logprobs: Optional[int] = None,
        echo: bool = False,
        best_of: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Completion, AsyncIterator[Completion]]:
        """Create an async text completion."""
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
        }

        if stop is not None:
            request_data["stop"] = stop
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if presence_penalty != 0.0:
            request_data["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            request_data["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            request_data["logit_bias"] = logit_bias
        if user is not None:
            request_data["user"] = user
        if logprobs is not None:
            request_data["logprobs"] = logprobs
        if echo:
            request_data["echo"] = echo
        if best_of is not None:
            request_data["best_of"] = best_of

        request_data.update(kwargs)

        if stream:
            return self._create_stream(request_data)
        else:
            return await self._create(request_data)

    async def _create(self, request_data: Dict[str, Any]) -> Completion:
        """Create a non-streaming completion."""
        response = await self.transport.post("/v1/completions", json_data=request_data)
        return Completion(**response)

    async def _create_stream(self, request_data: Dict[str, Any]) -> AsyncIterator[Completion]:
        """Create a streaming completion."""
        async for chunk_data in self.transport.stream("/v1/completions", json_data=request_data):
            yield Completion(**chunk_data)


class AsyncEmbeddings:
    """Async OpenAI embeddings API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.transport = transport

    async def create(
        self,
        model: str,
        input: Union[str, List[str]],
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create async embeddings."""
        request_data = {
            "model": model,
            "input": input,
        }

        if user is not None:
            request_data["user"] = user

        request_data.update(kwargs)

        response = await self.transport.post("/v1/embeddings", json_data=request_data)
        return EmbeddingResponse(**response)


class AsyncModels:
    """Async OpenAI models API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.transport = transport

    async def list(self) -> ModelList:
        """List available models async."""
        response = await self.transport.get("/v1/models")
        return ModelList(**response)

    async def retrieve(self, model_id: str) -> Model:
        """Retrieve a specific model async."""
        response = await self.transport.get(f"/v1/models/{model_id}")
        return Model(**response)


class AsyncOllamaAPI:
    """Async Ollama-compatible API client."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.transport = transport

    async def generate(
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
    ) -> Union[OllamaGenerateResponse, AsyncIterator[OllamaGenerateResponse]]:
        """Generate an async completion."""
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
            return await self._generate(request_data)

    async def _generate(self, request_data: Dict[str, Any]) -> OllamaGenerateResponse:
        """Generate a non-streaming completion."""
        response = await self.transport.post("/api/generate", json_data=request_data)
        return OllamaGenerateResponse(**response)

    async def _generate_stream(
        self, request_data: Dict[str, Any]
    ) -> AsyncIterator[OllamaGenerateResponse]:
        """Generate a streaming completion."""
        async for chunk_data in self.transport.stream("/api/generate", json_data=request_data):
            yield OllamaGenerateResponse(**chunk_data)

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[OllamaChatResponse, AsyncIterator[OllamaChatResponse]]:
        """Async chat with a model."""
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
            return await self._chat(request_data)

    async def _chat(self, request_data: Dict[str, Any]) -> OllamaChatResponse:
        """Chat non-streaming."""
        response = await self.transport.post("/api/chat", json_data=request_data)
        return OllamaChatResponse(**response)

    async def _chat_stream(
        self, request_data: Dict[str, Any]
    ) -> AsyncIterator[OllamaChatResponse]:
        """Chat streaming."""
        async for chunk_data in self.transport.stream("/api/chat", json_data=request_data):
            yield OllamaChatResponse(**chunk_data)

    async def embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        **kwargs: Any,
    ) -> OllamaEmbeddingResponse:
        """Generate async embeddings."""
        request_data = {
            "model": model,
            "prompt": prompt,
        }

        if options is not None:
            request_data["options"] = options
        if keep_alive is not None:
            request_data["keep_alive"] = keep_alive

        request_data.update(kwargs)

        response = await self.transport.post("/api/embeddings", json_data=request_data)
        return OllamaEmbeddingResponse(**response)

    async def pull(
        self,
        name: str,
        insecure: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[OllamaPullResponse, AsyncIterator[OllamaPullResponse]]:
        """Async pull a model."""
        request_data = {
            "name": name,
            "insecure": insecure,
            "stream": stream,
        }
        request_data.update(kwargs)

        if stream:
            return self._pull_stream(request_data)
        else:
            return await self._pull(request_data)

    async def _pull(self, request_data: Dict[str, Any]) -> OllamaPullResponse:
        """Pull non-streaming."""
        response = await self.transport.post("/api/pull", json_data=request_data)
        return OllamaPullResponse(**response)

    async def _pull_stream(
        self, request_data: Dict[str, Any]
    ) -> AsyncIterator[OllamaPullResponse]:
        """Pull streaming."""
        async for chunk_data in self.transport.stream("/api/pull", json_data=request_data):
            yield OllamaPullResponse(**chunk_data)

    async def create(
        self,
        name: str,
        modelfile: str,
        stream: bool = False,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[OllamaCreateResponse, AsyncIterator[OllamaCreateResponse]]:
        """Async create a model."""
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
            return await self._create(request_data)

    async def _create(self, request_data: Dict[str, Any]) -> OllamaCreateResponse:
        """Create non-streaming."""
        response = await self.transport.post("/api/create", json_data=request_data)
        return OllamaCreateResponse(**response)

    async def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> AsyncIterator[OllamaCreateResponse]:
        """Create streaming."""
        async for chunk_data in self.transport.stream("/api/create", json_data=request_data):
            yield OllamaCreateResponse(**chunk_data)

    async def copy(self, source: str, destination: str, **kwargs: Any) -> Dict[str, Any]:
        """Async copy a model."""
        request_data = {"source": source, "destination": destination}
        request_data.update(kwargs)
        return await self.transport.post("/api/copy", json_data=request_data)

    async def delete(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """Async delete a model."""
        request_data = {"name": name}
        request_data.update(kwargs)
        return await self.transport.delete("/api/delete", json_data=request_data)

    async def show(self, name: str, **kwargs: Any) -> OllamaShowResponse:
        """Async show model information."""
        request_data = {"name": name}
        request_data.update(kwargs)
        response = await self.transport.post("/api/show", json_data=request_data)
        return OllamaShowResponse(**response)

    async def list(self) -> OllamaModelList:
        """Async list local models."""
        response = await self.transport.get("/api/tags")
        return OllamaModelList(**response)

    async def ps(self) -> OllamaProcessList:
        """Async list running models."""
        response = await self.transport.get("/api/ps")
        return OllamaProcessList(**response)


class AsyncMLXR:
    """
    Async MLXR client.

    Provides async access to both OpenAI-compatible and Ollama-compatible APIs.

    Example:
        >>> async with AsyncMLXR() as client:
        ...     response = await client.chat.completions.create(
        ...         model="TinyLlama-1.1B",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     print(response.choices[0].message.content)

    Attributes:
        chat: Async OpenAI chat API
        completions: Async OpenAI completions API
        embeddings: Async OpenAI embeddings API
        models: Async OpenAI models API
        ollama: Async Ollama-compatible API
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize async MLXR client.

        Args:
            socket_path: Path to Unix Domain Socket
            base_url: HTTP base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.transport = AsyncTransport(
            socket_path=socket_path,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Initialize async OpenAI API
        self.chat = AsyncChat(self.transport)
        self.completions = AsyncCompletions(self.transport)
        self.embeddings = AsyncEmbeddings(self.transport)
        self.models = AsyncModels(self.transport)

        # Initialize async Ollama API
        self.ollama = AsyncOllamaAPI(self.transport)

    async def health(self) -> dict:
        """Async check daemon health."""
        return await self.transport.get("/health")

    async def metrics(self) -> dict:
        """Async get daemon metrics."""
        return await self.transport.get("/metrics")

    async def close(self) -> None:
        """Close the async client connection."""
        await self.transport.close()

    async def __aenter__(self) -> "AsyncMLXR":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
