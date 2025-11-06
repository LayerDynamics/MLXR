"""
OpenAI-compatible API client for MLXR.

Provides OpenAI API compatibility for chat, completions, and embeddings.
"""

import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from .transport import BaseTransport
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatMessage,
    Completion,
    CompletionChoice,
    Embedding,
    EmbeddingResponse,
    Model,
    ModelList,
    Usage,
)


class ChatCompletions:
    """OpenAI chat completions API."""

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport

    def create(
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
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.

        Args:
            model: Model ID to use
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stream: Whether to stream responses
            stop: Stop sequences
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            logit_bias: Token bias adjustments
            user: User identifier
            **kwargs: Additional parameters

        Returns:
            ChatCompletion or Iterator[ChatCompletionChunk]
        """
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

        request_data |= kwargs

        if stream:
            return self._create_stream(request_data)
        else:
            return self._create(request_data)

    def _create(self, request_data: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        response = self.transport.post("/v1/chat/completions", json_data=request_data)
        return ChatCompletion(**response)

    def _create_stream(self, request_data: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """Create a streaming chat completion."""
        for chunk_data in self.transport.stream("/v1/chat/completions", json_data=request_data):
            yield ChatCompletionChunk(**chunk_data)


class Chat:
    """OpenAI chat API."""

    def __init__(self, transport: BaseTransport) -> None:
        self.completions = ChatCompletions(transport)


class Completions:
    """OpenAI text completions API."""

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport

    def create(
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
    ) -> Union[Completion, Iterator[Completion]]:
        """
        Create a text completion.

        Args:
            model: Model ID to use
            prompt: Prompt(s) to generate completions for
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stream: Whether to stream responses
            stop: Stop sequences
            max_tokens: Maximum tokens to generate
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            logit_bias: Token bias adjustments
            user: User identifier
            logprobs: Include log probabilities
            echo: Echo back the prompt
            best_of: Generate best_of completions and return the best
            **kwargs: Additional parameters

        Returns:
            Completion or Iterator[Completion]
        """
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

        request_data |= kwargs

        if stream:
            return self._create_stream(request_data)
        else:
            return self._create(request_data)

    def _create(self, request_data: Dict[str, Any]) -> Completion:
        """Create a non-streaming completion."""
        response = self.transport.post("/v1/completions", json_data=request_data)
        return Completion(**response)

    def _create_stream(self, request_data: Dict[str, Any]) -> Iterator[Completion]:
        """Create a streaming completion."""
        for chunk_data in self.transport.stream("/v1/completions", json_data=request_data):
            yield Completion(**chunk_data)


class Embeddings:
    """OpenAI embeddings API."""

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport

    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Create embeddings.

        Args:
            model: Model ID to use
            input: Text or list of texts to embed
            user: User identifier
            **kwargs: Additional parameters

        Returns:
            EmbeddingResponse
        """
        request_data = {
            "model": model,
            "input": input,
        }

        if user is not None:
            request_data["user"] = user

        request_data |= kwargs

        response = self.transport.post("/v1/embeddings", json_data=request_data)
        return EmbeddingResponse(**response)


class Models:
    """OpenAI models API."""

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport

    def list(self) -> ModelList:
        """
        List available models.

        Returns:
            ModelList
        """
        response = self.transport.get("/v1/models")
        return ModelList(**response)

    def retrieve(self, model_id: str) -> Model:
        """
        Retrieve a specific model.

        Args:
            model_id: Model ID

        Returns:
            Model
        """
        response = self.transport.get(f"/v1/models/{model_id}")
        return Model(**response)


class OpenAIAPI:
    """
    OpenAI-compatible API client.

    Provides access to chat, completions, embeddings, and models endpoints.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self.transport = transport
        self.chat = Chat(transport)
        self.completions = Completions(transport)
        self.embeddings = Embeddings(transport)
        self.models = Models(transport)
