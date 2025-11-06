"""
Type definitions for MLXR Python SDK.

Provides Pydantic models for OpenAI and Ollama API compatibility.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ============================================================================
# OpenAI-Compatible Types
# ============================================================================


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: Literal["system", "user", "assistant", "function"]
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionChunkChoice(BaseModel):
    """A choice in a streaming chat completion chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class CompletionChoice(BaseModel):
    """A choice in a text completion response."""

    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class Completion(BaseModel):
    """Text completion response."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None


class Embedding(BaseModel):
    """An embedding vector."""

    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: Literal["list"] = "list"
    data: List[Embedding]
    model: str
    usage: Usage


class Model(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """List of models."""

    object: Literal["list"] = "list"
    data: List[Model]


# ============================================================================
# Ollama-Compatible Types
# ============================================================================


class OllamaMessage(BaseModel):
    """Ollama chat message."""

    role: Literal["system", "user", "assistant"]
    content: str
    images: Optional[List[str]] = None


class OllamaGenerateRequest(BaseModel):
    """Ollama generate request."""

    model: str
    prompt: str
    images: Optional[List[str]] = None
    format: Optional[Literal["json"]] = None
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    keep_alive: Optional[str] = None


class OllamaGenerateResponse(BaseModel):
    """Ollama generate response."""

    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaChatRequest(BaseModel):
    """Ollama chat request."""

    model: str
    messages: List[OllamaMessage]
    format: Optional[Literal["json"]] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = True
    keep_alive: Optional[str] = None


class OllamaChatResponse(BaseModel):
    """Ollama chat response."""

    model: str
    created_at: str
    message: OllamaMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaEmbeddingRequest(BaseModel):
    """Ollama embedding request."""

    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None


class OllamaEmbeddingResponse(BaseModel):
    """Ollama embedding response."""

    embedding: List[float]


class OllamaModelDetails(BaseModel):
    """Ollama model details."""

    parent_model: str = ""
    format: str
    family: str
    families: Optional[List[str]] = None
    parameter_size: str
    quantization_level: str


class OllamaModel(BaseModel):
    """Ollama model information."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaModelList(BaseModel):
    """List of Ollama models."""

    models: List[OllamaModel]


class OllamaPullRequest(BaseModel):
    """Ollama pull request."""

    name: str
    insecure: bool = False
    stream: bool = True


class OllamaPullResponse(BaseModel):
    """Ollama pull response."""

    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None


class OllamaShowRequest(BaseModel):
    """Ollama show request."""

    name: str


class OllamaShowResponse(BaseModel):
    """Ollama show response."""

    modelfile: str
    parameters: str
    template: str
    details: OllamaModelDetails
    model_info: Optional[Dict[str, Any]] = None


class OllamaCreateRequest(BaseModel):
    """Ollama create request."""

    name: str
    modelfile: str
    stream: bool = True
    path: Optional[str] = None


class OllamaCreateResponse(BaseModel):
    """Ollama create response."""

    status: str


class OllamaCopyRequest(BaseModel):
    """Ollama copy request."""

    source: str
    destination: str


class OllamaDeleteRequest(BaseModel):
    """Ollama delete request."""

    name: str


class OllamaProcessModel(BaseModel):
    """Running model process information."""

    name: str
    model: str
    size: int
    digest: str
    details: OllamaModelDetails
    expires_at: str
    size_vram: int


class OllamaProcessList(BaseModel):
    """List of running models."""

    models: List[OllamaProcessModel]


# ============================================================================
# MLXR-Specific Types
# ============================================================================


class MLXRHealth(BaseModel):
    """MLXR health check response."""

    status: Literal["ok", "error"]
    version: Optional[str] = None
    uptime: Optional[int] = None


class MLXRMetrics(BaseModel):
    """MLXR metrics."""

    requests_total: int
    requests_active: int
    tokens_generated: int
    throughput_tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    kv_cache_usage_bytes: int
    kv_cache_hit_rate: float
    memory_allocated_bytes: int
    memory_peak_bytes: int


# ============================================================================
# Request Parameter Types
# ============================================================================


class ChatCompletionRequest(BaseModel):
    """Chat completion request parameters."""

    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None


class CompletionRequest(BaseModel):
    """Text completion request parameters."""

    model: str
    prompt: Union[str, List[str]]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False
    best_of: Optional[int] = None


class EmbeddingRequest(BaseModel):
    """Embedding request parameters."""

    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
