// Copyright © 2025 MLXR Development
// Main module file

import Foundation

/// MLXR Swift SDK version
public let MLXRVersion = "0.1.0"

// Re-export all public types for convenience
@_exported import struct MLXR.MLXRClient
@_exported import struct MLXR.MLXRClientConfig
@_exported import enum MLXR.MLXRError

// OpenAI models
@_exported import struct MLXR.ChatMessage
@_exported import struct MLXR.ChatCompletionRequest
@_exported import struct MLXR.ChatCompletionResponse
@_exported import struct MLXR.ChatCompletionChunk
@_exported import struct MLXR.CompletionRequest
@_exported import struct MLXR.CompletionResponse
@_exported import struct MLXR.EmbeddingRequest
@_exported import struct MLXR.EmbeddingResponse
@_exported import struct MLXR.ModelInfo
@_exported import struct MLXR.ModelListResponse

// Ollama models
@_exported import struct MLXR.OllamaChatMessage
@_exported import struct MLXR.OllamaChatRequest
@_exported import struct MLXR.OllamaChatResponse
@_exported import struct MLXR.OllamaGenerateRequest
@_exported import struct MLXR.OllamaGenerateResponse
@_exported import struct MLXR.OllamaEmbeddingsRequest
@_exported import struct MLXR.OllamaEmbeddingsResponse
@_exported import struct MLXR.OllamaTagsResponse
@_exported import struct MLXR.OllamaModelInfo
@_exported import struct MLXR.OllamaProcessResponse
