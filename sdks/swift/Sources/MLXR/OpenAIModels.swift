// Copyright © 2025 MLXR Development
// OpenAI-compatible API models

import Foundation

// MARK: - Chat Completion

/// A message in a chat completion conversation
public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String
    public let name: String?
    public let functionCall: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case functionCall = "function_call"
    }
}

/// Function definition for function calling
public struct FunctionDefinition: Codable, Sendable {
    public let name: String
    public let description: String
    public let parametersJSON: String

    enum CodingKeys: String, CodingKey {
        case name, description
        case parametersJSON = "parameters_json"
    }
}

/// Tool definition
public struct ToolDefinition: Codable, Sendable {
    public let type: String
    public let function: FunctionDefinition
}

/// Chat completion request
public struct ChatCompletionRequest: Codable, Sendable {
    public let model: String
    public let messages: [ChatMessage]
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let maxTokens: Int?
    public let stream: Bool?
    public let stop: [String]?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    public let n: Int?
    public let user: String?
    public let tools: [ToolDefinition]?
    public let toolChoice: String?
    public let seed: Int?

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, stop, n, user, tools, seed
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case maxTokens = "max_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case toolChoice = "tool_choice"
    }
}

/// Token usage statistics
public struct UsageInfo: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

/// Chat completion choice
public struct ChatCompletionChoice: Codable, Sendable {
    public let index: Int
    public let message: ChatMessage
    public let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }
}

/// Chat completion response (non-streaming)
public struct ChatCompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int64
    public let model: String
    public let choices: [ChatCompletionChoice]
    public let usage: UsageInfo
}

/// Streaming delta
public struct ChatCompletionDelta: Codable, Sendable {
    public let role: String?
    public let content: String?
    public let functionCall: String?

    enum CodingKeys: String, CodingKey {
        case role, content
        case functionCall = "function_call"
    }
}

/// Streaming choice
public struct ChatCompletionStreamChoice: Codable, Sendable {
    public let index: Int
    public let delta: ChatCompletionDelta
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

/// Streaming chunk
public struct ChatCompletionChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int64
    public let model: String
    public let choices: [ChatCompletionStreamChoice]
}

// MARK: - Completion (non-chat)

/// Completion request
public struct CompletionRequest: Codable, Sendable {
    public let model: String
    public let prompt: String
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let maxTokens: Int?
    public let stream: Bool?
    public let stop: [String]?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    public let n: Int?
    public let suffix: String?
    public let seed: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, stream, stop, n, suffix, seed
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case maxTokens = "max_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

/// Completion choice
public struct CompletionChoice: Codable, Sendable {
    public let index: Int
    public let text: String
    public let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

/// Completion response
public struct CompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int64
    public let model: String
    public let choices: [CompletionChoice]
    public let usage: UsageInfo
}

// MARK: - Embeddings

/// Embedding request
public struct EmbeddingRequest: Codable, Sendable {
    public let model: String
    public let input: String
    public let encodingFormat: String?
    public let user: String?

    enum CodingKeys: String, CodingKey {
        case model, input, user
        case encodingFormat = "encoding_format"
    }
}

/// Single embedding object
public struct EmbeddingObject: Codable, Sendable {
    public let index: Int
    public let embedding: [Float]
    public let object: String
}

/// Embedding response
public struct EmbeddingResponse: Codable, Sendable {
    public let object: String
    public let data: [EmbeddingObject]
    public let model: String
    public let usage: UsageInfo
}

// MARK: - Models

/// Model information
public struct ModelInfo: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int64
    public let ownedBy: String

    enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

/// Model list response
public struct ModelListResponse: Codable, Sendable {
    public let object: String
    public let data: [ModelInfo]
}
