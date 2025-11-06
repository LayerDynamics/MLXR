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

    public init(role: String, content: String, name: String? = nil, functionCall: String? = nil) {
        self.role = role
        self.content = content
        self.name = name
        self.functionCall = functionCall
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

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

/// Tool definition
public struct ToolDefinition: Codable, Sendable {
    public let type: String
    public let function: FunctionDefinition

    public init(type: String = "function", function: FunctionDefinition) {
        self.type = type
        self.function = function
    }
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

    public init(
        model: String,
        messages: [ChatMessage],
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        maxTokens: Int? = nil,
        stream: Bool? = nil,
        stop: [String]? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        n: Int? = nil,
        user: String? = nil,
        tools: [ToolDefinition]? = nil,
        toolChoice: String? = nil,
        seed: Int? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.stream = stream
        self.stop = stop
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.n = n
        self.user = user
        self.tools = tools
        self.toolChoice = toolChoice
        self.seed = seed
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

    public init(promptTokens: Int = 0, completionTokens: Int = 0, totalTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
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

    public init(index: Int, message: ChatMessage, finishReason: String) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
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

    public init(
        id: String,
        object: String = "chat.completion",
        created: Int64,
        model: String,
        choices: [ChatCompletionChoice],
        usage: UsageInfo
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
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

    public init(role: String? = nil, content: String? = nil, functionCall: String? = nil) {
        self.role = role
        self.content = content
        self.functionCall = functionCall
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

    public init(index: Int, delta: ChatCompletionDelta, finishReason: String? = nil) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }
}

/// Streaming chunk
public struct ChatCompletionChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int64
    public let model: String
    public let choices: [ChatCompletionStreamChoice]

    public init(
        id: String,
        object: String = "chat.completion.chunk",
        created: Int64,
        model: String,
        choices: [ChatCompletionStreamChoice]
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
    }
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

    public init(
        model: String,
        prompt: String,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        maxTokens: Int? = nil,
        stream: Bool? = nil,
        stop: [String]? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        n: Int? = nil,
        suffix: String? = nil,
        seed: Int? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.stream = stream
        self.stop = stop
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.n = n
        self.suffix = suffix
        self.seed = seed
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

    public init(index: Int, text: String, finishReason: String) {
        self.index = index
        self.text = text
        self.finishReason = finishReason
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

    public init(
        id: String,
        object: String = "text_completion",
        created: Int64,
        model: String,
        choices: [CompletionChoice],
        usage: UsageInfo
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
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

    public init(model: String, input: String, encodingFormat: String? = nil, user: String? = nil) {
        self.model = model
        self.input = input
        self.encodingFormat = encodingFormat
        self.user = user
    }
}

/// Single embedding object
public struct EmbeddingObject: Codable, Sendable {
    public let index: Int
    public let embedding: [Float]
    public let object: String

    public init(index: Int, embedding: [Float], object: String = "embedding") {
        self.index = index
        self.embedding = embedding
        self.object = object
    }
}

/// Embedding response
public struct EmbeddingResponse: Codable, Sendable {
    public let object: String
    public let data: [EmbeddingObject]
    public let model: String
    public let usage: UsageInfo

    public init(
        object: String = "list",
        data: [EmbeddingObject],
        model: String,
        usage: UsageInfo
    ) {
        self.object = object
        self.data = data
        self.model = model
        self.usage = usage
    }
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

    public init(id: String, object: String = "model", created: Int64, ownedBy: String = "mlxr") {
        self.id = id
        self.object = object
        self.created = created
        self.ownedBy = ownedBy
    }
}

/// Model list response
public struct ModelListResponse: Codable, Sendable {
    public let object: String
    public let data: [ModelInfo]

    public init(object: String = "list", data: [ModelInfo]) {
        self.object = object
        self.data = data
    }
}
