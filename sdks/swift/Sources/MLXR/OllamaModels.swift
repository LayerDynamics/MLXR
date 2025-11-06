// Copyright © 2025 MLXR Development
// Ollama-compatible API models

import Foundation

// MARK: - Generate

/// Ollama generate request
public struct OllamaGenerateRequest: Codable, Sendable {
    public let model: String
    public let prompt: String
    public let system: String?
    public let template: String?
    public let context: String?
    public let stream: Bool?
    public let raw: Bool?
    public let format: String?
    public let numPredict: Int?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Float?
    public let repeatPenalty: Float?
    public let seed: Int?
    public let stop: [String]?

    enum CodingKeys: String, CodingKey {
        case model, prompt, system, template, context, stream, raw, format, temperature, seed, stop
        case topP = "top_p"
        case topK = "top_k"
        case numPredict = "num_predict"
        case repeatPenalty = "repeat_penalty"
    }
}

/// Ollama generate response
public struct OllamaGenerateResponse: Codable, Sendable {
    public let model: String
    public let createdAt: String
    public let response: String
    public let done: Bool
    public let context: String?
    public let totalDuration: Int64?
    public let loadDuration: Int64?
    public let promptEvalCount: Int?
    public let promptEvalDuration: Int64?
    public let evalCount: Int?
    public let evalDuration: Int64?

    enum CodingKeys: String, CodingKey {
        case model, response, done, context
        case createdAt = "created_at"
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }
}

// MARK: - Chat

/// Ollama chat message
public struct OllamaChatMessage: Codable, Sendable {
    public let role: String
    public let content: String
    public let images: [String]?
}

/// Ollama chat request
public struct OllamaChatRequest: Codable, Sendable {
    public let model: String
    public let messages: [OllamaChatMessage]
    public let stream: Bool?
    public let format: String?
    public let numPredict: Int?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Float?
    public let repeatPenalty: Float?
    public let seed: Int?
    public let stop: [String]?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, format, temperature, seed, stop
        case numPredict = "num_predict"
        case topP = "top_p"
        case topK = "top_k"
        case repeatPenalty = "repeat_penalty"
    }
}

/// Ollama chat response
public struct OllamaChatResponse: Codable, Sendable {
    public let model: String
    public let createdAt: String
    public let message: OllamaChatMessage
    public let done: Bool
    public let totalDuration: Int64?
    public let loadDuration: Int64?
    public let promptEvalCount: Int?
    public let promptEvalDuration: Int64?
    public let evalCount: Int?
    public let evalDuration: Int64?

    enum CodingKeys: String, CodingKey {
        case model, message, done
        case createdAt = "created_at"
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }
}

// MARK: - Embeddings

/// Ollama embeddings request
public struct OllamaEmbeddingsRequest: Codable, Sendable {
    public let model: String
    public let prompt: String
}

/// Ollama embeddings response
public struct OllamaEmbeddingsResponse: Codable, Sendable {
    public let embedding: [Float]
}

// MARK: - Model Management

/// Ollama pull request
public struct OllamaPullRequest: Codable, Sendable {
    public let name: String
    public let insecure: Bool?
    public let stream: Bool?
}

/// Ollama pull response (streaming)
public struct OllamaPullResponse: Codable, Sendable {
    public let status: String
    public let digest: String?
    public let total: Int64?
    public let completed: Int64?
}

/// Ollama create request
public struct OllamaCreateRequest: Codable, Sendable {
    public let name: String
    public let modelfile: String?
    public let path: String?
    public let stream: Bool?
}

/// Ollama create response (streaming)
public struct OllamaCreateResponse: Codable, Sendable {
    public let status: String
}

/// Ollama model details
public struct OllamaModelDetails: Codable, Sendable {
    public let format: String
    public let family: String
    public let families: [String]
    public let parameterSize: String
    public let quantizationLevel: String

    enum CodingKeys: String, CodingKey {
        case format, family, families
        case parameterSize = "parameter_size"
        case quantizationLevel = "quantization_level"
    }
}

/// Ollama model info
public struct OllamaModelInfo: Codable, Sendable {
    public let name: String
    public let modifiedAt: String
    public let size: Int64
    public let digest: String
    public let details: OllamaModelDetails?

    enum CodingKeys: String, CodingKey {
        case name, size, digest, details
        case modifiedAt = "modified_at"
    }
}

/// Ollama tags response (model list)
public struct OllamaTagsResponse: Codable, Sendable {
    public let models: [OllamaModelInfo]
}

/// Ollama running model
public struct OllamaRunningModel: Codable, Sendable {
    public let name: String
    public let model: String
    public let size: Int64
    public let digest: String
    public let details: OllamaModelDetails?
    public let expiresAt: String?
    public let sizeVRAM: Int64?

    enum CodingKeys: String, CodingKey {
        case name, model, size, digest, details
        case expiresAt = "expires_at"
        case sizeVRAM = "size_vram"
    }
}

/// Ollama process response (running models)
public struct OllamaProcessResponse: Codable, Sendable {
    public let models: [OllamaRunningModel]
}

/// Ollama show request
public struct OllamaShowRequest: Codable, Sendable {
    public let name: String
}

/// Ollama show response
public struct OllamaShowResponse: Codable, Sendable {
    public let modelfile: String
    public let parameters: String
    public let template: String
    public let details: OllamaModelDetails?
}

/// Ollama copy request
public struct OllamaCopyRequest: Codable, Sendable {
    public let source: String
    public let destination: String
}

/// Ollama delete request
public struct OllamaDeleteRequest: Codable, Sendable {
    public let name: String
}
