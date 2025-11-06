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

    public init(
        model: String,
        prompt: String,
        system: String? = nil,
        template: String? = nil,
        context: String? = nil,
        stream: Bool? = nil,
        raw: Bool? = nil,
        format: String? = nil,
        numPredict: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Float? = nil,
        repeatPenalty: Float? = nil,
        seed: Int? = nil,
        stop: [String]? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.system = system
        self.template = template
        self.context = context
        self.stream = stream
        self.raw = raw
        self.format = format
        self.numPredict = numPredict
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.seed = seed
        self.stop = stop
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

    public init(
        model: String,
        createdAt: String,
        response: String,
        done: Bool,
        context: String? = nil,
        totalDuration: Int64? = nil,
        loadDuration: Int64? = nil,
        promptEvalCount: Int? = nil,
        promptEvalDuration: Int64? = nil,
        evalCount: Int? = nil,
        evalDuration: Int64? = nil
    ) {
        self.model = model
        self.createdAt = createdAt
        self.response = response
        self.done = done
        self.context = context
        self.totalDuration = totalDuration
        self.loadDuration = loadDuration
        self.promptEvalCount = promptEvalCount
        self.promptEvalDuration = promptEvalDuration
        self.evalCount = evalCount
        self.evalDuration = evalDuration
    }
}

// MARK: - Chat

/// Ollama chat message
public struct OllamaChatMessage: Codable, Sendable {
    public let role: String
    public let content: String
    public let images: [String]?

    public init(role: String, content: String, images: [String]? = nil) {
        self.role = role
        self.content = content
        self.images = images
    }
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

    public init(
        model: String,
        messages: [OllamaChatMessage],
        stream: Bool? = nil,
        format: String? = nil,
        numPredict: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Float? = nil,
        repeatPenalty: Float? = nil,
        seed: Int? = nil,
        stop: [String]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.format = format
        self.numPredict = numPredict
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.seed = seed
        self.stop = stop
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

    public init(
        model: String,
        createdAt: String,
        message: OllamaChatMessage,
        done: Bool,
        totalDuration: Int64? = nil,
        loadDuration: Int64? = nil,
        promptEvalCount: Int? = nil,
        promptEvalDuration: Int64? = nil,
        evalCount: Int? = nil,
        evalDuration: Int64? = nil
    ) {
        self.model = model
        self.createdAt = createdAt
        self.message = message
        self.done = done
        self.totalDuration = totalDuration
        self.loadDuration = loadDuration
        self.promptEvalCount = promptEvalCount
        self.promptEvalDuration = promptEvalDuration
        self.evalCount = evalCount
        self.evalDuration = evalDuration
    }
}

// MARK: - Embeddings

/// Ollama embeddings request
public struct OllamaEmbeddingsRequest: Codable, Sendable {
    public let model: String
    public let prompt: String

    public init(model: String, prompt: String) {
        self.model = model
        self.prompt = prompt
    }
}

/// Ollama embeddings response
public struct OllamaEmbeddingsResponse: Codable, Sendable {
    public let embedding: [Float]

    public init(embedding: [Float]) {
        self.embedding = embedding
    }
}

// MARK: - Model Management

/// Ollama pull request
public struct OllamaPullRequest: Codable, Sendable {
    public let name: String
    public let insecure: Bool?
    public let stream: Bool?

    public init(name: String, insecure: Bool? = nil, stream: Bool? = nil) {
        self.name = name
        self.insecure = insecure
        self.stream = stream
    }
}

/// Ollama pull response (streaming)
public struct OllamaPullResponse: Codable, Sendable {
    public let status: String
    public let digest: String?
    public let total: Int64?
    public let completed: Int64?

    public init(status: String, digest: String? = nil, total: Int64? = nil, completed: Int64? = nil) {
        self.status = status
        self.digest = digest
        self.total = total
        self.completed = completed
    }
}

/// Ollama create request
public struct OllamaCreateRequest: Codable, Sendable {
    public let name: String
    public let modelfile: String?
    public let path: String?
    public let stream: Bool?

    public init(name: String, modelfile: String? = nil, path: String? = nil, stream: Bool? = nil) {
        self.name = name
        self.modelfile = modelfile
        self.path = path
        self.stream = stream
    }
}

/// Ollama create response (streaming)
public struct OllamaCreateResponse: Codable, Sendable {
    public let status: String

    public init(status: String) {
        self.status = status
    }
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

    public init(
        format: String,
        family: String,
        families: [String],
        parameterSize: String,
        quantizationLevel: String
    ) {
        self.format = format
        self.family = family
        self.families = families
        self.parameterSize = parameterSize
        self.quantizationLevel = quantizationLevel
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

    public init(
        name: String,
        modifiedAt: String,
        size: Int64,
        digest: String,
        details: OllamaModelDetails? = nil
    ) {
        self.name = name
        self.modifiedAt = modifiedAt
        self.size = size
        self.digest = digest
        self.details = details
    }
}

/// Ollama tags response (model list)
public struct OllamaTagsResponse: Codable, Sendable {
    public let models: [OllamaModelInfo]

    public init(models: [OllamaModelInfo]) {
        self.models = models
    }
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

    public init(
        name: String,
        model: String,
        size: Int64,
        digest: String,
        details: OllamaModelDetails? = nil,
        expiresAt: String? = nil,
        sizeVRAM: Int64? = nil
    ) {
        self.name = name
        self.model = model
        self.size = size
        self.digest = digest
        self.details = details
        self.expiresAt = expiresAt
        self.sizeVRAM = sizeVRAM
    }
}

/// Ollama process response (running models)
public struct OllamaProcessResponse: Codable, Sendable {
    public let models: [OllamaRunningModel]

    public init(models: [OllamaRunningModel]) {
        self.models = models
    }
}

/// Ollama show request
public struct OllamaShowRequest: Codable, Sendable {
    public let name: String

    public init(name: String) {
        self.name = name
    }
}

/// Ollama show response
public struct OllamaShowResponse: Codable, Sendable {
    public let modelfile: String
    public let parameters: String
    public let template: String
    public let details: OllamaModelDetails?

    public init(
        modelfile: String,
        parameters: String,
        template: String,
        details: OllamaModelDetails? = nil
    ) {
        self.modelfile = modelfile
        self.parameters = parameters
        self.template = template
        self.details = details
    }
}

/// Ollama copy request
public struct OllamaCopyRequest: Codable, Sendable {
    public let source: String
    public let destination: String

    public init(source: String, destination: String) {
        self.source = source
        self.destination = destination
    }
}

/// Ollama delete request
public struct OllamaDeleteRequest: Codable, Sendable {
    public let name: String

    public init(name: String) {
        self.name = name
    }
}
