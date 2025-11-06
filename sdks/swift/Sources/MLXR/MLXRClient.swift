// Copyright © 2025 MLXR Development
// Main MLXR client with OpenAI and Ollama-compatible APIs

import Foundation

/// Configuration for the MLXR client
public struct MLXRClientConfig {
    /// Transport type
    public enum TransportType {
        /// Unix domain socket (default for macOS)
        case unixSocket(path: String)
        /// HTTP connection
        case http(baseURL: URL)
    }

    /// Transport type to use
    public let transport: TransportType

    /// Request timeout in seconds (default: 120)
    public let timeout: TimeInterval

    /// API key for authentication (optional)
    public let apiKey: String?

    /// Create config for Unix domain socket
    public static func unixSocket(
        path: String = "~/Library/Application Support/MLXRunner/run/mlxrunner.sock",
        timeout: TimeInterval = 120,
        apiKey: String? = nil
    ) -> MLXRClientConfig {
        let expandedPath = NSString(string: path).expandingTildeInPath
        return MLXRClientConfig(transport: .unixSocket(path: expandedPath), timeout: timeout, apiKey: apiKey)
    }

    /// Create config for HTTP connection
    public static func http(
        baseURL: URL = URL(string: "http://127.0.0.1:11434")!,
        timeout: TimeInterval = 120,
        apiKey: String? = nil
    ) -> MLXRClientConfig {
        return MLXRClientConfig(transport: .http(baseURL: baseURL), timeout: timeout, apiKey: apiKey)
    }

    public init(transport: TransportType, timeout: TimeInterval = 120, apiKey: String? = nil) {
        self.transport = transport
        self.timeout = timeout
        self.apiKey = apiKey
    }
}

/// MLXR client for interacting with the inference daemon
public actor MLXRClient {
    private let config: MLXRClientConfig
    private let transport: Transport
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    /// Initialize a new MLXR client
    public init(config: MLXRClientConfig = .unixSocket()) {
        self.config = config

        switch config.transport {
        case .http(let baseURL):
            self.transport = HTTPTransport(baseURL: baseURL)
        case .unixSocket(let path):
            #if canImport(Darwin)
            self.transport = UnixSocketTransport(socketPath: path)
            #else
            fatalError("Unix domain socket transport is only available on macOS")
            #endif
        }

        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()

        // Configure JSON encoder/decoder
        encoder.keyEncodingStrategy = .convertToSnakeCase
        decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    // MARK: - OpenAI-Compatible API

    /// List available models
    public func listModels() async throws -> ModelListResponse {
        let (data, _) = try await request(
            method: "GET",
            path: "/v1/models",
            body: nil
        )

        return try decoder.decode(ModelListResponse.self, from: data)
    }

    /// Get information about a specific model
    public func getModel(id: String) async throws -> ModelInfo {
        let (data, _) = try await request(
            method: "GET",
            path: "/v1/models/\(id)",
            body: nil
        )

        return try decoder.decode(ModelInfo.self, from: data)
    }

    /// Create a chat completion
    public func chatCompletion(request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/v1/chat/completions",
            body: requestData
        )

        return try decoder.decode(ChatCompletionResponse.self, from: data)
    }

    /// Create a streaming chat completion
    public func chatCompletionStream(
        request: ChatCompletionRequest
    ) async throws -> AsyncThrowingStream<ChatCompletionChunk, Error> {
        let streamRequest = ChatCompletionRequest(
            model: request.model,
            messages: request.messages,
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repetitionPenalty: request.repetitionPenalty,
            maxTokens: request.maxTokens,
            stream: true,
            stop: request.stop,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            n: request.n,
            user: request.user,
            tools: request.tools,
            toolChoice: request.toolChoice,
            seed: request.seed
        )

        return try await streamSSE(
            request: streamRequest,
            method: "POST",
            path: "/v1/chat/completions"
        )
    }

    /// Create a text completion
    public func completion(request: CompletionRequest) async throws -> CompletionResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/v1/completions",
            body: requestData
        )

        return try decoder.decode(CompletionResponse.self, from: data)
    }

    /// Create a streaming text completion
    public func completionStream(
        request: CompletionRequest
    ) async throws -> AsyncThrowingStream<CompletionResponse, Error> {
        let streamRequest = CompletionRequest(
            model: request.model,
            prompt: request.prompt,
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repetitionPenalty: request.repetitionPenalty,
            maxTokens: request.maxTokens,
            stream: true,
            stop: request.stop,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            n: request.n,
            suffix: request.suffix,
            seed: request.seed
        )

        return try await streamSSE(
            request: streamRequest,
            method: "POST",
            path: "/v1/completions"
        )
    }

    /// Create embeddings
    public func embeddings(request: EmbeddingRequest) async throws -> EmbeddingResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/v1/embeddings",
            body: requestData
        )

        return try decoder.decode(EmbeddingResponse.self, from: data)
    }

    // MARK: - Ollama-Compatible API

    /// Generate text using Ollama API
    public func ollamaGenerate(request: OllamaGenerateRequest) async throws -> OllamaGenerateResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/api/generate",
            body: requestData
        )

        return try decoder.decode(OllamaGenerateResponse.self, from: data)
    }

    /// Stream text generation using Ollama API
    public func ollamaGenerateStream(
        request: OllamaGenerateRequest
    ) async throws -> AsyncThrowingStream<OllamaGenerateResponse, Error> {
        let streamRequest = OllamaGenerateRequest(
            model: request.model,
            prompt: request.prompt,
            system: request.system,
            template: request.template,
            context: request.context,
            stream: true,
            raw: request.raw,
            format: request.format,
            numPredict: request.numPredict,
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repeatPenalty: request.repeatPenalty,
            seed: request.seed,
            stop: request.stop
        )

        return try await streamNDJSON(
            request: streamRequest,
            method: "POST",
            path: "/api/generate"
        )
    }

    /// Chat using Ollama API
    public func ollamaChat(request: OllamaChatRequest) async throws -> OllamaChatResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/api/chat",
            body: requestData
        )

        return try decoder.decode(OllamaChatResponse.self, from: data)
    }

    /// Stream chat using Ollama API
    public func ollamaChatStream(
        request: OllamaChatRequest
    ) async throws -> AsyncThrowingStream<OllamaChatResponse, Error> {
        let streamRequest = OllamaChatRequest(
            model: request.model,
            messages: request.messages,
            stream: true,
            format: request.format,
            numPredict: request.numPredict,
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            repeatPenalty: request.repeatPenalty,
            seed: request.seed,
            stop: request.stop
        )

        return try await streamNDJSON(
            request: streamRequest,
            method: "POST",
            path: "/api/chat"
        )
    }

    /// Create embeddings using Ollama API
    public func ollamaEmbeddings(request: OllamaEmbeddingsRequest) async throws -> OllamaEmbeddingsResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/api/embeddings",
            body: requestData
        )

        return try decoder.decode(OllamaEmbeddingsResponse.self, from: data)
    }

    /// List models using Ollama API
    public func ollamaTags() async throws -> OllamaTagsResponse {
        let (data, _) = try await request(
            method: "GET",
            path: "/api/tags",
            body: nil
        )

        return try decoder.decode(OllamaTagsResponse.self, from: data)
    }

    /// List running models using Ollama API
    public func ollamaPS() async throws -> OllamaProcessResponse {
        let (data, _) = try await request(
            method: "GET",
            path: "/api/ps",
            body: nil
        )

        return try decoder.decode(OllamaProcessResponse.self, from: data)
    }

    /// Show model information using Ollama API
    public func ollamaShow(request: OllamaShowRequest) async throws -> OllamaShowResponse {
        let requestData = try encoder.encode(request)

        let (data, _) = try await self.request(
            method: "POST",
            path: "/api/show",
            body: requestData
        )

        return try decoder.decode(OllamaShowResponse.self, from: data)
    }

    /// Copy a model using Ollama API
    public func ollamaCopy(request: OllamaCopyRequest) async throws {
        let requestData = try encoder.encode(request)

        _ = try await self.request(
            method: "POST",
            path: "/api/copy",
            body: requestData
        )
    }

    /// Delete a model using Ollama API
    public func ollamaDelete(request: OllamaDeleteRequest) async throws {
        let requestData = try encoder.encode(request)

        _ = try await self.request(
            method: "DELETE",
            path: "/api/delete",
            body: requestData
        )
    }

    /// Pull a model using Ollama API (streaming)
    public func ollamaPull(
        request: OllamaPullRequest
    ) async throws -> AsyncThrowingStream<OllamaPullResponse, Error> {
        let streamRequest = OllamaPullRequest(
            name: request.name,
            insecure: request.insecure,
            stream: true
        )

        return try await streamNDJSON(
            request: streamRequest,
            method: "POST",
            path: "/api/pull"
        )
    }

    /// Create a model using Ollama API (streaming)
    public func ollamaCreate(
        request: OllamaCreateRequest
    ) async throws -> AsyncThrowingStream<OllamaCreateResponse, Error> {
        let streamRequest = OllamaCreateRequest(
            name: request.name,
            modelfile: request.modelfile,
            path: request.path,
            stream: true
        )

        return try await streamNDJSON(
            request: streamRequest,
            method: "POST",
            path: "/api/create"
        )
    }

    // MARK: - Private helpers

    private func request(
        method: String,
        path: String,
        body: Data?
    ) async throws -> (Data, HTTPURLResponse) {
        var headers: [String: String] = [
            "Content-Type": "application/json",
            "Accept": "application/json"
        ]

        if let apiKey = config.apiKey {
            headers["Authorization"] = "Bearer \(apiKey)"
        }

        do {
            return try await transport.request(
                method: method,
                path: path,
                headers: headers,
                body: body,
                timeout: config.timeout
            )
        } catch let error as MLXRError {
            throw error
        } catch {
            throw MLXRError.networkError(error)
        }
    }

    private func streamRequest(
        method: String,
        path: String,
        body: Data?
    ) async throws -> AsyncThrowingStream<Data, Error> {
        var headers: [String: String] = [
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        ]

        if let apiKey = config.apiKey {
            headers["Authorization"] = "Bearer \(apiKey)"
        }

        do {
            return try await transport.streamRequest(
                method: method,
                path: path,
                headers: headers,
                body: body,
                timeout: config.timeout
            )
        } catch let error as MLXRError {
            throw error
        } catch {
            throw MLXRError.networkError(error)
        }
    }

    /// Helper to stream with SSE parsing (OpenAI format)
    private func streamSSE<Req: Encodable, Res: Decodable>(
        request: Req,
        method: String,
        path: String
    ) async throws -> AsyncThrowingStream<Res, Error> {
        let requestData = try encoder.encode(request)
        let dataStream = try await streamRequest(method: method, path: path, body: requestData)
        return SSEStream<Res>(dataStream: dataStream, decoder: decoder).events()
    }

    /// Helper to stream with NDJSON parsing (Ollama format)
    private func streamNDJSON<Req: Encodable, Res: Decodable>(
        request: Req,
        method: String,
        path: String
    ) async throws -> AsyncThrowingStream<Res, Error> {
        let requestData = try encoder.encode(request)
        let dataStream = try await streamRequest(method: method, path: path, body: requestData)
        return parseNDJSON(dataStream)
    }

    /// Parse newline-delimited JSON (used by Ollama)
    private func parseNDJSON<T: Decodable>(_ dataStream: AsyncThrowingStream<Data, Error>) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var buffer = Data()

                do {
                    for try await chunk in dataStream {
                        buffer.append(chunk)

                        // Split by newlines
                        while let newlineRange = buffer.range(of: Data([0x0A])) {
                            let line = buffer[..<newlineRange.lowerBound]
                            buffer = buffer[newlineRange.upperBound...]

                            if line.isEmpty {
                                continue
                            }

                            do {
                                let decoded = try decoder.decode(T.self, from: line)
                                continuation.yield(decoded)
                            } catch {
                                // Log skipped invalid JSON line for traceability
                                #if DEBUG
                                let lineString = String(data: line, encoding: .utf8) ?? "<unreadable>"
                                print("MLXRClient.parseNDJSON: Skipping invalid JSON line: \(lineString), error: \(error)")
                                #endif
                                continue
                            }
                        }
                    }

                    // Process any remaining data
                    if !buffer.isEmpty {
                        do {
                            let decoded = try decoder.decode(T.self, from: buffer)
                            continuation.yield(decoded)
                        } catch {
                            // Log skipped invalid JSON for traceability
                            #if DEBUG
                            let bufferString = String(data: buffer, encoding: .utf8) ?? "<unreadable>"
                            print("MLXRClient.parseNDJSON: Skipping invalid JSON at end: \(bufferString), error: \(error)")
                            #endif
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: MLXRError.streamingError(error.localizedDescription))
                }
            }
        }
    }
}
