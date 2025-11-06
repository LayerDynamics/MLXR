// Copyright © 2025 MLXR Development
// Example: Generate embeddings

import Foundation
import MLXR

@main
struct EmbeddingsExample {
    static func main() async throws {
        let client = MLXRClient(config: .unixSocket())

        print("=== OpenAI-compatible embeddings ===\n")
        try await openAIEmbeddings(client: client)

        print("\n=== Ollama-compatible embeddings ===\n")
        try await ollamaEmbeddings(client: client)
    }

    static func openAIEmbeddings(client: MLXRClient) async throws {
        let request = EmbeddingRequest(
            model: "TinyLlama-1.1B",
            input: "Machine learning is transforming the world."
        )

        let response = try await client.embeddings(request: request)

        print("Model: \(response.model)")
        print("Embeddings count: \(response.data.count)")

        if let embedding = response.data.first {
            print("Embedding dimensions: \(embedding.embedding.count)")
            print("First 10 values: \(embedding.embedding.prefix(10))")
        }

        print("Tokens used: \(response.usage.totalTokens)")
    }

    static func ollamaEmbeddings(client: MLXRClient) async throws {
        let request = OllamaEmbeddingsRequest(
            model: "TinyLlama-1.1B",
            prompt: "Machine learning is transforming the world."
        )

        let response = try await client.ollamaEmbeddings(request: request)

        print("Embedding dimensions: \(response.embedding.count)")
        print("First 10 values: \(response.embedding.prefix(10))")
    }
}
