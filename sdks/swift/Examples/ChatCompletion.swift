// Copyright © 2025 MLXR Development
// Example: Chat completion with streaming

import Foundation
import MLXR

@main
struct ChatCompletionExample {
    static func main() async throws {
        // Create client (defaults to Unix socket)
        let client = MLXRClient(config: .unixSocket())

        // Or use HTTP:
        // let client = MLXRClient(config: .http())

        print("=== Non-streaming chat completion ===\n")
        try await nonStreamingExample(client: client)

        print("\n=== Streaming chat completion ===\n")
        try await streamingExample(client: client)
    }

    static func nonStreamingExample(client: MLXRClient) async throws {
        let request = ChatCompletionRequest(
            model: "TinyLlama-1.1B",
            messages: [
                ChatMessage(role: "system", content: "You are a helpful assistant."),
                ChatMessage(role: "user", content: "Write a haiku about machine learning.")
            ],
            temperature: 0.7,
            maxTokens: 100
        )

        let response = try await client.chatCompletion(request: request)

        print("Model: \(response.model)")
        print("Response: \(response.choices.first?.message.content ?? "No response")")
        print("Usage: \(response.usage.totalTokens) tokens")
    }

    static func streamingExample(client: MLXRClient) async throws {
        let request = ChatCompletionRequest(
            model: "TinyLlama-1.1B",
            messages: [
                ChatMessage(role: "system", content: "You are a helpful assistant."),
                ChatMessage(role: "user", content: "Count from 1 to 10.")
            ],
            temperature: 0.7,
            maxTokens: 100,
            stream: true
        )

        print("Streaming response: ", terminator: "")

        let stream = try await client.chatCompletionStream(request: request)

        for try await chunk in stream {
            if let content = chunk.choices.first?.delta.content {
                print(content, terminator: "")
                fflush(stdout)
            }
        }

        print("\n")
    }
}
