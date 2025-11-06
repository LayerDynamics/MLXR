// Copyright © 2025 MLXR Development
// Example: Ollama-compatible text generation

import Foundation
import MLXR

@main
struct OllamaGenerateExample {
    static func main() async throws {
        // Create client
        let client = MLXRClient(config: .unixSocket())

        print("=== Ollama text generation ===\n")
        try await generateExample(client: client)

        print("\n=== Ollama streaming generation ===\n")
        try await streamingExample(client: client)
    }

    static func generateExample(client: MLXRClient) async throws {
        let request = OllamaGenerateRequest(
            model: "TinyLlama-1.1B",
            prompt: "Explain quantum computing in simple terms.",
            temperature: 0.8,
            numPredict: 150
        )

        let response = try await client.ollamaGenerate(request: request)

        print("Response: \(response.response)")
        print("Done: \(response.done)")

        if let evalCount = response.evalCount, let evalDuration = response.evalDuration {
            let tokensPerSec = Double(evalCount) / (Double(evalDuration) / 1_000_000_000)
            print(String(format: "Speed: %.2f tokens/sec", tokensPerSec))
        }
    }

    static func streamingExample(client: MLXRClient) async throws {
        let request = OllamaGenerateRequest(
            model: "TinyLlama-1.1B",
            prompt: "Write a short story about a robot.",
            stream: true,
            temperature: 0.9,
            numPredict: 200
        )

        print("Streaming response: ", terminator: "")

        let stream = try await client.ollamaGenerateStream(request: request)

        for try await chunk in stream {
            print(chunk.response, terminator: "")
            fflush(stdout)

            if chunk.done {
                print("\n")
                if let evalCount = chunk.evalCount, let evalDuration = chunk.evalDuration {
                    let tokensPerSec = Double(evalCount) / (Double(evalDuration) / 1_000_000_000)
                    print(String(format: "\nSpeed: %.2f tokens/sec", tokensPerSec))
                }
            }
        }
    }
}
