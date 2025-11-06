// Copyright © 2025 MLXR Development
// Example: List available models

import Foundation
import MLXR

@main
struct ModelListExample {
    static func main() async throws {
        let client = MLXRClient(config: .unixSocket())

        print("=== OpenAI-compatible models list ===\n")
        try await listOpenAIModels(client: client)

        print("\n=== Ollama-compatible models list ===\n")
        try await listOllamaModels(client: client)

        print("\n=== Running models ===\n")
        try await listRunningModels(client: client)
    }

    static func listOpenAIModels(client: MLXRClient) async throws {
        let response = try await client.listModels()

        print("Available models (\(response.data.count)):")
        for model in response.data {
            print("  - \(model.id) (owned by: \(model.ownedBy))")
        }
    }

    static func listOllamaModels(client: MLXRClient) async throws {
        let response = try await client.ollamaTags()

        print("Available models (\(response.models.count)):")
        for model in response.models {
            let sizeGB = Double(model.size) / (1024 * 1024 * 1024)
            print(String(format: "  - %@ (%.2f GB)", model.name, sizeGB))

            if let details = model.details {
                print("    Format: \(details.format)")
                print("    Family: \(details.family)")
                print("    Size: \(details.parameterSize)")
                print("    Quantization: \(details.quantizationLevel)")
            }
        }
    }

    static func listRunningModels(client: MLXRClient) async throws {
        let response = try await client.ollamaPS()

        print("Running models (\(response.models.count)):")
        for model in response.models {
            print("  - \(model.name)")

            if let sizeVRAM = model.sizeVRAM {
                let vramGB = Double(sizeVRAM) / (1024 * 1024 * 1024)
                print(String(format: "    VRAM: %.2f GB", vramGB))
            }

            if let expiresAt = model.expiresAt {
                print("    Expires: \(expiresAt)")
            }
        }
    }
}
