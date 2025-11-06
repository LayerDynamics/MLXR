# MLXR Swift SDK Examples

This directory contains example programs demonstrating the MLXR Swift SDK.

## Prerequisites

1. Install and start the MLXR daemon:
   ```bash
   cd ../../..  # Go to MLXR root
   make install-dev
   ./build/cmake/bin/mlxrunnerd
   ```

2. Ensure you have a model loaded (e.g., TinyLlama-1.1B)

## Running Examples

Each example is a standalone Swift program that you can run directly:

```bash
# From the Examples directory
swift run ChatCompletion
swift run OllamaGenerate
swift run ModelList
swift run Embeddings
```

Or build and run individually:

```bash
swiftc -o chat_example ChatCompletion.swift -I ../Sources
./chat_example
```

## Examples

### ChatCompletion.swift

Demonstrates both streaming and non-streaming chat completions using the OpenAI-compatible API.

**Features:**
- System and user messages
- Temperature and token limit configuration
- Real-time streaming output

### OllamaGenerate.swift

Shows text generation using the Ollama-compatible API with streaming support.

**Features:**
- Simple prompt-based generation
- Performance metrics (tokens/sec)
- Streaming output

### ModelList.swift

Lists available and running models using both API formats.

**Features:**
- OpenAI models list
- Ollama models with detailed metadata
- Running models with VRAM usage

### Embeddings.swift

Generates text embeddings using both OpenAI and Ollama APIs.

**Features:**
- Vector embedding generation
- Dimension inspection
- Token usage tracking

## Customization

All examples can be easily modified to:
- Use different models
- Adjust generation parameters (temperature, top-p, etc.)
- Change prompts and messages
- Switch between HTTP and Unix socket transport

Example configuration change:

```swift
// Use HTTP instead of Unix socket
let client = MLXRClient(config: .http(
    baseURL: URL(string: "http://localhost:11434")!
))
```

## Error Handling

All examples include basic error handling. For production use, implement more robust error handling:

```swift
do {
    let response = try await client.chatCompletion(request: request)
    // Handle response
} catch let error as MLXRError {
    switch error {
    case .modelNotFound(let model):
        print("Model '\(model)' not found. Please load it first.")
    case .timeout:
        print("Request timed out. Try increasing the timeout.")
    case .serverError(let message, _, _):
        print("Server error: \(message)")
    default:
        print("Error: \(error.localizedDescription)")
    }
} catch {
    print("Unexpected error: \(error)")
}
```

## Next Steps

- Explore the [main README](../README.md) for complete API documentation
- Check the [test suite](../Tests/MLXRTests/) for more usage examples
- Read the [MLXR documentation](../../../docs/) for daemon configuration
