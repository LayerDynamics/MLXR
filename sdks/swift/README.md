# MLXR Swift SDK

Official Swift SDK for MLXR - High-performance LLM inference engine for Apple Silicon.

## Features

- 🚀 **Native Swift async/await** APIs for modern concurrency
- 🔌 **Dual transport support**: Unix Domain Socket (default) and HTTP
- 🌊 **Streaming responses** with SSE (Server-Sent Events) support
- 🔄 **OpenAI-compatible** and **Ollama-compatible** APIs
- 📦 **Type-safe models** with full Codable support
- 🎯 **Zero dependencies** - uses only Swift standard library
- 🍎 **macOS native** with optimized Unix socket communication

## Requirements

- macOS 13.0+ or iOS 16.0+
- Swift 5.9+
- Xcode 15.0+
- MLXR daemon running (see main MLXR documentation)

## Installation

### Swift Package Manager

Add the following to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/LayerDynamics/MLXR.git", from: "0.1.0")
]
```

Or in Xcode:
1. File → Add Packages...
2. Enter the repository URL: `https://github.com/LayerDynamics/MLXR.git`
3. Select version and add to your target

### Local Development

```swift
.package(path: "../path/to/MLXR/sdks/swift")
```

## Quick Start

### Basic Chat Completion

```swift
import MLXR

// Create client (uses Unix socket by default)
let client = MLXRClient(config: .unixSocket())

// Create a chat completion request
let request = ChatCompletionRequest(
    model: "TinyLlama-1.1B",
    messages: [
        ChatMessage(role: "system", content: "You are a helpful assistant."),
        ChatMessage(role: "user", content: "What is machine learning?")
    ],
    temperature: 0.7,
    maxTokens: 150
)

// Get response
let response = try await client.chatCompletion(request: request)
print(response.choices.first?.message.content ?? "")
```

### Streaming Chat Completion

```swift
import MLXR

let client = MLXRClient(config: .unixSocket())

let request = ChatCompletionRequest(
    model: "TinyLlama-1.1B",
    messages: [
        ChatMessage(role: "user", content: "Write a story about a robot.")
    ],
    temperature: 0.8,
    maxTokens: 200,
    stream: true
)

// Stream tokens as they're generated
let stream = try await client.chatCompletionStream(request: request)

for try await chunk in stream {
    if let content = chunk.choices.first?.delta.content {
        print(content, terminator: "")
        fflush(stdout)
    }
}
```

## Configuration

### Unix Domain Socket (Default)

Best performance for local macOS applications:

```swift
let config = MLXRClientConfig.unixSocket(
    path: "~/Library/Application Support/MLXRunner/run/mlxrunner.sock",
    timeout: 120,
    apiKey: nil  // Optional API key
)

let client = MLXRClient(config: config)
```

### HTTP Connection

For remote connections or when UDS is not available:

```swift
let config = MLXRClientConfig.http(
    baseURL: URL(string: "http://127.0.0.1:11434")!,
    timeout: 120,
    apiKey: "your-api-key"  // Optional
)

let client = MLXRClient(config: config)
```

## API Reference

### OpenAI-Compatible API

#### Chat Completions

```swift
// Non-streaming
let response = try await client.chatCompletion(request: ChatCompletionRequest(...))

// Streaming
let stream = try await client.chatCompletionStream(request: ChatCompletionRequest(...))
for try await chunk in stream {
    // Process chunk
}
```

#### Text Completions

```swift
// Non-streaming
let response = try await client.completion(request: CompletionRequest(...))

// Streaming
let stream = try await client.completionStream(request: CompletionRequest(...))
```

#### Embeddings

```swift
let response = try await client.embeddings(request: EmbeddingRequest(
    model: "TinyLlama-1.1B",
    input: "Your text here"
))

print(response.data.first?.embedding ?? [])
```

#### Models

```swift
// List all models
let models = try await client.listModels()

// Get specific model info
let model = try await client.getModel(id: "TinyLlama-1.1B")
```

### Ollama-Compatible API

#### Generate

```swift
// Non-streaming
let response = try await client.ollamaGenerate(request: OllamaGenerateRequest(
    model: "TinyLlama-1.1B",
    prompt: "Explain quantum computing"
))

// Streaming
let stream = try await client.ollamaGenerateStream(request: OllamaGenerateRequest(...))
for try await chunk in stream {
    print(chunk.response, terminator: "")
}
```

#### Chat

```swift
let response = try await client.ollamaChat(request: OllamaChatRequest(
    model: "TinyLlama-1.1B",
    messages: [
        OllamaChatMessage(role: "user", content: "Hello!")
    ]
))
```

#### Model Management

```swift
// List models
let tags = try await client.ollamaTags()

// Show model details
let details = try await client.ollamaShow(request: OllamaShowRequest(name: "TinyLlama-1.1B"))

// List running models
let running = try await client.ollamaPS()

// Pull a model (streaming)
let pullStream = try await client.ollamaPull(request: OllamaPullRequest(name: "llama2"))
for try await progress in pullStream {
    print("Status: \(progress.status)")
}

// Copy a model
try await client.ollamaCopy(request: OllamaCopyRequest(
    source: "model1",
    destination: "model2"
))

// Delete a model
try await client.ollamaDelete(request: OllamaDeleteRequest(name: "old-model"))
```

## Error Handling

The SDK provides comprehensive error handling with the `MLXRError` enum:

```swift
do {
    let response = try await client.chatCompletion(request: request)
} catch let error as MLXRError {
    switch error {
    case .networkError(let underlying):
        print("Network error: \(underlying)")
    case .httpError(let statusCode, let message):
        print("HTTP \(statusCode): \(message)")
    case .serverError(let message, let type, let code):
        print("Server error: \(message)")
    case .modelNotFound(let model):
        print("Model not found: \(model)")
    case .timeout:
        print("Request timeout")
    case .streamingError(let message):
        print("Streaming error: \(message)")
    default:
        print("Error: \(error.localizedDescription)")
    }
}
```

## Advanced Usage

### Custom Request Parameters

```swift
let request = ChatCompletionRequest(
    model: "TinyLlama-1.1B",
    messages: [...],
    temperature: 0.9,           // Randomness (0-2)
    topP: 0.95,                 // Nucleus sampling
    topK: 50,                   // Top-k sampling
    repetitionPenalty: 1.1,     // Penalize repetition
    maxTokens: 500,             // Max output length
    stop: ["END", "\n\n"],      // Stop sequences
    seed: 42                    // Reproducible generation
)
```

### Function Calling

```swift
let request = ChatCompletionRequest(
    model: "TinyLlama-1.1B",
    messages: [...],
    tools: [
        ToolDefinition(
            type: "function",
            function: FunctionDefinition(
                name: "get_weather",
                description: "Get the weather for a location",
                parametersJSON: """
                {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
                """
            )
        )
    ],
    toolChoice: "auto"
)
```

### Vision (Multi-modal)

```swift
let request = OllamaChatRequest(
    model: "llava",
    messages: [
        OllamaChatMessage(
            role: "user",
            content: "What's in this image?",
            images: [base64EncodedImage]  // Base64 encoded image data
        )
    ]
)
```

## Examples

See the [Examples](Examples/) directory for complete, runnable examples:

- **[ChatCompletion.swift](Examples/ChatCompletion.swift)**: Chat completions with streaming
- **[OllamaGenerate.swift](Examples/OllamaGenerate.swift)**: Ollama text generation
- **[ModelList.swift](Examples/ModelList.swift)**: List and inspect models
- **[Embeddings.swift](Examples/Embeddings.swift)**: Generate embeddings

### Running Examples

```bash
cd Examples
swift run ChatCompletion
swift run OllamaGenerate
swift run ModelList
```

## Testing

Run the test suite:

```bash
swift test
```

Run with verbose output:

```bash
swift test --verbose
```

## Performance Tips

1. **Use Unix sockets**: ~20% faster than HTTP for local connections
2. **Enable streaming**: Start displaying results immediately
3. **Batch requests**: Use the same client instance for multiple requests
4. **Adjust timeout**: Increase for large models or long generations
5. **Use appropriate temperature**: Lower (0.1-0.3) for factual, higher (0.7-1.0) for creative

## Troubleshooting

### "Socket not available" error

Make sure the MLXR daemon is running:

```bash
# Check if daemon is running
ps aux | grep mlxrunnerd

# Check socket exists
ls -la ~/Library/Application\ Support/MLXRunner/run/mlxrunner.sock
```

### "Model not found" error

List available models:

```swift
let models = try await client.listModels()
print(models.data.map { $0.id })
```

### Timeout errors

Increase timeout for large models:

```swift
let config = MLXRClientConfig.unixSocket(timeout: 300)  // 5 minutes
```

### Streaming stops early

Check for stop sequences in your request and model configuration.

## Architecture

The SDK uses a layered architecture:

```
┌─────────────────────────────────────┐
│         MLXRClient                  │  ← High-level API
├─────────────────────────────────────┤
│  OpenAI Models  │  Ollama Models    │  ← Request/Response types
├─────────────────────────────────────┤
│        SSEStream                    │  ← Streaming support
├─────────────────────────────────────┤
│  UnixSocketTransport │ HTTPTransport│  ← Transport layer
└─────────────────────────────────────┘
```

### Key Components

- **MLXRClient**: Main client class (actor for thread safety)
- **Transport**: Protocol for network communication
  - **UnixSocketTransport**: Native macOS socket communication
  - **HTTPTransport**: Standard HTTP/HTTPS using URLSession
- **SSEStream**: Server-Sent Events parser for streaming
- **Models**: Type-safe Codable models matching server API

## Contributing

See the main [MLXR repository](https://github.com/LayerDynamics/MLXR) for contribution guidelines.

## License

Copyright © 2025 MLXR Development. See LICENSE for details.

## Links

- [MLXR Project](https://github.com/LayerDynamics/MLXR)
- [Documentation](https://docs.mlxr.dev)
- [Issue Tracker](https://github.com/LayerDynamics/MLXR/issues)
- [Discord Community](https://discord.gg/mlxr)

## Version History

### 0.1.0 (Initial Release)

- OpenAI-compatible API support
- Ollama-compatible API support
- Unix socket and HTTP transports
- SSE streaming
- Comprehensive error handling
- Full test coverage
- Complete documentation and examples
