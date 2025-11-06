# MLXR TypeScript SDK

TypeScript SDK for [MLXR](https://github.com/LayerDynamics/MLXR) - High-performance LLM inference engine for Apple Silicon.

## Features

- ✅ **OpenAI-compatible API** - Drop-in replacement for OpenAI client
- ✅ **Ollama-compatible API** - Full Ollama API support
- ✅ **Unix Domain Socket** - Native macOS IPC for best performance
- ✅ **HTTP/HTTPS Support** - Standard HTTP and HTTPS connections
- ✅ **SSE Streaming** - Real-time token streaming
- ✅ **Automatic Retry** - Configurable exponential backoff for transient errors
- ✅ **TypeScript Native** - Full type safety and IDE autocomplete
- ✅ **Zero Dependencies** - Uses only Node.js built-in modules

## Installation

```bash
npm install @mlxr/typescript-sdk
```

Or with yarn:

```bash
yarn add @mlxr/typescript-sdk
```

## Quick Start

```typescript
import { MLXRClient } from '@mlxr/typescript-sdk';

// Create client (auto-detects Unix socket on macOS)
const client = new MLXRClient();

// OpenAI-style chat
const response = await client.openai.createChatCompletion({
  model: 'tinyllama',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is machine learning?' },
  ],
  temperature: 0.7,
  max_tokens: 100,
});

console.log(response.choices[0].message.content);
```

## Configuration

```typescript
const client = new MLXRClient({
  // Optional: HTTP endpoint (default: http://localhost:11434)
  baseUrl: 'http://localhost:11434',

  // Optional: Unix socket path (default: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock)
  unixSocketPath: '/path/to/mlxrunner.sock',

  // Optional: API key for authentication
  apiKey: 'your-api-key',

  // Optional: Request timeout in milliseconds (default: 30000)
  timeout: 60000,

  // Optional: Prefer Unix socket over HTTP (default: true on macOS)
  preferUnixSocket: true,

  // Optional: Custom headers
  headers: {
    'X-Custom-Header': 'value',
  },

  // Optional: Retry configuration
  retry: {
    maxRetries: 3,                    // Maximum retry attempts (default: 3)
    initialDelay: 1000,                // Initial delay in ms (default: 1000)
    backoffMultiplier: 2,              // Exponential backoff multiplier (default: 2)
    maxDelay: 10000,                   // Maximum delay in ms (default: 10000)
    retryableStatusCodes: [408, 429, 500, 502, 503, 504], // HTTP status codes to retry
  },
});
```

## OpenAI API

### Chat Completions

```typescript
// Non-streaming
const response = await client.openai.createChatCompletion({
  model: 'tinyllama',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' },
  ],
  temperature: 0.7,
  max_tokens: 100,
});

// Streaming
const stream = client.openai.streamChatCompletion({
  model: 'tinyllama',
  messages: [{ role: 'user', content: 'Tell me a story.' }],
  temperature: 0.8,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) {
    process.stdout.write(content);
  }
}
```

### Text Completions

```typescript
// Non-streaming
const response = await client.openai.createCompletion({
  model: 'tinyllama',
  prompt: 'Once upon a time',
  max_tokens: 50,
});

// Streaming
const stream = client.openai.streamCompletion({
  model: 'tinyllama',
  prompt: 'The meaning of life is',
  max_tokens: 100,
});

for await (const chunk of stream) {
  const text = chunk.choices[0]?.text;
  if (text) {
    process.stdout.write(text);
  }
}
```

### Embeddings

```typescript
const response = await client.openai.createEmbedding({
  model: 'tinyllama',
  input: 'Machine learning is transforming the world.',
});

const { embedding } = response.data[0];
console.log('Embedding dimensions:', embedding.length);
```

### Model Management

```typescript
// List all models
const models = await client.openai.listModels();
console.log(models.data);

// Get specific model info
const model = await client.openai.getModel('tinyllama');
console.log(model);
```

## Ollama API

### Chat

```typescript
// Non-streaming
const response = await client.ollama.chat({
  model: 'tinyllama',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain quantum computing.' },
  ],
  temperature: 0.7,
});

console.log(response.message.content);

// Streaming
const stream = client.ollama.streamChat({
  model: 'tinyllama',
  messages: [{ role: 'user', content: 'Tell me a joke.' }],
});

for await (const chunk of stream) {
  const content = chunk.message?.content;
  if (content) {
    process.stdout.write(content);
  }
}
```

### Generate

```typescript
// Non-streaming
const response = await client.ollama.generate({
  model: 'tinyllama',
  prompt: 'Why is the sky blue?',
  num_predict: 100,
});

console.log(response.response);

// Streaming
const stream = client.ollama.streamGenerate({
  model: 'tinyllama',
  prompt: 'Write a poem about coding.',
  temperature: 0.9,
});

for await (const chunk of stream) {
  if (chunk.response) {
    process.stdout.write(chunk.response);
  }
}
```

### Embeddings

```typescript
const response = await client.ollama.embeddings({
  model: 'tinyllama',
  prompt: 'Artificial intelligence is the future.',
});

console.log('Embedding dimensions:', response.embedding.length);
```

### Model Management

```typescript
// List all models
const tags = await client.ollama.tags();
tags.models.forEach((model) => {
  console.log(`${model.name} - ${(model.size / 1024 / 1024).toFixed(2)} MB`);
});

// Show model details
const details = await client.ollama.show({ name: 'tinyllama' });
console.log(details);

// Pull a model
const stream = client.ollama.streamPull({
  name: 'tinyllama',
  stream: true,
});

for await (const progress of stream) {
  if (progress.total && progress.completed) {
    const percent = ((progress.completed / progress.total) * 100).toFixed(1);
    console.log(`Progress: ${percent}%`);
  }
}

// Copy a model
await client.ollama.copy({
  source: 'tinyllama',
  destination: 'my-tinyllama',
});

// Delete a model
await client.ollama.delete({ name: 'my-tinyllama' });

// List running models
const running = await client.ollama.ps();
console.log(running.models);
```

## Advanced Usage

### Error Handling

```typescript
try {
  const response = await client.openai.createChatCompletion({
    model: 'tinyllama',
    messages: [{ role: 'user', content: 'Hello!' }],
  });
  console.log(response.choices[0].message.content);
} catch (error) {
  if (error instanceof Error) {
    console.error('Error:', error.message);
  }
}
```

### Custom Configuration

```typescript
// Update configuration after initialization
client.updateConfig({
  timeout: 60000,
  apiKey: 'new-api-key',
});

// Get current configuration
const config = client.getConfig();
console.log(config);

// Check Unix socket availability
if (client.isUnixSocketAvailable()) {
  console.log('Using Unix domain socket for best performance');
} else {
  console.log('Using HTTP connection');
}
```

### Health Check

```typescript
try {
  const health = await client.health();
  console.log('MLXR daemon is running:', health.status);
} catch (error) {
  console.error('MLXR daemon is not available');
}
```

## Examples

See the [`examples/`](./examples) directory for complete working examples:

- [`openai-chat.ts`](./examples/openai-chat.ts) - OpenAI chat completions
- [`ollama-chat.ts`](./examples/ollama-chat.ts) - Ollama chat
- [`completions.ts`](./examples/completions.ts) - Text completions
- [`embeddings.ts`](./examples/embeddings.ts) - Generate and compare embeddings
- [`model-management.ts`](./examples/model-management.ts) - Model management operations

Run examples:

```bash
npm install
npm run build
node dist/examples/openai-chat.js
```

## API Reference

Full API documentation is available in the [TypeScript types](./src/types).

### Main Classes

- `MLXRClient` - Unified client with both OpenAI and Ollama APIs
- `OpenAIClient` - OpenAI-compatible API client
- `OllamaClient` - Ollama-compatible API client

### Types

All request and response types are exported from the main package:

```typescript
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  OllamaChatRequest,
  OllamaChatResponse,
  // ... and many more
} from '@mlxr/typescript-sdk';
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run watch

# Format code
npm run format

# Lint
npm run lint

# Run tests
npm test
```

## Requirements

- Node.js >= 18.0.0
- MLXR daemon running (either via Unix socket or HTTP)

## License

MIT

## Contributing

Contributions are welcome! Please see the main [MLXR repository](https://github.com/LayerDynamics/MLXR) for contribution guidelines.

## Links

- [MLXR Repository](https://github.com/LayerDynamics/MLXR)
- [Documentation](https://github.com/LayerDynamics/MLXR/tree/main/docs)
- [Issue Tracker](https://github.com/LayerDynamics/MLXR/issues)

## Advanced Features

### Automatic Retry with Exponential Backoff

The SDK automatically retries requests that fail due to transient network errors or specific HTTP status codes (408, 429, 500, 502, 503, 504 by default).

```typescript
const client = new MLXRClient({
  retry: {
    maxRetries: 3,                // Retry up to 3 times
    initialDelay: 1000,           // Start with 1 second delay
    backoffMultiplier: 2,         // Double the delay each time (1s, 2s, 4s)
    maxDelay: 10000,              // Cap delays at 10 seconds
    retryableStatusCodes: [408, 429, 500, 502, 503, 504],
  },
});
```

The following errors are automatically retried:
- Network errors (ECONNRESET, ETIMEDOUT, ENOTFOUND, ECONNREFUSED)
- Configurable HTTP status codes (default: 408, 429, 500, 502, 503, 504)

### Enhanced Health Check

Check the health of both OpenAI and Ollama APIs:

```typescript
// Check OpenAI API only (default)
const health = await client.health();
console.log(health); // { status: 'ok', details: { openai: 'ok' } }

// Check both APIs
const health = await client.health(['openai', 'ollama']);
console.log(health);
// { status: 'ok', details: { openai: 'ok', ollama: 'ok' } }

// Handle errors
try {
  const health = await client.health(['openai', 'ollama']);
  if (health.status === 'error') {
    console.error('Health check failed:', health.details);
  }
} catch (error) {
  console.error('Health check error:', error);
}
```

### HTTPS Support

The SDK automatically uses HTTPS when the base URL starts with `https://`:

```typescript
const client = new MLXRClient({
  baseUrl: 'https://api.example.com',  // Automatically uses HTTPS
});
```

### Error Handling

All errors include detailed information for debugging:

```typescript
try {
  const response = await client.openai.createChatCompletion({
    model: 'nonexistent-model',
    messages: [{ role: 'user', content: 'Hello!' }],
  });
} catch (error) {
  if (error instanceof Error) {
    console.error('Error message:', error.message);
    // For HTTP errors, statusCode is included
    const statusCode = (error as any).statusCode;
    if (statusCode) {
      console.error('HTTP status code:', statusCode);
    }
  }
}
```

