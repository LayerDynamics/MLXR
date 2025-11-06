# MLXR TypeScript SDK Examples

This directory contains comprehensive examples demonstrating the usage of the MLXR TypeScript SDK.

## Prerequisites

1. MLXR daemon must be running
2. At least one model should be available (e.g., `tinyllama`)

## Running Examples

First, build the SDK:

```bash
cd ..
npm install
npm run build
```

Then run any example:

```bash
# OpenAI-style chat
npm run openai-chat

# Ollama-style chat
npm run ollama-chat

# Text completions
npm run completions

# Embeddings
npm run embeddings

# Model management
npm run model-management
```

Or run directly:

```bash
node ../dist/examples/openai-chat.js
```

## Examples Overview

### 1. OpenAI Chat (`openai-chat.ts`)

Demonstrates:
- Non-streaming chat completions
- Streaming chat completions
- System and user messages
- Temperature and token limits

### 2. Ollama Chat (`ollama-chat.ts`)

Demonstrates:
- Non-streaming chat with Ollama API
- Streaming chat with Ollama API
- Performance metrics (eval count, duration)

### 3. Completions (`completions.ts`)

Demonstrates:
- OpenAI-style text completions
- OpenAI-style streaming completions
- Ollama-style generation
- Ollama-style streaming generation
- Stop sequences

### 4. Embeddings (`embeddings.ts`)

Demonstrates:
- OpenAI-style embeddings
- Ollama-style embeddings
- Computing text similarity with cosine similarity
- Batch embedding generation

### 5. Model Management (`model-management.ts`)

Demonstrates:
- Listing all available models
- Getting detailed model information
- Listing running models
- Pulling models with progress tracking
- Model statistics and metadata

## Customization

All examples use default configuration. You can customize by modifying the client initialization:

```typescript
const client = new MLXRClient({
  baseUrl: 'http://localhost:11434',
  apiKey: 'your-api-key',
  timeout: 60000,
});
```

## Troubleshooting

**Connection Error**
- Ensure MLXR daemon is running
- Check if the Unix socket exists (macOS): `~/Library/Application Support/MLXRunner/run/mlxrunner.sock`
- Verify the HTTP port is correct (default: 11434)

**Model Not Found**
- List available models: see `model-management.ts` example
- Pull the required model first

**Timeout Error**
- Increase timeout in client configuration
- Check if the model is loaded and responsive
