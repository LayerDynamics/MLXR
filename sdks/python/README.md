# MLXR Python SDK

Python client library for [MLXR](https://github.com/LayerDynamics/MLXR) - a high-performance LLM inference engine for Apple Silicon.

## Installation

```bash
pip install mlxr
```

Or install from source:

```bash
cd sdks/python
pip install -e .
```

## Quick Start

### Using the OpenAI-Compatible API

```python
from mlxrunner import MLXR

# Initialize client
client = MLXR()

# Chat completion
response = client.chat.completions.create(
    model="TinyLlama-1.1B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)
print(response.choices[0].message.content)

# Streaming chat completion
stream = client.chat.completions.create(
    model="TinyLlama-1.1B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Using the Ollama-Compatible API

```python
from mlxrunner import MLXR

client = MLXR()

# Generate
response = client.ollama.generate(
    model="TinyLlama-1.1B",
    prompt="Why is the sky blue?"
)
print(response["response"])

# Streaming generate
for chunk in client.ollama.generate(
    model="TinyLlama-1.1B",
    prompt="Write a haiku",
    stream=True
):
    print(chunk["response"], end="", flush=True)

# Chat
response = client.ollama.chat(
    model="TinyLlama-1.1B",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["message"]["content"])
```

### Model Management

```python
from mlxrunner import MLXR

client = MLXR()

# List models
models = client.models.list()
for model in models:
    print(f"{model.id}: {model.params} parameters")

# Pull a model
client.models.pull("llama2:7b")

# Show model info
info = client.models.show("llama2:7b")
print(f"Context length: {info['context_length']}")

# Delete a model
client.models.delete("llama2:7b")
```

### Async Support

```python
import asyncio
from mlxrunner import AsyncMLXR

async def main():
    client = AsyncMLXR()

    # Async chat completion
    response = await client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

    # Async streaming
    stream = await client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[{"role": "user", "content": "Tell me a joke"}],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(main())
```

## Configuration

### Unix Domain Socket (Default)

By default, MLXR connects via Unix Domain Socket:

```python
client = MLXR()  # Uses default socket path
```

### HTTP Connection

To connect via HTTP:

```python
client = MLXR(base_url="http://localhost:11434")
```

### Custom Socket Path

```python
client = MLXR(socket_path="/custom/path/to/mlxrunner.sock")
```

### Authentication

```python
client = MLXR(api_key="your-api-key")
```

## API Reference

### OpenAI-Compatible API

The SDK implements the OpenAI API interface:

- `client.chat.completions.create()` - Chat completions
- `client.completions.create()` - Text completions
- `client.embeddings.create()` - Text embeddings
- `client.models.list()` - List models
- `client.models.retrieve(model_id)` - Get model details

### Ollama-Compatible API

The SDK also implements the Ollama API:

- `client.ollama.generate()` - Generate completion
- `client.ollama.chat()` - Chat with model
- `client.ollama.embeddings()` - Generate embeddings
- `client.ollama.pull()` - Pull a model
- `client.ollama.create()` - Create a model from Modelfile
- `client.ollama.copy()` - Copy a model
- `client.ollama.delete()` - Delete a model
- `client.ollama.show()` - Show model information
- `client.ollama.list()` - List local models
- `client.ollama.ps()` - List running models

## CLI Usage

The SDK includes a command-line interface:

```bash
# Check status
mlxr status

# List models
mlxr models list

# Pull a model
mlxr models pull llama2:7b

# Run inference
mlxr chat "What is the meaning of life?" -m TinyLlama-1.1B

# Start the daemon
mlxr serve
```

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLXR daemon running

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mlxrunner tests

# Lint
ruff check mlxrunner tests

# Type check
mypy mlxrunner
```

## License

MIT License - see LICENSE file for details.

## Links

- [GitHub Repository](https://github.com/LayerDynamics/MLXR)
- [Documentation](https://github.com/LayerDynamics/MLXR/tree/main/docs)
- [Issues](https://github.com/LayerDynamics/MLXR/issues)
