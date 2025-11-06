# MLXR Python SDK Examples

This directory contains example scripts demonstrating how to use the MLXR Python SDK.

## Prerequisites

1. Install the SDK:
   ```bash
   pip install mlxr
   # or for development:
   pip install -e ../
   ```

2. Ensure the MLXR daemon is running:
   ```bash
   # Check daemon status
   mlxr status

   # Or start the daemon
   mlxrunnerd
   ```

## Examples

### Basic Chat (`basic_chat.py`)

Simple chat completion using the OpenAI-compatible API.

```bash
python basic_chat.py
```

Demonstrates:
- Initializing the MLXR client
- Creating a chat completion
- Accessing response data
- Token usage tracking

### Streaming Chat (`streaming_chat.py`)

Real-time token streaming with SSE.

```bash
python streaming_chat.py
```

Demonstrates:
- Streaming chat completions
- Processing chunks as they arrive
- Real-time output display

### Ollama Generate (`ollama_generate.py`)

Using the Ollama-compatible API for text generation.

```bash
python ollama_generate.py
```

Demonstrates:
- Non-streaming generation
- Streaming generation
- Ollama API compatibility

### Model Management (`model_management.py`)

Listing and managing models.

```bash
python model_management.py
```

Demonstrates:
- Listing models (OpenAI & Ollama APIs)
- Checking running models
- Pulling models from registry
- Viewing model information

### Async Chat (`async_chat.py`)

Async/await usage with AsyncMLXR client.

```bash
python async_chat.py
```

Demonstrates:
- Async client initialization
- Async chat completions
- Async streaming
- Context manager usage

### Embeddings (`embeddings.py`)

Generating text embeddings.

```bash
python embeddings.py
```

Demonstrates:
- Single text embedding
- Batch embeddings
- OpenAI and Ollama embedding APIs

### Health and Metrics (`health_and_metrics.py`)

Checking daemon health and performance metrics.

```bash
python health_and_metrics.py
```

Demonstrates:
- Health check endpoint
- Metrics retrieval
- Performance monitoring

## Running All Examples

```bash
# Make scripts executable
chmod +x *.py

# Run all examples
for script in *.py; do
    echo "Running $script..."
    python "$script"
    echo ""
done
```

## Configuration

### Using Unix Domain Socket (Default)

```python
from mlxrunner import MLXR

client = MLXR()  # Uses default socket path
```

### Using HTTP

```python
from mlxrunner import MLXR

client = MLXR(base_url="http://localhost:11434")
```

### Custom Socket Path

```python
from mlxrunner import MLXR

client = MLXR(socket_path="/custom/path/to/mlxrunner.sock")
```

### With Authentication

```python
from mlxrunner import MLXR

client = MLXR(api_key="your-api-key")
```

## Error Handling

All examples use basic error handling. For production use, consider:

```python
from mlxrunner import MLXR, MLXRConnectionError, MLXRTimeoutError

try:
    client = MLXR()
    response = client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except MLXRConnectionError:
    print("Failed to connect to daemon. Is it running?")
except MLXRTimeoutError:
    print("Request timed out")
except Exception as e:
    print(f"Error: {e}")
```

## Advanced Usage

### Context Manager

```python
from mlxrunner import MLXR

with MLXR() as client:
    response = client.chat.completions.create(
        model="TinyLlama-1.1B",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
# Client automatically closed
```

### Async Context Manager

```python
import asyncio
from mlxrunner import AsyncMLXR

async def main():
    async with AsyncMLXR() as client:
        response = await client.chat.completions.create(
            model="TinyLlama-1.1B",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

### Custom Sampling Parameters

```python
response = client.chat.completions.create(
    model="TinyLlama-1.1B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0.9,      # Higher = more creative
    top_p=0.95,          # Nucleus sampling
    max_tokens=500,      # Maximum response length
    stop=["END"],        # Stop sequences
    presence_penalty=0.5, # Penalize repetition
    frequency_penalty=0.5 # Penalize frequency
)
```

## Troubleshooting

### Daemon Not Running

```bash
# Check if daemon is running
ps aux | grep mlxrunnerd

# Check socket exists
ls -la ~/Library/Application\ Support/MLXRunner/run/mlxrunner.sock

# Start daemon
mlxrunnerd
```

### Connection Issues

```python
from mlxrunner import MLXR, MLXRConnectionError

try:
    client = MLXR()
    health = client.health()
    print(f"Daemon status: {health['status']}")
except MLXRConnectionError as e:
    print(f"Connection error: {e}")
    print("Is the daemon running?")
```

### Import Errors

```bash
# Ensure SDK is installed
pip show mlxr

# Or install in development mode
pip install -e ../
```

## Further Reading

- [SDK Documentation](../README.md)
- [MLXR Documentation](https://github.com/LayerDynamics/MLXR)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
