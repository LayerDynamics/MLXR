# MLXR Examples

This directory contains example programs demonstrating how to use MLXR for LLM inference on Apple Silicon.

## Building Examples

Examples are built automatically when you run:

```bash
make build
```

The compiled executables will be in `build/cmake/bin/`.

## Examples

### 1. Simple Generation

**File**: `simple_generation.cpp`

A basic example showing text generation with a pre-trained model.

**Usage**:
```bash
./build/cmake/bin/simple_generation <model_dir> <tokenizer_path> <prompt>
```

**Example**:
```bash
./build/cmake/bin/simple_generation \
    ./models/TinyLlama-1.1B \
    ./models/tokenizer.model \
    "Once upon a time"
```

**What it does**:
- Loads a Llama-based model from safetensors format
- Loads a SentencePiece tokenizer
- Generates text using top-p sampling with temperature
- Displays generation progress in real-time

## Preparing Models

### Option 1: Download Pre-converted Models

If you have a model already in safetensors format from HuggingFace:

```bash
# Example with TinyLlama
mkdir -p models/TinyLlama-1.1B
cd models/TinyLlama-1.1B

# Download model files
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --include "*.safetensors" "config.json" "tokenizer.model" \
    --local-dir .
```

### Option 2: Convert MLX Models

If you have MLX format models (`.npz`), you need to convert them to safetensors:

```python
import mlx.core as mx
import numpy as np

# Load MLX weights
weights = mx.load("model.npz")

# Save as safetensors
mx.save_safetensors("model.safetensors", weights)
```

### Required Files

For each model, you need:
- `config.json` - Model configuration (architecture, dimensions, etc.)
- `model.safetensors` - Model weights in safetensors format
- `tokenizer.model` - SentencePiece tokenizer model

## Generation Configuration

You can customize generation behavior by modifying the `GenerationConfig`:

```cpp
mlxr::runtime::GenerationConfig config;

// Maximum tokens to generate
config.max_new_tokens = 128;

// Sampling parameters
config.sampler_config.temperature = 0.7f;  // Randomness (0.0 = greedy, higher = more random)
config.sampler_config.top_p = 0.9f;        // Nucleus sampling threshold
config.sampler_config.top_k = 50;          // Top-k sampling (0 = disabled)

// Stop tokens
config.stop_tokens = {2};  // EOS token ID

// Output options
config.echo_prompt = true;   // Include prompt in output
config.verbose = true;       // Show generation progress
```

## Sampling Strategies

MLXR supports multiple sampling strategies:

1. **Greedy Sampling**: Set `temperature = 0.0` for deterministic, highest-probability selection
2. **Temperature Sampling**: Control randomness with `temperature` (0.1-2.0)
3. **Top-k Sampling**: Sample from k most likely tokens
4. **Top-p (Nucleus) Sampling**: Sample from tokens with cumulative probability ≤ p
5. **Combined**: Use top-k + top-p together for best results

## Performance Tips

- Models run entirely on Apple Silicon GPU via Metal
- First run may be slower due to Metal shader compilation
- Subsequent runs use cached shaders for faster startup
- Generation speed depends on model size and sequence length

## Troubleshooting

### "Failed to load model"
- Verify `config.json` exists and is valid JSON
- Check that safetensors file exists and is not corrupted
- Ensure model architecture matches TinyLlama/Llama format

### "Failed to load tokenizer"
- Tokenizer must be SentencePiece format (`.model` file)
- HuggingFace tokenizers (`.json`) are not yet supported

### "NPZ loading not yet implemented"
- Convert MLX `.npz` models to safetensors format
- Use `mx.save_safetensors()` in Python

## API Usage

For integration into your own C++ projects:

```cpp
#include "runtime/engine.h"

// Load engine
auto engine = mlxr::runtime::load_engine(
    "./models/TinyLlama-1.1B",
    "./models/tokenizer.model"
);

// Generate text
std::string output = engine->generate("Hello, world!");
```

## Next Steps

- See the main [README.md](../README.md) for project overview
- Check [SETUP.md](../SETUP.md) for build instructions
- Explore the [tests/](../tests/) directory for unit tests
- Read the [docs/](../docs/) for architecture details
