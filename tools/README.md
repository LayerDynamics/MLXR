# MLXR Tools

Utilities for model conversion, quantization, and development.

## Model Conversion Tools

### convert_hf_to_mlx.py

Convert HuggingFace models (safetensors/pytorch) to MLX format.

**Usage:**
```bash
python tools/convert_hf_to_mlx.py \
    --input meta-llama/Llama-2-7b-hf \
    --output models/llama-2-7b-mlx \
    --dtype float16
```

**Supported Models:**
- Llama (all variants)
- Mistral
- Qwen
- Gemma
- Phi

**Features:**
- Automatic weight name conversion
- FP16/BF16 quantization
- Tokenizer preservation
- MLX NPZ format output

### Additional Converters (External Tools)

For GGUF format support, use external tools:

**HuggingFace → GGUF:**
```bash
# Using llama.cpp's convert.py
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python convert.py /path/to/hf/model --outtype f16 --outfile model.gguf
```

**GGUF Quantization:**
```bash
# Quantize to Q4_K_M
./llama.cpp/quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

**MLX → GGUF:**
```bash
# Convert MLX back to HF, then to GGUF
python tools/mlx_to_hf.py --input models/mlx --output models/hf
python llama.cpp/convert.py models/hf --outfile model.gguf
```

## Configuration Validation

### validate_configs.py

Validates server and model configuration files.

**Usage:**
```bash
python tools/validate_configs.py [config_dir]
```

**Checks:**
- Server config structure (transport, scheduler, kv_cache)
- Model config completeness
- YAML syntax errors
- Required fields

## Development Tools

### Metal Kernel Compilation

Compile custom Metal shaders:

```bash
./scripts/build_metal.sh
```

Generates `kernels.metallib` with all kernel variants.

### Test Data Generation

Generate synthetic test data for benchmarking:

```bash
python tools/generate_test_data.py \
    --vocab-size 32000 \
    --seq-length 2048 \
    --num-samples 100 \
    --output tests/data/
```

## Model Management

### Downloading Models

**From HuggingFace:**
```bash
# Using huggingface-cli
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir models/llama-2-7b-hf
```

**From Ollama Registry:**
```bash
# Pull via daemon
curl -X POST http://localhost:11434/api/pull \
    -d '{"name": "llama2:7b"}'
```

### Model Registry Management

Add model to registry:

```bash
# Create model config
cat > configs/models/my-model.yaml << EOF
model:
  name: "My Custom Model"
  family: "llama"
  path: "~/models/my-model"
  format: "mlx"
# ... rest of config
EOF

# Validate
python tools/validate_configs.py configs

# Register via API
curl -X POST http://localhost:11434/api/tags \
    -d @configs/models/my-model.yaml
```

## Benchmarking

### Throughput Benchmark

```bash
python tools/benchmark.py \
    --model models/llama-2-7b-mlx \
    --batch-sizes 1,4,8,16 \
    --seq-lengths 128,512,1024,2048 \
    --dtype float16
```

### Latency Profiling

```bash
python tools/profile_latency.py \
    --model models/llama-2-7b-mlx \
    --prompt "The quick brown fox" \
    --max-tokens 100 \
    --iterations 10
```

## Debugging Tools

### Daemon Logs

View daemon logs:

```bash
tail -f ~/Library/Logs/mlxrunnerd.out.log
tail -f ~/Library/Logs/mlxrunnerd.err.log
```

### Health Check

```bash
curl http://localhost:11434/health
```

### Metrics Export

```bash
curl http://localhost:11434/metrics
```

## Dependencies

Install all tool dependencies:

```bash
# Python packages
pip install -r tools/requirements.txt

# Or via conda
conda env create -f environment.yml
conda activate mlxr
```

## Contributing Tools

When adding new tools:

1. Place in `tools/` directory
2. Add shebang: `#!/usr/bin/env python3`
3. Make executable: `chmod +x tools/your_tool.py`
4. Add argparse with `--help` support
5. Document in this README
6. Add to `tools/requirements.txt` if needed

## License

Copyright © 2025 MLXR Development. All rights reserved.
