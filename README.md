# MLXR

High-performance, macOS-native LLM inference engine for Apple Silicon.

## Overview

MLXR is a local LLM runner built specifically for Apple Silicon (M4, M3, M2) that combines:

- **MLX framework** for tensor/graph management
- **Custom Metal kernels** for performance-critical operations
- **OpenAI and Ollama-compatible APIs** for seamless integration
- **React-based GUI** with real-time streaming

### Key Features

- **Native Performance**: Custom Metal kernels optimized for Apple's unified memory architecture
- **Memory Efficient**: Paged KV cache with smart eviction policies
- **High Throughput**: Continuous batching and speculative decoding
- **Model Support**: GGUF, HF safetensors, and native MLX formats
- **Quantization**: Full support for Q2_K through Q8_K, FP8, and NF4
- **Developer Friendly**: OpenAI and Ollama-compatible REST APIs

## Architecture

```
┌─────────────────────────┐
│   React WebView GUI     │
│   (Tray/Dock App)       │
└───────────┬─────────────┘
            │ Unix Domain Socket
┌───────────▼─────────────┐
│   Daemon (REST/gRPC)    │
│   - OpenAI API          │
│   - Ollama API          │
│   - Model Registry      │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│   Inference Core        │
│   - MLX Graph           │
│   - Metal Kernels       │
│   - Paged KV Cache      │
│   - Continuous Batching │
└─────────────────────────┘
```

## Performance Targets (M4)

- **First Token**: < 1s for 7B-8B models at 4-bit
- **Decode**: < 80ms/token steady-state
- **Embeddings**: < 20ms/sample
- **Occupancy**: ≥ 60% GPU utilization on attention kernels

## Requirements

- macOS 14.0 (Sonoma) or later
- Apple Silicon (M2, M3, or M4)
- Xcode 15+ (for building)
- Homebrew package manager
- CMake 3.20+
- Python 3.11+ (with MLX installed)
- Node.js 18+ (for frontend development)

### System Dependencies

The following Homebrew packages are required:

```bash
# Install all dependencies at once
brew install cmake ninja mlx sentencepiece nlohmann-json cpp-httplib googletest

# Or install individually
brew install cmake          # Build system
brew install ninja          # Fast build tool
brew install mlx            # Apple's ML framework
brew install sentencepiece  # Tokenization library
brew install nlohmann-json  # JSON for C++
brew install cpp-httplib    # HTTP server library
brew install googletest     # C++ testing framework
```

## Project Status

🚧 **Early Development** - Phase 0 (Foundation)

This project is in active development. Currently implementing:

- [x] Repository structure
- [x] Build system setup
- [ ] Metal kernel compilation pipeline
- [ ] Basic MLX integration
- [ ] Minimal inference engine

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidelines.

## Repository Structure

```
MLXR/
  app/           # macOS app bundle & React UI
  daemon/        # Background server (REST/gRPC)
  core/          # Inference engine (C++/MLX/Metal)
  tools/         # Model converters and utilities
  sdks/          # Client libraries (Python, TS, Swift)
  configs/       # Configuration files
  scripts/       # Build and development scripts
  tests/         # Test suites
  plan/          # Architecture specifications
```

## Getting Started

### 1. Install Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Clone the repository
git clone <repository-url>
cd MLXR

# Install system dependencies
make install-deps

# Setup Python environment (recommended)
make setup
conda activate mlxr

# Check installation
make status
```

### 2. Build

```bash
# Full build (Metal shaders + C++ core)
make build

# Or quick development build (Metal only)
make dev

# Run tests
make test-cpp
```

### 3. Run

```bash
# Run daemon
./build/cmake/bin/mlxrunnerd

# Develop frontend (separate terminal)
cd app/ui
yarn install
yarn dev
```

For detailed build instructions, see [CLAUDE.md](CLAUDE.md).

## Development Phases

### Phase 0: Foundation (Current)

- Repository structure and build system
- Metal shader compilation pipeline
- Basic toolchain validation

### Phase 1: Minimal Inference Core

- MLX integration and basic model loading
- SentencePiece tokenizer
- Single-request inference (FP16)

### Phase 2: REST API & Daemon

- Daemon process with Unix domain socket
- Basic OpenAI-compatible endpoints
- Simple model registry

### Phase 3+

See [plan/SPEC01.md](plan/SPEC01.md) for complete roadmap.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Comprehensive development guide for contributors
- [plan/SPEC01.md](plan/SPEC01.md) - Complete specification and requirements
- [plan/Structure.md](plan/Structure.md) - Architecture and component overview
- [plan/MetalKernelsPlan.md](plan/MetalKernelsPlan.md) - Metal kernel specifications
- [plan/PackagingDistro.md](plan/PackagingDistro.md) - Distribution strategy
- [plan/FrontendPlan.md](plan/FrontendPlan.md) - React UI implementation plan

## Contributing

This project is in early development. Contributions welcome once Phase 1 is complete.

For development guidelines, see [CLAUDE.md](CLAUDE.md).

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- Metal - Apple's GPU compute API
- Inspired by [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), and [Ollama](https://github.com/ollama/ollama)

---

**Status**: Pre-alpha development | **Target**: Q2 2025 MVP release
