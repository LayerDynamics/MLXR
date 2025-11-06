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

# Or use the Makefile convenience target
make install-deps
```

**Note:** CMake and Ninja must be installed via Homebrew, not Conda. The conda-forge "cmake" package is unrelated to the CMake build system.

## Project Status

✅ **Phase 2 Complete** - Service Layer & Optimization (~85%)

This project has completed Phase 1 (Minimal Inference) and most of Phase 2 (Optimization).

### Completed Features

**Phase 1: Minimal Inference** ✅ 100%
- [x] Complete Llama model with safetensors loading
- [x] SentencePiece tokenizer integration
- [x] Sampling strategies (greedy, temperature, top-k, top-p)
- [x] Working text generation pipeline
- [x] Example: [simple_generation.cpp](examples/simple_generation.cpp)

**Phase 2: Optimization** ✅ ~85%
- [x] **KV Cache System** - Paged arena with LRU eviction, GQA support (87.5% memory reduction)
- [x] **Scheduler** - Continuous batching with prefill/decode separation
- [x] **Metal Kernels** - RMSNorm (fully tested), attention, RoPE, SwiGLU, Q-Gemm (implemented)
- [x] **Test Daemon** - Working HTTP server with health/models endpoints
- ⏳ **Integration** - CachedLlamaModel exists but needs Engine integration

**Phase 3: Service Layer** ✅ ~60%
- [x] **REST API** - OpenAI & Ollama-compatible endpoints
- [x] **SSE Streaming** - Real-time token generation
- [x] **Model Registry** - SQLite catalog with GGUF parser
- [x] **Telemetry** - Comprehensive metrics collection
- [x] **Test Suite** - 14 C++ unit test files (81/81 RMSNorm tests passing)

**Frontend** ✅ 100%
- [x] **React UI** - 78 components fully implemented
- [x] **Chat Interface** - With streaming and tool calls
- [x] **Model Management** - Pull, import, convert, quantize
- [x] **Metrics Dashboard** - Real-time performance visualization

### Current Performance (TinyLlama 1.1B)
- Prefill: 198-459 ms (5-10 tokens)
- Decode: 53-220 ms/token
- Throughput: 4.5-18.9 tokens/sec
- Memory: 87.5% reduction with GQA (308 MB saved)

See [CLAUDE.md](CLAUDE.md) and [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for detailed status.

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

### ✅ Phase 0: Foundation (COMPLETE)
- Repository structure and build system
- Metal shader compilation pipeline
- Toolchain validation (Homebrew, CMake, Ninja)

### ✅ Phase 1: Minimal Inference Core (COMPLETE)
- MLX integration and model loading (safetensors)
- SentencePiece tokenizer
- Single-request inference with FP16
- Working examples in `examples/`

### ⏳ Phase 2: Optimization (85% COMPLETE)
- Paged KV cache with eviction policies
- Metal kernel implementations (RMSNorm tested, others implemented)
- Continuous batching scheduler
- GQA support for memory efficiency
- **Next:** CachedLlamaModel integration with Engine

### ⏳ Phase 3: Service Layer (60% COMPLETE)
- REST API daemon (OpenAI & Ollama compatible)
- Model registry with SQLite backend
- SSE streaming for real-time generation
- Telemetry and metrics
- **Next:** Full API endpoint integration

### 🔜 Phase 4: Frontend & Distribution (Planned)
- macOS app bundle with React WebView
- Unix domain socket communication
- Auto-updates via Sparkle
- Code signing and notarization

See [plan/SPEC01.md](plan/SPEC01.md) for complete roadmap and [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for current status.

## Documentation

**Developer Guides:**
- [CLAUDE.md](CLAUDE.md) - Comprehensive development guide and coding standards
- [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) - Current implementation status and metrics
- [docs/SECURITY_FIXES.md](docs/SECURITY_FIXES.md) - Security vulnerability tracking and best practices

**Architecture & Planning:**
- [plan/SPEC01.md](plan/SPEC01.md) - Complete specification and requirements
- [plan/Structure.md](plan/Structure.md) - Architecture and component overview
- [plan/MetalKernelsPlan.md](plan/MetalKernelsPlan.md) - Metal kernel specifications
- [plan/PackagingDistro.md](plan/PackagingDistro.md) - Distribution strategy
- [plan/FrontendPlan.md](plan/FrontendPlan.md) - React UI implementation plan

**Implementation Details:**
- [docs/PHASE2_COMPLETION.md](docs/PHASE2_COMPLETION.md) - Scheduler-engine integration details
- [docs/DAEMON_STATUS.md](docs/DAEMON_STATUS.md) - Daemon components status
- [docs/KV_CACHE_IMPLEMENTATION.md](docs/KV_CACHE_IMPLEMENTATION.md) - Paged KV cache architecture
- [docs/GQA_RESHAPE_FIX.md](docs/GQA_RESHAPE_FIX.md) - Critical GQA support fix
- [app/ui/COMPONENTS.md](app/ui/COMPONENTS.md) - React UI components documentation

## Contributing

This project is actively developed and welcomes contributions!

**Current Focus Areas:**
- CachedLlamaModel integration with Engine
- Metal kernel optimization and testing
- OpenAI API endpoint completion
- Performance benchmarking and profiling

**Before Contributing:**
1. Read [CLAUDE.md](CLAUDE.md) for development guidelines
2. Check [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for current status
3. Review [docs/SECURITY_FIXES.md](docs/SECURITY_FIXES.md) for security best practices

**Development Standards:**
- ✅ All C++ code passes unit tests
- ✅ Security: No `system()` calls, proper input validation, ReDoS-safe regex
- ✅ Cross-platform: Use `std::filesystem` for paths
- ✅ Documentation: Update docs for significant changes

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- Metal - Apple's GPU compute API
- Inspired by [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), and [Ollama](https://github.com/ollama/ollama)

---

**Status**: Active Development (Phase 2 Complete, Phase 3 In Progress)
**Target**: Q1 2025 MVP release
**Latest**: See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for current metrics and progress
