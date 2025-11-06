# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CRITICAL: Always activate conda before working: `conda activate mlxr`**
(If conda isn't initialized: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mlxr`)

**CRITICAL DEVELOPMENT PRINCIPLE: If something is called but missing, it should be IMPLEMENTED, not removed.**
This is a core tenet of the project - we build forward, not backward.

MLXR is a high-performance, macOS-native LLM inference engine built specifically for Apple Silicon (M4, M3, M2). It combines MLX (Apple's machine learning framework) with custom Metal compute kernels and a native Objective-C/Swift/C++ runtime to deliver vLLM/llama.cpp/Ollama feature parity with optimal performance on unified memory architectures.

**Core differentiators:**

- MLX-first tensor/graph management with custom Metal kernels for hot paths
- Paged KV cache with continuous batching across unified memory
- OpenAI and Ollama-compatible REST APIs
- React-based tray/dock GUI with real-time streaming
- Support for GGUF, HF safetensors, and native MLX model formats

## Repository Structure

```
MLXR/
  app/
    macos/          # Swift/ObjC host (tray/dock, Sparkle updater)
    ui/             # React + Vite WebView bundle
  daemon/
    server/         # REST/gRPC, SSE streaming, OpenAI/Ollama API shims
    scheduler/      # Prefill/decode queues, continuous batching
    registry/       # SQLite model catalog, mmap loaders
    telemetry/      # Metrics, tracing, profiling
  core/
    graph/          # MLX module definitions (layers, attention, MLP)
    kernels/
      metal/        # .metal shaders (fused attention, RoPE, RMSNorm, quantized matmuls)
      cpu/          # Neon/SIMD fallbacks
    runtime/
      tokenizer/    # SentencePiece, HF tokenizers, tiktoken
      kv/           # Arena, pager, eviction, persistence
      spec/         # Speculative decoding (draft model proposer/verifier)
  tools/            # Model converters (HF↔GGUF↔MLX) and quantizers
  sdks/             # Client SDKs (Python, TypeScript, Swift)
  configs/          # Server & model configs (YAML)
  scripts/          # Build helpers (Metal compilation, app bundle, daemon)
  plan/             # Architecture specs and planning documents
```

## Quick Start

Before any development work:

```bash
# 1. Activate conda (REQUIRED)
conda activate mlxr

# 2. Check environment status
make status

# 3. Build the project
make build              # Full build (Metal + CMake)
# OR
make dev                # Quick dev setup (Metal only)

# 4. Run tests
make test-cpp           # C++ unit tests
make validate           # Quick validation
```

## Current Implementation Phase

**Status**: Phase 1 COMPLETE, Phase 2 IN PROGRESS

### ✅ Phase 1: Minimal Inference Core (COMPLETE)

- Complete Llama model with safetensors loading
- SentencePiece tokenizer
- Sampling strategies (greedy, temperature, top-k, top-p)
- Working text generation
- Example: [simple_generation.cpp](examples/simple_generation.cpp)

### 🚧 Phase 2: Optimization (IN PROGRESS)

- ✅ **KV Cache System**: Paged cache with eviction policies - COMPLETE
- ✅ **RMSNorm Metal Kernel**: Custom kernel via MLX Primitive API - 81/81 tests passing
- ✅ **Attention Decode Kernel**: Paged KV decode path with Metal - COMPLETE (266 lines)
- 🚧 **Attention Prefill Kernel**: Metal shader complete, primitive implementation in progress
- ⏳ **Quantization**: GGUF loading and K-quants - PENDING

**Integration Status:**

- ⚠️ **CachedLlamaModel exists but not integrated with Engine** - Infrastructure complete, needs Engine refactor
- See [docs/SESSION_2025_11_06_INTEGRATION.md](docs/SESSION_2025_11_06_INTEGRATION.md) for integration plan

### 🚧 Phase 3: Service Layer (PARTIALLY COMPLETE - ~40%)

The daemon layer has significant components already implemented:

- ✅ **Scheduler**: Request management, batching logic ([daemon/scheduler/](daemon/scheduler/))
- ✅ **REST Server**: HTTP server with routing ([daemon/server/rest_server.{h,cpp}](daemon/server/rest_server.h))
- ✅ **Ollama API**: Compatible endpoints ([daemon/server/ollama_api.{h,cpp}](daemon/server/ollama_api.h))
- ✅ **SSE Streaming**: Server-sent events ([daemon/server/sse_stream.{h,cpp}](daemon/server/sse_stream.h))
- ✅ **Metrics**: Telemetry collection ([daemon/telemetry/metrics.{h,cpp}](daemon/telemetry/metrics.h))
- ✅ **Model Registry**: Basic catalog ([daemon/registry/model_registry.{h,cpp}](daemon/registry/model_registry.h))
- ✅ **GGUF Parser**: Format reader ([daemon/registry/gguf_parser.{h,cpp}](daemon/registry/gguf_parser.h))
- ✅ **Scheduler Worker**: Request processing ([daemon/server/scheduler_worker.{h,cpp}](daemon/server/scheduler_worker.h))
- ⏳ OpenAI-compatible endpoints - PENDING
- ⏳ Authentication - PENDING

**Test Status:** 261 tests total, 259 passing (99.2% pass rate)

- Scheduler tests: 10/12 passing
- Worker tests: 9/9 passing
- REST server tests: 15/15 passing
- Metrics tests: 15/15 passing

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for detailed status.

## Architecture Principles

### Unified Memory Strategy

- Memory-map model weights with page-aligned offsets matching KV block size
- Paged KV cache: global arena split into fixed blocks (16-64 tokens/block)
- Smart CPU/GPU/ANE placement with unified-memory-aware paging
- Minimize cross-device copies; use pinned host staging buffers

### Performance Core

- **Continuous batching**: Merge requests at token boundaries; split prefill (GPU-bound) and decode (latency-sensitive) queues
- **Paged KV cache**: Free lists and page tables per sequence; working-set-aware LRU eviction
- **Kernel fusion**: Custom Metal kernels fuse QKV projection → RoPE → attention score → softmax → context (FlashAttention-style)
- **Speculative decoding**: Optional draft model (enabled by default) proposes k tokens; main model verifies using shared KV cache
- **Quantization**: Support GGUF K-quants (Q2_K-Q8_K), IQ variants, FP8/NF4; dequant in Metal shaders with vectorized loads

### Device Placement Heuristics

- Weights on GPU
- KV blocks on GPU with overflow to CPU pinned memory
- Prefill operations on GPU
- Light samplers on CPU
- Opportunistic ANE for activation functions and small convolutions in VLMs

## Metal Kernel Implementation

**Phase 1 Status**: ✅ **RMSNorm kernel COMPLETE** - 81/81 tests passing

- Fully functional Metal RMSNorm implementation using MLX Primitive API
- Handles contiguous and non-contiguous inputs via flatten-reshape pattern
- See [docs/PHASE1_METAL_RMSNORM_COMPLETION.md](docs/PHASE1_METAL_RMSNORM_COMPLETION.md) for details

### Key Kernels

1. **attention_prefill_fused**: Fused prefill path (X·Wqkv → [Q,K,V] → RoPE → attention → context) [Phase 2]
2. **attention_decode_fused**: Decode path with paged KV walker [Phase 2]
3. **q_gemm_dequant**: Quantized matmul with on-the-fly dequantization [Phase 2]
4. **rope_apply**: Rotary positional embedding (supports base, NTK-scaled, YaRN-scaled) [Phase 2]
5. **rmsnorm_fused**: ✅ **IMPLEMENTED** - RMSNorm with FP32/FP16 variants (see [core/kernels/metal/rmsnorm.metal](core/kernels/metal/rmsnorm.metal))
6. **swiglu_mlp_fused**: Gated MLP with optional quantized weights [Phase 2]
7. **kv_pack_store / kv_load_unpack**: Efficient KV block storage/retrieval [Phase 2]
8. **kv_persist_copy**: Async copy between GPU/CPU for persistence/eviction [Phase 2]

### Kernel Variants

- **head_dim**: 64, 80, 96, 112, 128, 160, 192, 256
- **block_tokens**: 16, 32
- **weight dtype**: fp16, fp8 (E4M3/E5M2), int4 (Q2_K-Q8_K), int8
- **RoPE scaling**: base, NTK, YaRN

### Build Process

- `scripts/build_metal.sh` compiles all variants
- Runtime selects best kernel by shape & dtype
- Output: `kernels/{kernel}_{variant}.metallib` combined into `kernels.metallib`

### MLX Primitive Integration Pattern

Custom Metal kernels are integrated using MLX's Primitive API. This ensures proper integration with MLX's compute graph and memory management.

**Implementation Structure:**

1. **Metal Shader** (`core/kernels/metal/*.metal`)
   - GPU implementation with threadgroup memory
   - Multiple variants for different shapes/types
   - Optimized for Apple Silicon unified memory

2. **MLX Primitive Class** (`core/kernels/primitives/*_primitive.{h,mm}`)
   - Inherits from `mlx::core::Primitive`
   - Implements `eval_gpu()` and `eval_cpu()` methods
   - Manages Metal pipeline states and buffer bindings
   - Handles both contiguous and non-contiguous inputs

3. **High-level Wrapper** (in `core/graph/layers.{h,cpp}`)
   - Clean C++ API for the primitive
   - Integrated into model architecture
   - Follows MLX computation graph patterns

**Example - RMSNorm Implementation:**

- Metal shader: [core/kernels/metal/rmsnorm.metal](core/kernels/metal/rmsnorm.metal)
- Primitive class: [core/kernels/primitives/rmsnorm_primitive.{h,mm}](core/kernels/primitives/rmsnorm_primitive.h)
- Integration: Used in `TransformerBlock` layers
- Status: ✅ Complete with 81/81 tests passing

This pattern ensures:

- Efficient GPU execution via Metal
- Proper MLX graph integration
- Automatic differentiation support (when needed)
- Graceful fallback to CPU when required

## API Surface

### OpenAI-Compatible

- `POST /v1/chat/completions` (SSE streaming)
- `POST /v1/completions`
- `POST /v1/embeddings`

### Ollama-Compatible

- `POST /api/generate`
- `POST /api/chat`
- `POST /api/embeddings`
- Model management: `/api/pull`, `/api/create`, `/api/tags`, `/api/ps`

### Transport

- Primary: Unix Domain Socket at `~/Library/Application Support/MLXRunner/run/mlxrunner.sock`
- Optional HTTP localhost server (disabled by default)
- SSE for token streaming
- Auth via capability token stored in Keychain

## Model Support

### Formats

- **GGUF/GGML**: Parse tensor shards, quant metadata, tokenizer assets
- **HF safetensors**: Streaming load with memory mapping; optional conversion to MLX on first run
- **MLX native**: Load via MLX's array API

### Quantization

- GGUF K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- IQ variants
- FP8/NF4 per-channel/groupwise
- Dynamic activation quantization

### Adapters

- LoRA/QLoRA/IA3 loaded at runtime
- Multiple stacked adapters supported
- Adapter fusion

## Daemon Status

### Current Implementation (Phase 3 - ~40% Complete)

The background daemon has several components already implemented:

**Scheduler System** ([daemon/scheduler/](daemon/scheduler/))

- Request queue management
- Prefill/decode separation
- Basic batching logic
- Worker thread coordination

**REST API** ([daemon/server/](daemon/server/))

- HTTP server with routing
- SSE streaming for token generation
- Ollama-compatible API endpoints
- Worker thread pool

**Model Management** ([daemon/registry/](daemon/registry/))

- Model registry with metadata
- GGUF file format parser
- Weight loading (mmap support planned)

**Telemetry** ([daemon/telemetry/](daemon/telemetry/))

- Metrics collection
- Performance tracking
- Logging infrastructure

**Status:** Core infrastructure exists but needs integration with Engine and testing.

## Configuration

### Server Config (`configs/server.yaml`)

- UDS path and optional HTTP port
- `max_batch_tokens`: Token budget for batching
- `target_latency_ms`: Adaptive batch size target (e.g., 50-80ms/tok)
- `enable_speculative`: Enable speculative decoding (default: true)
- `draft_model`: Draft model for speculation
- `kv_persistence`: Enable KV cache persistence (default: true)

### Model Configs (`configs/models/*.yaml`)

- Model path/URI (GGUF/HF/MLX)
- Tokenizer type (SentencePiece, HF, tiktoken)
- Max context length
- Quantization settings
- RoPE scaling parameters
- Chat template

## Data & Registry

### SQLite Schema

- `models`: id, name, family, format, path, dtype, quant, params, n_ctx, n_layer, vocab_size, created_at
- `adapters`: id, model_id, type, path, scale, rank
- `tags`: id, ref_type, ref_id, key, value
- `cache_entries`: id, model_id, prompt_hash, tokens, logits_path, created_at, last_access

### Paths (User Scope)

- Models: `~/Library/Application Support/MLXRunner/models/`
- Cache: `~/Library/Application Support/MLXRunner/cache/`
- Config: `~/Library/Application Support/MLXRunner/server.yaml`
- Daemon socket: `~/Library/Application Support/MLXRunner/run/mlxrunner.sock`
- Logs: `~/Library/Logs/mlxrunnerd.{out,err}.log`

## Frontend (React WebView)

### Tech Stack

- React + TypeScript (Vite build)
- TailwindCSS + shadcn/ui components
- Zustand state management
- TanStack Query for API data
- Recharts for metrics visualization

### Pages

1. **Chat**: SSE streaming, tool calls, vision attachments
2. **Models**: Registry table, pull/import/convert/quantize actions
3. **Playgrounds**: Embeddings, completion, vision testing
4. **Metrics**: Latency histograms, KV usage, GPU/CPU kernel timing
5. **Settings**: Server config editor, paths, privacy, updates
6. **Logs**: Structured log viewer with search

### IPC Bridge (`window.__HOST__`)

- `request(path, init)`: Proxied fetch to daemon UDS
- `openPathDialog(kind)`: File picker for models/cache
- `readConfig() / writeConfig(yaml)`: Config management
- `startDaemon() / stopDaemon()`: Daemon lifecycle
- `getVersion()`: App and daemon versions

## Build & Development

### Toolchain Requirements

- Xcode (latest stable) for Swift/ObjC compilation
- CMake for C++ core and runtime
- Metal compiler (`xcrun metal`)
- Node.js + Yarn for React frontend
- Python 3.11+ with MLX for model authoring/prototyping

### Build Targets

- **metallib**: Compiled Metal kernels (`scripts/build_metal.sh`)
- **libmlxr_core.a**: C++ engine + MLX glue (CMake)
- **mlxrunnerd**: Background daemon binary
- **MLXR.app**: macOS app bundle with WebView
- **SDKs**: Python wheel, npm package, SwiftPM package

### Build Commands

Use the Makefile for all build operations:

```bash
# Core builds
make metal              # Compile Metal shaders only
make cmake              # Configure CMake
make build              # Full build (metal + cmake + core)
make clean              # Clean build artifacts

# Development
make status             # Check environment and build status
make dev                # Quick dev setup (Metal only, Phase 0-1)
make install-dev        # Install with dev dependencies

# Testing
make test-cpp           # Run C++ unit tests
make test-cpp-verbose   # Verbose test output with colors
make test-phase0        # Validate Phase 0 setup
make validate           # Quick validation (Phase 0 + C++ tests)
make test               # Run Python tests (pytest)
make test-all           # Run all tests (C++ and Python)

# Component builds
make build:core         # Build only core library
make build:daemon       # Build daemon components

# Daemon development
make mlxr_daemon        # Build daemon binary
make test_daemon        # Build and run daemon tests

# Code quality
make format             # Format code (black, clang-format)
make lint               # Lint code (ruff, mypy)
```

See [Makefile](Makefile) for complete list of available commands.

### Example Programs

Four working examples are available in [examples/](examples/):

1. **simple_generation.cpp** - Basic text generation
   - Demonstrates model loading, tokenization, and sampling
   - Single-sequence inference with configurable parameters

2. **kv_cache_test.cpp** - KV cache validation
   - Tests paged KV cache with prefill/decode separation
   - Measures latency and throughput metrics
   - Validates GQA implementation

3. **metal_kernel_test.cpp** - Metal kernel testing
   - Tests custom RMSNorm and attention kernels
   - Validates MLX Primitive integration

4. **cached_model_test.cpp** - Zero-copy optimization verification
   - Tests CachedLlamaModel with Metal attention kernels
   - Verifies zero-copy block format
   - Measures performance improvements

**Usage:**

```bash
# Basic generation
./build/cmake/bin/simple_generation \
    ./models/TinyLlama-1.1B \
    ./models/tokenizer.model \
    "Write a haiku about machine learning"

# KV cache test
./build/cmake/bin/kv_cache_test \
    ./models/TinyLlama-1.1B \
    ./models/tokenizer.model

# Cached model test (zero-copy optimization)
./build/cmake/bin/cached_model_test \
    ./models/TinyLlama-1.1B \
    ./models/tokenizer.model

# Model conversion examples (future)
python tools/convert_hf_to_mlx.py --input path/to/hf --output path/to/mlx
python tools/convert_to_gguf.py --input path/to/hf --output path/to/gguf --quant q4_k
```

## Packaging & Distribution

### Artifacts

- **MLXR.app**: GUI bundle (React WebView + Swift/ObjC host)
- **mlxrunnerd**: Background daemon (launchd agent)
- **mlx**: CLI shim for model management and API calls
- Distribution formats: `.dmg` (drag-and-drop), `.pkg` (installer), `.zip` (portable)

### Code Signing & Notarization

- Sign with Developer ID Application certificate
- Enable Hardened Runtime
- Notarize via `notarytool` and staple
- Sandbox entitlements: file access (user-selected), network client/server

### Auto-Updates

- Sparkle framework with EdDSA signing
- Appcast over HTTPS with delta updates
- Daemon updates orchestrated by app

### Launchd Agent

- Plist location: `~/Library/LaunchAgents/com.company.mlxrunnerd.plist`
- RunAtLoad and KeepAlive enabled
- Logs to `~/Library/Logs/mlxrunnerd.{out,err}.log`

## Performance Targets (M4)

### Current Measured (Phase 2 - TinyLlama 1.1B)

- Prefill: **198-459 ms** (5-10 tokens)
- Decode: **53-220 ms/token** (varies by implementation)
- Throughput: **4.5-18.9 tokens/sec**
- Memory: 87.5% reduction with GQA (308 MB saved on 22-layer model)

### Target Latency (Phase 2 Complete)

- First token: < 1s for 7B-8B models at 4-bit
- Decode: < 80ms/token steady-state
- Embeddings: < 20ms/sample
- Prefill bandwidth: ≥ 1.3× decode throughput

### Target Occupancy

- Attention kernels: ≥ 60% occupancy at D≤128; ≥ 50% at D≥192
- Decode kernel budget: < 0.6ms/head for 7B models (D=128, block_tokens=32)

### Memory Optimization

- Maximize GPU utilization
- Page-aligned KV blocks with working-set-aware LRU eviction
- Memory-mapped weights to minimize copies
- GQA support provides 87.5% KV cache memory reduction for compatible models

## Development Milestones

### M0 – Skeleton

- Repo layout, CMake, Metal toolchain, MLX integration
- Minimal REST server
- Run FP16 llama-style model single-request

### M1 – Batching & KV

- Continuous batching, paged KV arena with eviction
- SSE streaming
- SQLite registry
- GGUF loader

### M2 – Quant & Kernels

- Q-dequant matmul, fused attention/rope/norm kernels
- Latency target: < 80ms/token on 7B at 4-bit

### M3 – Speculative & Persistence

- Draft model verify path
- KV/logits persistence (default on)
- Acceptance rate auto-tuning

### M4 – APIs & GUI

- OpenAI & Ollama shims
- React tray/dock app
- Metrics dashboard

### M5 – Adapters & Vision (optional)

- LoRA stacking
- CLIP/ViT encoder path
- Image-chat template

### M6 – Polish & Release

- Sandboxing, Sparkle updates
- Code signing and notarization
- Documentation and SDKs

## Testing

### Running Tests

```bash
# C++ tests (requires build first)
make test-cpp                    # Run all C++ unit tests
make test-cpp-verbose            # Verbose output with colors
./build/cmake/bin/mlxr_unit_tests --gtest_color=yes  # Direct execution

# Python tests
make test                        # Run pytest

# Phase validation
make test-phase0                 # Phase 0 validation
make validate                    # Quick validation (Phase 0 + C++ tests)
make test-all                    # Run all tests (C++ and Python)
```

### Test Organization

```bash
tests/
  unit/           # C++ unit tests (Google Test)
    tensor_test.cpp
    layers_test.cpp
    rmsnorm_primitive_test.cpp
    mmap_loader_test.cpp
  integration/    # Integration tests (future)
  e2e/            # End-to-end tests (future)
```

### Testing Strategy

#### Functional

- API compatibility tests (OpenAI/Ollama schemas)
- Model zoo smoke tests (Llama, Mistral, Gemma, Qwen)

#### Performance

- Tokens/s (prefill & decode)
- p50/p95 latency
- Peak memory and KV hit rate
- Speculative acceptance rate

#### Stability

- Long-run soak tests (24-72h)
- Memory leak checks
- KV persistence correctness
- Recovery from sleep/wake

#### Frontend

- Unit tests: Vitest + React Testing Library
- Contract tests against mock daemon
- E2E: Playwright (startup, chat session, model pull, update flow)

## Security & Sandboxing

- App and daemon run sandboxed with minimal entitlements
- Model files verified via SHA-256 (optional Ed25519 signature)
- UDS with 0600 permissions
- Capability token auth stored in Keychain
- Telemetry opt-in; no PII collected
- Default bind to UDS only; HTTP port disabled unless toggled

## Important Context

- **GQA Reshape Fix**: A critical MLX lazy evaluation bug was fixed for GQA support (models like TinyLlama with 4 KV heads, 32 Q heads). The fix ensures proper materialization of repeated tensors before reshaping. See [docs/GQA_RESHAPE_FIX.md](docs/GQA_RESHAPE_FIX.md) for details.
- **Speculative decoding** and **KV persistence** are **enabled by default** and tunable in `server.yaml`
- All kernels use **argument buffers** with 16-byte alignment for descriptors
- Weights are **memory-mapped** read-only; KV blocks prefer GPU with CPU overflow
- Quantization groups default to 32-128 along K-dimension
- RoPE tables precomputed for max context length
- Softmax uses two-pass (max, then exp/sum) with fp32 accumulation for numerical stability
- Draft model auto-tunes proposal count k based on acceptance rate monitoring
- GUI communicates with daemon over UDS via JS bridge; SSE for token streaming

## References

See [plan/](plan/) directory for detailed specifications:

- [SPEC01.md](plan/SPEC01.md): Complete requirements and architecture
- [Structure.md](plan/Structure.md): Component view and request lifecycle
- [MetalKernelsPlan.md](plan/MetalKernelsPlan.md): Kernel catalog and variants
- [PackagingDistro.md](plan/PackagingDistro.md): Build, signing, and distribution
- [FrontendPlan.md](plan/FrontendPlan.md): React UI details and IPC bridge
