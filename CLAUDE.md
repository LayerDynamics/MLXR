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
    server/         # REST + gRPC, SSE streaming, OpenAI/Ollama API shims
    scheduler/      # Prefill/decode queues, continuous batching
    registry/       # SQLite model catalog, mmap loaders
    telemetry/      # Metrics, tracing, profiling
  core/
    graph/          # MLX module definitions (layers, attention, MLP)
    kernels/
      metal/        # .metal shaders (fused attention, RoPE, RMSNorm, quantized matmuls)
      cpu/          # Neon/SIMD fallbacks (⏳ NOT IMPLEMENTED - GPU-only currently)
    runtime/
      tokenizer/    # SentencePiece, HF tokenizers, tiktoken
      kv/           # Arena, pager, eviction, persistence
      spec/         # Speculative decoding (draft model proposer/verifier)
  tools/            # Model converters (HF↔GGUF↔MLX) and quantizers (⏳ PLANNED - Phase 6)
  sdks/             # Client SDKs (Python, TypeScript, Swift)
  configs/          # Server & model configs (YAML) ✅ NEWLY CREATED
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

**Status**: Phase 1 COMPLETE (100%), Phase 2 COMPLETE (~95%), Phase 3 SUBSTANTIAL PROGRESS (~70%)

**Total Actual Codebase**: ~50,000 LOC (core + daemon + app + tests + sdks)

### ✅ Phase 1: Minimal Inference Core (COMPLETE - 100%)

- Complete Llama model with safetensors loading (737 lines in model.cpp)
- SentencePiece tokenizer (252 lines in tokenizer/)
- Sampling strategies (greedy, temperature, top-k, top-p) - 534 lines in sampler.cpp
- Working text generation pipeline
- Example: [simple_generation.cpp](examples/simple_generation.cpp) - ✅ WORKS

### ✅ Phase 2: Optimization (COMPLETE - 95%)

#### KV Cache System - ✅ COMPLETE
- ✅ **Paged KV cache arena** with block allocation and free list management
- ✅ **Page tables** per sequence with copy-on-write support
- ✅ **LRU eviction policy** with working-set awareness
- ✅ **CachedAttention layer** with prefill/decode separation
- ✅ **GQA support** (87.5% memory reduction for compatible models)
- ✅ **Zero-copy block format** for Metal kernels

#### Scheduler-Engine Integration - ✅ COMPLETE
- ✅ **Single-step inference API**: `forward_prefill()` and `forward_decode()` methods
- ✅ **SchedulerWorker** with per-request cache management (not full-generation blocking)
- ✅ **Continuous batching** architecture enabled
- ✅ **Test daemon** (`test_daemon`) running and verified with health endpoints
- See [docs/PHASE2_COMPLETION.md](docs/PHASE2_COMPLETION.md) for architectural details

#### Metal Kernels - ✅ ALL IMPLEMENTED (~5,200 LOC total)
- ✅ **RMSNorm**: Metal shader (217 lines) + primitive (362 lines) - **FULLY INTEGRATED & TESTED** (81/81 tests passing)
- ✅ **Attention Decode**: Metal shader (295 lines) + primitive (574 lines) - ⚠️ Ready, integration pending
- ✅ **Attention Prefill**: Metal shader (370 lines) + primitive (633 lines) - ⚠️ Ready, integration pending
- ✅ **RoPE**: Metal shader (434 lines) + primitive (478 lines) - ⚠️ Ready, integration pending
- ✅ **SwiGLU MLP**: Metal shader (432 lines) + primitive (321 lines) - ⚠️ Ready, integration pending
- ✅ **Q-Gemm Dequant**: Metal shader (486 lines) + primitive (525 lines) - ⚠️ Ready, integration pending
- ✅ **All 6 Core Kernels IMPLEMENTED**: ~2,320 lines Metal + ~2,893 lines Primitives = ~5,200 LOC total

#### Quantization - ⏳ PENDING
- ⏳ **GGUF loading**: Parser exists, loader integration needed
- ⏳ **K-quants support**: Q-Gemm primitive ready, dequantization testing needed

**Critical Integration Gaps:**

⚠️ **Metal Kernel Integration - PRIMARY GAP**
- All 6 kernels exist with shaders and primitives (~5,200 LOC total) ✅
- CachedLlamaModel exists and Engine DOES load it ✅
- CachedAttention layer exists (636 lines) ✅
- **Gap**: CachedAttention doesn't call custom Metal kernels yet - still uses MLX defaults
- **Impact**: Missing 2-5x performance gains from fused attention/RoPE/MLP kernels
- **Next Step**: Wire kernel calls in attention_cached.cpp (8-16 hours estimated)

⚠️ **Daemon Model Loading Integration**
- REST/gRPC endpoints fully implemented ✅
- Scheduler and worker architecture complete ✅
- **Gap**: Model loading → Engine creation → Worker assignment incomplete
- **Impact**: Daemon can't serve inference requests yet
- **Next Step**: Complete load_model() in REST server (4-8 hours estimated)

### ✅ Phase 3: Service Layer (SUBSTANTIAL PROGRESS - ~70% COMPLETE)

The daemon layer has ~9,500 LOC of working code:

- ✅ **Scheduler** (439 lines): Continuous batching, prefill/decode queues, KV block allocation
- ✅ **SchedulerWorker** (241 lines): Background thread with single-step inference execution
- ✅ **REST Server** (1,758 lines): HTTP server with OpenAI & Ollama endpoints
- ✅ **gRPC Server** (1,101 lines): **FULLY IMPLEMENTED** with streaming support
  - Protobuf definitions: `mlxrunner.proto` (395 lines) with complete API surface
  - Server implementation: `grpc_server.{h,cpp}` with all RPC methods
  - OpenAI-compatible streaming: CreateChatCompletion, CreateCompletion
  - Ollama-compatible streaming: Generate, Chat, Embeddings
  - Model management RPCs: Load, Unload, Pull (streaming progress)
  - Health and metrics endpoints
- ✅ **Ollama API** (1,028 lines): Ollama-compatible endpoint implementations
- ✅ **SSE Streaming** (621 lines): Server-sent events for token streaming
- ✅ **Metrics** (769 lines): Metrics collection with 15/15 tests passing
- ✅ **Model Registry** (1,137 lines): SQLite-based model catalog and metadata
- ✅ **GGUF Parser** (891 lines): Complete GGUF format reader
- ✅ **Configuration System**: YAML configuration support
  - ⚠️ `configs/server.yaml`: **MISSING** - needs creation
  - ✅ `configs/models/*.yaml`: 3 example model configurations (TinyLlama, Llama-3, Mistral)
- ✅ **Test Daemon Binary**: Working executable (`test_daemon`) with health/models endpoints verified
- ⏳ **Model Loading Integration** - Endpoints exist, loader wiring needed
- ⏳ **Authentication** - Infrastructure ready, token validation pending

**Test Status:** 14 C++ unit test files in `tests/unit/`

Key test coverage:
- ✅ **RMSNorm primitive**: 81/81 tests passing (fully validated)
- ✅ **Scheduler tests**: Request management and batching (10/12 passing)
- ✅ **Worker tests**: Thread lifecycle and execution (9/9 passing)
- ✅ **REST server tests**: Endpoint routing and responses (15/15 passing)
- ✅ **Metrics tests**: Collection and reporting (15/15 passing)
- ✅ **GQA attention tests**: 6/6 passing (validates critical reshape fix)

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) and [docs/DAEMON_STATUS.md](docs/DAEMON_STATUS.md) for detailed status.

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

**Current Status**: All 6 critical kernels have complete Metal shaders (.metal) and MLX Primitives (.mm). RMSNorm fully tested and integrated; others ready for integration testing.

**Total Metal Kernel Code**: ~5,200 LOC (2,320 in shaders + 2,893 in primitives)

### Implementation Status by Kernel

1. ✅ **rmsnorm_fused** - **COMPLETE & FULLY INTEGRATED**
   - Metal shader: 217 lines ([core/kernels/metal/rmsnorm.metal](core/kernels/metal/rmsnorm.metal))
   - Primitive: 362 lines ([core/kernels/primitives/rmsnorm_primitive.mm](core/kernels/primitives/rmsnorm_primitive.mm))
   - Wrapper: rmsnorm.{h,cpp} provides clean API
   - Integration: Used in TransformerBlock layers ✅
   - Tests: 81/81 passing - fully validated ✅
   - See [docs/PHASE1_METAL_RMSNORM_COMPLETION.md](docs/PHASE1_METAL_RMSNORM_COMPLETION.md)

2. ✅ **attention_decode_fused** - **IMPLEMENTED, INTEGRATION PENDING**
   - Metal shader: 295 lines - Paged KV decode path
   - Primitive: 574 lines - Complete MLX Primitive implementation
   - Features: GQA support, sliding window, numerically stable softmax
   - Status: Ready for CachedAttention integration (needs wiring)

3. ✅ **attention_prefill_fused** - **IMPLEMENTED, INTEGRATION PENDING**
   - Metal shader: 370 lines - Fused prefill with KV storage
   - Primitive: 633 lines - Complete MLX Primitive implementation
   - Features: RoPE fusion, causal masking, GQA support
   - Status: Ready for CachedAttention integration (needs wiring)

4. ✅ **rope_apply** - **IMPLEMENTED, INTEGRATION PENDING**
   - Metal shader: 434 lines - Standalone RoPE kernel
   - Primitive: 478 lines - Complete MLX Primitive implementation
   - Features: Base, NTK-scaled, YaRN-scaled RoPE variants
   - Status: Ready for integration (can be standalone or fused)

5. ✅ **swiglu_mlp_fused** - **IMPLEMENTED, INTEGRATION PENDING**
   - Metal shader: 432 lines - Gated MLP fusion
   - Primitive: 321 lines - Complete MLX Primitive implementation
   - Features: Optional quantized weights support
   - Status: Ready for MLP layer integration

6. ✅ **q_gemm_dequant** - **IMPLEMENTED, INTEGRATION PENDING**
   - Metal shader: 486 lines - Quantized matmul with dequant
   - Primitive: 525 lines - Complete MLX Primitive implementation
   - Features: K-quants (Q2_K-Q8_K), on-the-fly dequantization
   - Status: Ready for GGUF quantization support

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

### Current Implementation (Phase 3 - ~70% Complete)

The background daemon is substantially implemented with ~9,500 LOC of working code:

**✅ Scheduler System** ([daemon/scheduler/](daemon/scheduler/))
- Complete continuous batching implementation (439 lines)
- Prefill and decode queue separation with prioritization
- KV block allocation and preemption policies
- Request state machine (WAITING → PREFILLING → DECODING → COMPLETED)
- Token budget constraints and batch formation
- Tests: 10/12 passing ✅

**✅ Scheduler Worker** ([daemon/server/scheduler_worker.{h,cpp}](daemon/server/scheduler_worker.h))
- Background thread implementation (241 lines)
- **Critical feature**: Single-step inference (not full-generation blocking)
- Per-request cache management with automatic cleanup
- Token callback integration for SSE streaming
- Graceful shutdown handling
- Tests: 9/9 passing ✅

**✅ REST API Server** ([daemon/server/rest_server.{h,cpp}](daemon/server/rest_server.h))
- Full HTTP server implementation (1,758 lines)
- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- SSE streaming support for real-time token generation
- CORS and API key authentication infrastructure
- Verified working with health checks
- Tests: 15/15 passing ✅

**✅ gRPC API Server** ([daemon/server/grpc_server.{h,cpp}](daemon/server/grpc_server.h))
- **FULLY IMPLEMENTED** (1,101 lines) - contrary to some outdated docs
- Protobuf definitions complete (395 lines in mlxrunner.proto)
- All RPC methods implemented: CreateChatCompletion, Generate, Chat, Embeddings
- Model management: LoadModel, UnloadModel, PullModel (streaming)
- Health and metrics endpoints
- Streaming support for all generation methods
- Tests: gRPC server tests passing ✅

**✅ Ollama API** ([daemon/server/ollama_api.{h,cpp}](daemon/server/ollama_api.h))
- Ollama-compatible endpoint implementations (1,028 lines)
- Model management: `/api/pull`, `/api/create`, `/api/tags`, `/api/ps`
- Chat and generation endpoints with streaming

**✅ SSE Streaming** ([daemon/server/sse_stream.{h,cpp}](daemon/server/sse_stream.h))
- Server-sent events implementation (621 lines)
- Token-by-token streaming with completion signals
- Error handling and connection management

**✅ Model Management** ([daemon/registry/](daemon/registry/))
- SQLite-based model registry (1,137 lines)
- Complete GGUF file format parser (891 lines)
- Model metadata and catalog management
- Weight loading infrastructure (mmap support ready)

**✅ Telemetry** ([daemon/telemetry/metrics.{h,cpp}](daemon/telemetry/metrics.h))
- Metrics collection implementation (769 lines)
- Request throughput, latency, and KV cache utilization tracking
- Prometheus-style metrics export ready
- Tests: 15/15 passing ✅

**✅ Test Daemon Binary** (`daemon/test_daemon_main.cpp`)
- Integrated executable that runs and responds to health checks
- Successfully starts and listens on port 11434
- Health endpoint verified: `GET /health` → `{"status":"ok"}`
- Models endpoint verified: `GET /v1/models` → returns empty list
- Graceful shutdown with SIGINT/SIGTERM handling

**Integration Status:**
- ✅ Scheduler ↔ Worker ↔ REST/gRPC Server: Fully wired
- ⏳ Worker ↔ Engine: Single-step API working, needs model loading wiring (4-8 hours)
- ⏳ Registry ↔ Engine: Model loading integration pending

See [docs/DAEMON_STATUS.md](docs/DAEMON_STATUS.md) and [docs/PHASE2_COMPLETION.md](docs/PHASE2_COMPLETION.md) for details.

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

**Status**: ✅ COMPLETE (100%) - 78 React components implemented

### Tech Stack

- React 18.3 + TypeScript (Vite build)
- TailwindCSS + shadcn/ui components
- Zustand state management
- TanStack Query for API data
- Recharts for metrics visualization
- Playwright for E2E testing

### Component Structure

The UI is fully implemented in `app/ui/src/` with 78 TypeScript React components organized by category:

- **Chat** (10 components): Message, MessageList, Composer, ChatPane, ConversationList, ModelSelector, SamplingControls, AttachmentButton, ToolCallView, TokenStream
- **Models** (7 components): RegistryTable, ModelCard, ModelImport, ModelPullDialog, ModelDetailDrawer, ModelStats, ModelActions
- **Settings** (10 components): Panels for General, Performance, Paths, Updates, Privacy, plus SettingRow, PathPicker, ConfigEditor, DaemonControl, KeyboardShortcuts
- **Metrics** (8 components): LiveMetrics, ThroughputChart, KVChart, LatencyChart, KernelTimeChart, MetricsCard, StatsCard, MetricsFilter
- **Logs** (2 components): LogViewer with TanStack Virtual, LogEntry with expandable context
- **Playground** (3 components): CompletionPlayground, EmbeddingsPlayground, VisionPlayground
- **Layout** (3 components): Navigation with tabs, CommandPalette (⌘K), TrayPopover
- **UI** (35+ components): shadcn/ui primitives (Button, Dialog, Input, Select, etc.)

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

See [app/ui/COMPONENTS.md](app/ui/COMPONENTS.md) for complete component documentation.

## Build & Development

### Toolchain Requirements

**System Tools:**
- Xcode (latest stable) for Swift/ObjC compilation
- CMake 3.20+ for C++ core and runtime
- Ninja build system
- Metal compiler (`xcrun metal`)
- Homebrew package manager

**Homebrew Dependencies:**
```bash
# Install system dependencies
brew install cmake ninja mlx sentencepiece nlohmann-json cpp-httplib googletest

# Or use the Makefile target
make install-deps
```

**IMPORTANT:** CMake and Ninja must be installed via Homebrew. The conda-forge "cmake" package
is NOT the CMake build system and will cause build failures.

Required packages:
- `cmake` - CMake build system (version 3.x)
- `ninja` - Fast build tool
- `mlx` - Apple's machine learning framework
- `sentencepiece` - Tokenization library (required by core/runtime/tokenizer)
- `nlohmann-json` - JSON library for C++ (required by daemon)
- `cpp-httplib` - HTTP server library (required by daemon/server)
- `googletest` - C++ testing framework (required by tests)

**Frontend:**
- Node.js 18+ + Yarn for React frontend

**Python:**
- Python 3.11+ with MLX for model authoring/prototyping
- Conda or virtualenv recommended (see `environment.yml`)

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

### Critical Architectural Fixes

- **Scheduler-Engine Integration Fix**: The original implementation had SchedulerWorker calling `engine->generate_tokens()` which runs a full autoregressive loop, blocking the thread until completion. This defeated continuous batching. **Fixed** by implementing single-step inference with `forward_prefill()` and `forward_decode()` methods that return after ONE token, allowing proper request interleaving. See [docs/PHASE2_COMPLETION.md](docs/PHASE2_COMPLETION.md) for details.

- **GQA Reshape Fix**: A critical MLX lazy evaluation bug was fixed for GQA support (models like TinyLlama with 4 KV heads, 32 Q heads). The fix ensures proper materialization of repeated tensors before reshaping. GQA provides 87.5% KV cache memory reduction for compatible models. See [docs/GQA_RESHAPE_FIX.md](docs/GQA_RESHAPE_FIX.md) for details.

- **CachedLlamaModel Integration Gap**: CachedLlamaModel with zero-copy paged KV cache exists but Engine currently uses simple LlamaModel. Metal attention kernels cannot be utilized until this integration is complete. See [docs/SESSION_2025_11_06_INTEGRATION.md](docs/SESSION_2025_11_06_INTEGRATION.md) for integration plan.

### Design Decisions

- **Speculative decoding** and **KV persistence** are **enabled by default** and tunable in `server.yaml`
- All kernels use **argument buffers** with 16-byte alignment for descriptors
- Weights are **memory-mapped** read-only; KV blocks prefer GPU with CPU overflow
- Quantization groups default to 32-128 along K-dimension
- RoPE tables precomputed for max context length
- Softmax uses two-pass (max, then exp/sum) with fp32 accumulation for numerical stability
- Draft model auto-tunes proposal count k based on acceptance rate monitoring
- GUI communicates with daemon over UDS via JS bridge; SSE for token streaming

## What Still Needs Implementation

Based on the comprehensive analysis of plan files vs actual codebase, here's the master implementation roadmap:

### 🔴 P0 - Critical Blockers (Blocks End-to-End Usage)

#### 1. Metal Kernel Integration (8-16 hours) ⚠️ **PRIMARY BLOCKER**

**Problem**: All 6 Metal kernels exist (~5,200 LOC) but only RMSNorm is integrated. CachedAttention layer doesn't call custom kernels yet.

**Required Work**:
- Wire `attention_decode_primitive` call in `attention_cached.cpp:forward_decode()`
- Wire `attention_prefill_primitive` call in `attention_cached.cpp:forward_prefill()`
- Add kernel dispatch logic based on head_dim and dtype
- Test with TinyLlama model
- Measure performance improvement (expect 2-5x speedup)

**Files to Modify**:
- `core/graph/attention_cached.cpp` - Add kernel calls
- `core/graph/attention_cached.h` - Add kernel headers
- Add integration tests

**Success Criteria**: Inference uses custom kernels; 2-5x performance gain measured

#### 2. Daemon Model Loading Integration (4-8 hours)

**Problem**: REST/gRPC endpoints exist, scheduler ready, but model loading → engine → worker assignment incomplete.

**Required Work**:
- Implement `load_model()` in REST server to:
  - Load weights via mmap_loader
  - Create CachedLlamaModel instance
  - Create Engine with model
  - Assign to SchedulerWorker
- Wire up model registry queries
- Add error handling for model not found
- Test complete request flow

**Files to Modify**:
- `daemon/server/rest_server.cpp` - Complete load_model()
- `daemon/server/grpc_server.cpp` - Complete LoadModel RPC
- `daemon/server/scheduler_worker.cpp` - Model assignment logic

**Success Criteria**: Full end-to-end inference works: curl → daemon → model → tokens

#### 3. Server Configuration File (2-4 hours)

**Problem**: `configs/server.yaml` doesn't exist; daemon has no config file.

**Required Work**:
- Create default `configs/server.yaml` with all settings:
  ```yaml
  server:
    uds_path: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock
    http_port: null  # disabled by default
    max_batch_tokens: 2048
    target_latency_ms: 80

  models:
    default_model: TinyLlama-1.1B
    models_dir: ~/Library/Application Support/MLXRunner/models/

  kv_cache:
    enable_persistence: true
    block_size: 32
    max_blocks: 8192

  speculative:
    enable_speculative: true
    draft_model: null  # auto-detect
    speculation_length: 4
  ```
- Load config in daemon startup
- Add config validation
- Document all settings

**Files to Create**:
- `configs/server.yaml` - Default configuration
- `daemon/config/` - Config loading module (if doesn't exist)

**Success Criteria**: Daemon starts with config; settings applied correctly

---

### 🟠 P1 - High Priority (Improves Performance & Features)

#### 4. Quantization Integration (8-12 hours)

**Problem**: GGUF parser exists, Q-gemm kernel ready, but quantized model loading incomplete.

**Required Work**:
- Integrate GGUF parser with model loader
- Wire `q_gemm_dequant_primitive` calls in Linear layers
- Add weight dtype detection and dispatch
- Test with Q4_K quantized model
- Verify accuracy vs FP16

**Files to Modify**:
- `core/runtime/mmap_loader.cpp` - GGUF weight loading
- `core/graph/layers.cpp` - Q-gemm dispatch in Linear
- Add quantization tests

**Success Criteria**: Can load and run Q4_K models with <2% accuracy loss

#### 5. RoPE and SwiGLU Kernel Integration (6-10 hours)

**Problem**: rope_apply and swiglu_mlp_fused kernels exist but not used.

**Required Work**:
- Wire `rope_apply_primitive` in Attention layer
- Wire `swiglu_mlp_fused_primitive` in MLP layer
- Add kernel variant selection
- Test and measure speedup

**Files to Modify**:
- `core/graph/attention_cached.cpp` - RoPE kernel calls
- `core/graph/layers.cpp` - SwiGLU kernel calls

**Success Criteria**: Additional 10-30% performance improvement measured

#### 6. Speculative Decoding Wiring (6-10 hours)

**Problem**: Spec decoder infrastructure exists (`core/runtime/spec/` 581 lines) but not wired.

**Required Work**:
- Connect draft model to scheduler
- Implement verification loop in decode
- Add acceptance rate tracking
- Auto-tune speculation length (k) based on acceptance rate
- Add config option to enable/disable

**Files to Modify**:
- `core/runtime/engine.cpp` - Spec decode integration
- `daemon/scheduler/scheduler.cpp` - Draft model support
- `configs/server.yaml` - Add spec settings

**Success Criteria**: 1.5-2x latency reduction on supported models

---

### 🟡 P2 - Medium Priority (Polish & Completeness)

#### 7. macOS App Bundle Creation (8-16 hours)

**Problem**: Swift files exist, but no .app bundle build, no code signing, no .dmg.

**Required Work**:
- Create Xcode build script for .app bundle
- Embed React UI dist/ in Resources/
- Embed daemon binary in bundle
- Code sign with Developer ID
- Create .dmg installer
- Add Sparkle auto-update integration

**Files to Create/Modify**:
- `scripts/build_app_bundle.sh` - Complete app build
- `scripts/sign_and_notarize.sh` - Code signing
- `scripts/create_dmg.sh` - DMG creation
- `app/macos/UpdateManager.swift` - Finish Sparkle integration

**Success Criteria**: Distributable MLXR.app that launches and auto-updates

#### 8. CPU Fallback Kernels (16-24 hours)

**Problem**: Plan specified Neon/SIMD fallbacks; only `.gitkeep` exists in `core/kernels/cpu/`.

**Required Work**:
- Implement Neon SIMD versions of:
  - RMSNorm
  - RoPE
  - Attention (simple version)
  - SwiGLU
- Add CPU/GPU dispatch logic
- Fallback when GPU unavailable or shapes unsupported

**Files to Create**:
- `core/kernels/cpu/rmsnorm_neon.cpp`
- `core/kernels/cpu/rope_neon.cpp`
- `core/kernels/cpu/attention_neon.cpp`
- `core/kernels/cpu/swiglu_neon.cpp`

**Success Criteria**: System runs on CPU when GPU unavailable (albeit slower)

#### 9. Model Conversion Tools (12-20 hours)

**Problem**: Only basic HF→MLX converter exists; missing GGUF→MLX and quantizers.

**Required Work**:
- Create `tools/convert_gguf_to_mlx.py`
- Create `tools/quantize_model.py` with Q2_K through Q8_K support
- Create `tools/merge_adapters.py` for LoRA merging
- Add calibration dataset support for quantization
- CLI interface for all tools

**Files to Create**:
- `tools/convert_gguf_to_mlx.py`
- `tools/quantize_model.py`
- `tools/merge_adapters.py`

**Success Criteria**: Can convert and quantize any HF/GGUF model

---

### 🟢 P3 - Low Priority (Future Features)

#### 10. Vision Support (40+ hours)

**Problem**: Plan includes LLaVA/CLIP support; not implemented.

**Required Work**:
- Implement CLIP encoder
- Add image preprocessing pipeline
- Create `clip_patchify_proj` Metal kernel
- Integrate with chat API
- Add vision model configs

**Files to Create**:
- `core/graph/vision/` - Vision encoder modules
- `core/kernels/metal/clip_patchify.metal`
- Vision tests and examples

**Success Criteria**: Can run LLaVA-style image+text chat

#### 11. Advanced Features (80+ hours)

- **Multi-model residency**: Load multiple models, hot-swap
- **LoRA adapter loading**: Runtime adapter application
- **Prompt caching**: Hash prompts, cache prefill results
- **Logits caching**: Disk-backed logits for common queries
- **Model pull from Hugging Face**: Download models directly
- **Model conversion on import**: Auto-convert to optimal format

---

## Implementation Priority Summary

**Week 1-2 (P0 - Get it Working)**:
1. Metal kernel integration (8-16h) ← **Start here**
2. Daemon model loading (4-8h)
3. Server config file (2-4h)

**Week 3-4 (P1 - Make it Fast)**:
4. Quantization integration (8-12h)
5. RoPE/SwiGLU kernels (6-10h)
6. Speculative decoding (6-10h)

**Week 5-8 (P2 - Polish & Ship)**:
7. App bundle creation (8-16h)
8. CPU fallback kernels (16-24h)
9. Model conversion tools (12-20h)

**Future (P3 - Advanced Features)**:
10. Vision support (40h)
11. Advanced features (80h)

**Total Estimated Effort**: 210-340 hours (6-10 weeks full-time)

---

## References

### Planning Documents

See [plan/](plan/) directory for detailed specifications:

- [SPEC01.md](plan/SPEC01.md): Complete requirements and architecture
- [Structure.md](plan/Structure.md): Component view and request lifecycle
- [MetalKernelsPlan.md](plan/MetalKernelsPlan.md): Kernel catalog and variants
- [PackagingDistro.md](plan/PackagingDistro.md): Build, signing, and distribution
- [FrontendPlan.md](plan/FrontendPlan.md): React UI details and IPC bridge

### Implementation Status Documents

See [docs/](docs/) directory for current implementation details:

- [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md): Overall implementation status and metrics
- [DAEMON_STATUS.md](docs/DAEMON_STATUS.md): Daemon components detailed status
- [PHASE2_COMPLETION.md](docs/PHASE2_COMPLETION.md): Scheduler-engine integration and architectural fix
- [SESSION_2025_11_06_INTEGRATION.md](docs/SESSION_2025_11_06_INTEGRATION.md): CachedLlamaModel integration plan
- [PHASE1_METAL_RMSNORM_COMPLETION.md](docs/PHASE1_METAL_RMSNORM_COMPLETION.md): RMSNorm kernel implementation
- [GQA_RESHAPE_FIX.md](docs/GQA_RESHAPE_FIX.md): Critical GQA support fix for MLX
- [KV_CACHE_IMPLEMENTATION.md](docs/KV_CACHE_IMPLEMENTATION.md): Paged KV cache architecture
