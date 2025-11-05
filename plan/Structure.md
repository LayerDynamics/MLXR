# MLXR – Project Overview & Architecture

## Project Overview

This document summarizes the repository layout and the runtime architecture of the macOS‑native LLM engine built on MLX + Objective‑C/C++ + Metal, with a React GUI (tray/dock) and OpenAI/Ollama‑compatible APIs.

---

## Folder Structure (Top Level)

```text
MLXR/
  app/                      # macOS app bundle & GUI
    macos/                  # Swift/ObjC host (tray/dock, Sparkle updater)
    ui/                     # React + Vite; built assets embedded in WebView
  daemon/                   # Background server (launch agent)
    server/                 # REST/gRPC, SSE streaming, auth, OpenAI/Ollama shims
    scheduler/              # Prefill/decode queues, batching, rate limiting
    registry/               # SQLite DB, model catalog, mmap loaders, tags
    telemetry/              # Metrics, tracing, profiling exporters
  core/                     # Inference engine (C++/ObjC++ + MLX)
    graph/                  # MLX module definitions (layers, attention, mlp)
    kernels/                # Performance‑critical ops
      metal/                # .metal shaders (fused attention, RoPE, RMSNorm, Q‑gemm)
      cpu/                  # Neon/SIMD fallbacks
    runtime/                # Engine orchestration, tokenizer, KV, sampler, spec
      tokenizer/            # SentencePiece/HF/tiktoken bridges
      kv/                   # Arena, pager, eviction, persistence
      spec/                 # Draft model proposer/verifier
  tools/                    # Converters and quantizers (HF↔GGUF↔MLX)
  sdks/                     # Client SDKs (python, ts, swift)
  configs/                  # Server & model configs (YAML)
  scripts/                  # Build/run helpers (metallib, app bundle, daemon)
  third_party/              # Vendored libs (if any)
  README.md                 # Quick start
  LICENSE                   # License
```

### Notable Files

* `daemon/server/openai_routes.cc` – `/v1/chat/completions`, `/v1/embeddings` (SSE)
* `daemon/server/ollama_routes.cc` – `/api/generate`, `/api/chat`, `/api/embed`, model mgmt
* `core/kernels/metal/attention_fused.metal` – QKV→RoPE→attn→softmax→ctx fused kernel
* `core/runtime/kv/arena.cc` – Paged KV arena; block allocator & page tables
* `configs/server.yaml` – batching, target latency, kv persistence, speculative settings

---

## Runtime Architecture (Component View)

```plantuml
@startuml
skinparam componentStyle rectangle
skinparam dpi 180
skinparam shadowing false

package "macOS App (.app)" {
  [React UI (Tray/Dock)] as UI
  [Swift/ObjC Host] as Host
}

package "Local Daemon" {
  [REST/gRPC Server] as API
  [Scheduler] as Sched
  [Model Registry + Cache] as Registry
  [Telemetry] as Telemetry
}

package "Inference Core" {
  [C++ Engine] as Core
  [MLX Graph] as MLX
  [Tokenizer] as Tok
  [Paged KV Cache] as KV
  [Speculative Decoder] as Spec
}

package "Accelerators" {
  [Metal Kernels] as Metal
  [CPU SIMD] as CPU
  [ANE/CoreML] as ANE
}

UI --> Host : WebView bridge
Host <--> API : Unix domain socket
API --> Sched
API --> Registry
Sched <--> Core
Core <--> MLX
Core <--> Tok
Core <--> KV
Core --> Metal
Core --> CPU
Core --> ANE
Registry --> "SQLite + mmap" : weights, adapters, cache
Telemetry --> UI : live stats
@enduml
```

**Key Principles**

* **Single‑machine, Apple Silicon first**: Maximize GPU via Metal; ANE opportunistic.
* **MLX arrays for model graphs** with escape hatches to custom kernels.
* **Continuous batching + paged KV** for throughput; **speculative decoding** for latency.
* **Unified memory aware**: mmap weights, minimal copies, pinned staging.

---

## Request Lifecycle (Sequence)

```plantuml
@startuml
skinparam dpi 180
actor Client
participant UI
participant API
participant Sched
participant Core
participant KV

Client -> UI : Prompt
UI -> API : POST /v1/chat/completions (SSE)
API -> Sched : Enqueue (prefill/decode)
Sched -> Core : Build/Select batch
Core -> KV : Allocate pages (prefill)
Core -> Core : Fused attention + MLP (Metal)
Core -> Sched : Emit token(s)
Sched -> API : Stream delta (SSE)
API -> UI : Token stream
... loop ...
Core -> KV : Page evict/swap if needed
@enduml
```

---

## Data & Configuration

* **SQLite schema**: models, adapters, tags, cache_entries.
* **Model configs** (`configs/models/*.yaml`): weight URIs (GGUF/HF/MLX), tokenizer, max context, quantization, rope scaling, chat template.
* **Server config** (`configs/server.yaml`): uds path/port, max_batch_tokens, target_latency_ms, enable_speculative, draft_model, kv_persistence.

---

## Build Targets

* **metallib**: compiled attention/rope/norm/Q‑gemm variants per head_dim & tile.
* **libmlxr_core.a**: C++ engine + MLX glue.
* **mlxrd**: daemon (REST/gRPC + scheduler + registry).
* **MLXR.app**: macOS bundle with WebView (React UI) + Host.
* **SDKs**: `pip install mlxrunner`, `npm i @mlx/runner`, SwiftPM package.

---

## Development Workflows

1. **Run daemon**: `./scripts/run_daemon.sh` → serves UDS + HTTP.
2. **GUI**: `yarn dev` in `app/ui` (hot reload) → Host points to dev server; `yarn build` to bundle.
3. **Models**: `tools/convert_hf_to_mlx.py` or `tools/convert_to_gguf.py`; add YAML to `configs/models/`.
4. **Kernels**: edit `.metal` → `scripts/build_metal.sh` → restart daemon.
5. **Perf**: enable profiler via `daemon/telemetry` flags; inspect p95 latency/KV arena utilization.

---

## API Surface (Compatibility)

* **OpenAI**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (SSE streaming).
* **Ollama**: `/api/chat`, `/api/generate`, `/api/embed`, plus model mgmt endpoints.

---

## Notes

* Speculative decoding and KV persistence are **enabled by default** (tunable in `server.yaml`).
* The tray app can quick‑switch models and show live tokens/s, latency, and KV arena graphs.
