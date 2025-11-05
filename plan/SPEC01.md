# SPEC-1-MLX-Llama Runner for Apple Silicon (M4)

## Background

You want a high‑performance, local LLM engine/model runner—built natively for macOS on Apple Silicon (M4, unified memory)—that matches the feature breadth users expect from vLLM, llama.cpp, and Ollama while leveraging Apple‑first technologies:

* **Core tech stack**: MLX (Apple’s machine learning array framework), Objective‑C/Swift/CPP host runtime, Metal compute kernels/shaders (including MPS where appropriate) tuned for M4 GPU/ANE, with careful unified-memory orchestration.
* **Primary intent**: Serve transformer‑based LLMs and other ML models (vision encoders, embedding models) with low latency on a single MacBook Pro, supporting chat, batch inference, embeddings, quantized weights, and plugin/tool use.
* **Why now**: Apple’s MLX provides NumPy‑like arrays directly on device with lazy execution and unified memory friendliness, while M4 offers improved GPU/ANE throughput. A bespoke runner can combine MLX ease with custom Metal kernels for hot paths (KV‑cache ops, attention, RMSNorm, quantized matmuls) to outperform general frameworks on macOS and expose a developer‑friendly API akin to vLLM/Ollama.
* **Differentiator**: A cohesive, macOS‑native engine with:

  * MLX-first graph & tensor management but escape hatches to hand‑rolled Metal for critical kernels.
  * Objective‑C/Swift host for macOS integration (sandboxing, app packaging) and C++ for portable core logic.
  * Tight KV‑cache paging across unified memory; smart CPU/GPU/ANE placement; and fast weight loading (memory‑mapped, quant‑aware).
  * Runtime parity with popular runners (model registry, REST/gRPC serving, tokenizer support, multi‑model residency, adapters/LoRA, chat templates).

---

## Requirements

**Assumptions based on your inputs**

* Support "any model in application" → ship with loaders/adapters for GGUF/GGML, safetensors/transformers (HF), and native MLX checkpoints. Include tokenizers (SentencePiece, HF tokenizers, tiktoken).
* "Whatever makes the model faster" → prioritize latency/throughput features even if they add complexity: continuous batching, paged KV cache, quantized matmuls, speculative decoding, flash attention, prefill/decoding split scheduling, model/kv offloading heuristics across CPU/GPU/ANE.

### MoSCoW Prioritization

**Must‑Have**

1. **Model formats**: Load Llama‑family, Mistral/Mixtral, Phi, Gemma, Qwen2, Yi, and generic decoder‑only (causal LM) from **GGUF** and **HF safetensors**; export/import to **MLX** weights.
2. **Quantization**: Run Q2_K–Q8_K (GGUF), NF4/FP8/FP16, per‑channel/groupwise; dynamic activation quant; on‑the‑fly dequant in kernels.
3. **Tokenizer support**: SentencePiece + HF tokenizers; byte‑fallback; special tokens/templates (chat, system, tools).
4. **Performance core**: Continuous batching, **paged KV cache** with eviction, **FlashAttention‑style** fused attention, **RMSNorm/SwiGLU** fused ops, **RoPE** with scaling (NTK/YaRN), **prefill–decode scheduler**, **speculative decoding** (tree/medusa‑style), **tensor fusion**.
5. **Device placement**: Heuristics for CPU vs GPU vs ANE; **unified‑memory aware** paging; weight memory‑mapping; pinned host staging buffers.
6. **APIs**: OpenAI‑compatible REST (/v1/chat/completions, /v1/completions, /v1/embeddings), **Ollama‑compatible** endpoints, and a native REST/gRPC with server‑sent streaming.
7. **CLI + local daemon**: `mlx-run serve/pull/convert/quant` parity; model registry, tags, and caching; offline‑first operation.
8. **Safety & templates**: Built‑in chat templates (llama/mistral/qwen/gemma), stop sequences, system prompts, tool‑use schemas (JSON schema function/tool calling pass‑through).
9. **Observability**: Prometheus metrics, structured logs, per‑request traces, kernel timing, memory/KV cache stats.
10. **Mac integration**: Universal app bundle + codesigning; sandbox‑friendly file access; keychain‑based token secrets for remote pulls (optional).

**Should‑Have**

1. **Adapters**: LoRA/QLoRA/IA3/adapter‑fusion at load time; multi‑adapter stacking.
2. **Multi‑tenant residency**: Hot‑swap multiple models; shared tokenizer and shared KV arena; per‑tenant quotas.
3. **Embeddings + rerankers**: Run text embedding models; light‑weight cross‑encoder reranker.
4. **Vision**: LLaVA/LLama‑Vision style image‑encoder support (CLIP/ViT) with Metal kernels for patchify/projection.
5. **Caching**: Prompt and logits cache; disk‑backed prefill cache with content hashing.
6. **Speculative decoders**: Draft model co‑resident; support lookahead tokens and acceptance policy tuning.
7. **Model conversion tools**: HF → GGUF/MLX, GGUF → MLX; quantizers (GPTQ/AWQ/TEQ) with Metal kernels for calibration.

**Could‑Have**

1. **Distillation utilities**: Small student generation on‑device.
2. **SFT/LoRA finetune** (single‑node) using MLX + Metal‑accelerated optimizers.
3. **Plugins**: Tool execution (local functions, shell‑guarded), retrieval connectors (SQLite/FAISS/LanceDB).

**Won’t‑Have (MVP)**

1. Multi‑node distributed serving; no cross‑machine tensor parallelism.
2. Training large base models.
3. Windows/Linux GPU backends (macOS‑only focus for MVP).

---

## Method

### High-Level Architecture

```plantuml
@startuml
skinparam componentStyle rectangle
skinparam shadowing false
skinparam dpi 180

package "macOS App (Bundle)" {
  [React Frontend (Tray/Dock GUI)] as UI
  [Native Host (Swift/ObjC)] as Host
}

package "Local Daemon" {
  [REST/gRPC Server]
as API
  [Scheduler]
as Sched
  [Model Registry/Cache]
as Registry
  [Telemetry + Metrics]
as Telemetry
}

package "Inference Core" {
  [C++ Runtime]
as Core
  [MLX Graph]
as MLX
  [Tokenizer (SPM/HF/tiktoken)] as Tok
  [Paged KV Cache]
as KV
  [Speculative Decoding Engine]
as Spec
}

package "Accelerators" {
  [Metal Shaders (Attention/Matmul/Norm)] as Metal
  [CPU Neon Kernels]
as CPU
  [ANE (optional)] as ANE
}

UI --> Host : WebView + JS bridge
Host <--> API : localhost unix domain socket
API --> Sched
API --> Registry
Sched <--> Core
Core <--> MLX
Core <--> Tok
Core <--> KV
Core --> Metal
Core --> CPU
Core --> ANE
Registry --> Disk : SQLite + mmap weights
Telemetry --> UI : live stats
@enduml
```

**Key choices**

* **Frontend**: React (tray/dock) packaged in the .app bundle using a lightweight WebView; communicates with the daemon via a localhost Unix domain socket for low overhead.
* **Backends**: Python (for MLX model authoring and ops prototyping) coexists with ObjC++/C++ runtime; hot paths compiled to Metal compute kernels and optionally ANE via Core ML when appropriate.
* **Graph**: MLX arrays for model layers with lazy evaluation; escape hatches to custom Metal for fused attention, RoPE, RMSNorm, and quantized matmuls.
* **APIs**: OpenAI‑compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) and Ollama‑compatible (`/api/generate`, `/api/chat`, `/api/embeddings`, model mgmt), plus a native REST/gRPC with streaming.

---

### Data & Model Registry

* **SQLite** database for:

  * `models(id, name, family, format, path, dtype, quant, params, n_ctx, n_layer, vocab_size, created_at)`
  * `adapters(id, model_id, type, path, scale, rank)`
  * `tags(id, ref_type, ref_id, key, value)`
  * `cache_entries(id, model_id, prompt_hash, tokens, logits_path, created_at, last_access)`
* **Weights** stored on disk; memory‑mapped (mmap) into unified memory with page‑aligned offsets matching KV block size.

---

### Memory & Performance Strategy (M4 Unified Memory)

1. **Paged KV Cache**: Global KV arena split into fixed‑size blocks (e.g., 16–64 tokens per block). Maintain free lists and a page table per sequence; evict with working‑set‑aware LRU when pressure exceeds threshold.
2. **Continuous Batching**: Merge requests at token boundaries; split prefill (GPU‑bound) and decode (latency‑sensitive) queues. Dynamically adjust max batch by observing kernel tail latency.
3. **Kernel Fusion**: Custom Metal kernels to fuse QKV projection → RoPE → attention score → softmax → context matmul (FlashAttention‑style), RMSNorm+SwiGLU, and dequant+matmul for quantized weights.
4. **Quantization**: Support GGUF K‑quants (Q2_K…Q8_K), IQ variants, and FP8/NF4. Provide dequant shaders using vectorized loads and shared‑memory tiles; fall back to MLX FP16 on shape edge cases.
5. **Placement Heuristics**: Weights on GPU; KV blocks on GPU with overflow to CPU pinned memory; prefill ops on GPU; light samplers on CPU; opportunistic ANE for activation functions and small convolutions in VLMs.
6. **Unified Memory Tips**: Prefer read‑only mappings for weights; avoid cross‑device thrash by staging copies for frequently reused tiles; coalesce I/O during load using parallel `mmap+mlock`.

---

### Execution Pipeline

1. **Load** → parse model (GGUF/HF/MLX), build MLX graph, compile or select Metal kernels by head dimension and block size.
2. **Prefill** → tokenize, allocate KV pages, run fused prefill kernel; capture `attention_mask`/`pos` in compact form.
3. **Decode loop** → continuous batch; run fused attention + MLP; **speculative decoding** if a draft model is present; sample (top‑p, nucleus, grammar, function‑call JSON schema) and stream tokens.
4. **Evict/Swap** → when KV arena pressure rises, evict tail tokens of least‑recent sequences; optionally persist KV to disk for long‑lived chats (default on).

---

### Speculative Decoding (Draft model)

* Optional draft model (e.g., 1/4 size) proposes `k` tokens; main model verifies using the same paged KV. Acceptance rate monitored to auto‑tune `k`. Toggle in UI (enabled by default).

---

### Metal Kernel Sketches (Hot Paths)

* **Fused Attention**: threadgroup tiles over `(seq_block, head_dim)`; shared memory for K/V tiles; half/FP8 inputs with on‑the‑fly dequant; warp‑level reductions for softmax normalization.
* **RoPE**: precompute cos/sin tables for max context; apply interleaved rotary in registers.
* **RMSNorm**: reduce over hidden size with Kahan compensation; scale/bias in FP16.
* **Dequantized Matmul**: K‑quant unpack → scale → dot; grouped‑quant scales cached in LDS; epilogue adds bias + activation fusion.

---

### API Surface (selected)

* **OpenAI‑compatible**: `POST /v1/chat/completions` (SSE streaming), `POST /v1/completions`, `POST /v1/embeddings`.
* **Ollama‑compatible**: `POST /api/generate`, `/api/chat`, `/api/embeddings`, model mgmt (`/api/pull`, `/api/create`, `/api/tags`, `/api/ps`).

---

### Packaging & Interop

* **App bundle** with launch‑at‑login, tray icon, minimal dock window; auto‑updates via Sparkle.
* **Daemon** runs as a background launch agent; communicates over uds with auth token; optional CORS for local dev.
* **Bindings/SDKs**: TypeScript (fetch wrapper), Python client, Swift package; both Ollama and OpenAI shims for drop‑in compatibility.

---

### Comparison Targets

* vLLM: emulate paged attention and continuous batching.
* llama.cpp: parity on GGUF loading and K‑quant support; CLI ergonomics.
* Ollama: REST surface + model registry behavior.

## Implementation

### Repository Layout

```
mlx-runner/
  app/
    macos/ (Swift/ObjC host, tray, Sparkle updater)
    ui/ (React + Vite, WebView bundle)
  daemon/
    server/ (REST/gRPC, SSE, auth)
    scheduler/ (prefill/decode queues, batching)
    registry/ (SQLite, model store, mmap loader)
    telemetry/ (metrics, tracing)
  core/
    graph/ (MLX module defs, layer registry)
    kernels/metal/
      attention_fused.metal
      rmsnorm.metal
      rope.metal
      q_gemm_dequant.metal
    kernels/cpu/
      rope_neon.cpp
      rmsnorm_neon.cpp
    runtime/
      engine.cc (C++)
      tokenizer/
        spm.cc hf_tokenizers.cc tiktoken.cc
      kv/
        arena.cc pager.cc eviction.cc
      spec/
        draft_model.cc verifier.cc
  tools/
    convert_hf_to_mlx.py
    convert_to_gguf.py
    quantize_fp16_to_fp8.py
  sdks/
    python/
    ts/
    swift/
  configs/
    models/*.yaml
    server.yaml
  scripts/
    build_metal.sh build_app.sh run_daemon.sh
```

### Build & Toolchain

* **Metal**: compile `.metal` shaders into `.metallib` per `head_dim` and tile size; embed in app bundle and daemon.
* **C++/ObjC++**: CMake project generating static libs for core and runtime; link with MLX libs.
* **Python**: optional wheels that expose `mlxrunner` Python API; use pybind11 to call into C++ runtime.
* **React UI**: Vite build → WebView bundle; auto‑update channel via Sparkle.

### Model Loading

* **GGUF**: parse tensor shards, quant metadata, tokenizer assets; map into MLX tensors or pass to custom dequant kernels.
* **HF safetensors**: streaming load with memory mapping; optional conversion to MLX checkpoint on first run.
* **MLX native**: load via MLX’s array API.
* **Adapters**: LoRA/QLoRA/IA3 loaded and merged at runtime; supports multiple stacked adapters.

### Scheduler & Batching

* Separate **prefill** and **decode** queues; global **token budget** guard; adaptive batch size using p95 latency target (e.g., 50–80 ms/tok).
* **Chunked prefill** for very long prompts; interleave with decode to avoid starvation.

### KV Cache & Persistence

* **Arena**: contiguous device buffer segmented into blocks; per‑sequence page tables.
* **Persistence**: optional disk snapshots of tail blocks and logits cache; restored on session resume by id (default enabled).

### Sampling & Controls

* Top‑k, top‑p, temperature, repetition penalty, frequency/presence penalties, grammars/JSON schema constrained decoding; tool/function calling schema passthrough.

### Security & Sandboxing

* App sandbox with file‑access prompts; keychain storage for remote registry tokens; uds auth via capability token.

### Observability

* Metrics: tokens/s, prefill/decoding latency histograms, KV arena utilization, cache hit rates, acceptance rate (speculative), kernel time breakdown.
* Tracing: per‑request spans; flamegraphs for kernels.

### API Specs (selected)

* **OpenAI**: `/v1/chat/completions` (SSE), `/v1/embeddings`.
* **Ollama**: `/api/chat`, `/api/generate`, `/api/embed`, `/api/pull`, `/api/create`, `/api/ps`, `/api/tags`.

### Default Configs

* `server.yaml`: port, uds path, max_batch_tokens, target_latency_ms, enable_speculative=true, draft_model="llama-3-8b-instruct‑draft", kv_persistence=true.
* `models/*.yaml`: model path/uri, tokenizer, chat template, quantization, max context, rope scaling.

### Performance Playbook (M4)

1. Pre‑warm kernels per shape; cache compiled metallib variants.
2. Enable speculative decoding by default; auto‑tune `k` to keep verification throughput ≥1.3× draft.
3. Cap batch by p95 latency; increase if KV headroom >20%.
4. Use FP16 compute for attention and MLP; dequant on load for Q‑weights.

---

## Milestones

1. **M0 – Skeleton**

   * Repo layout, CMake, Metal toolchain, MLX integration; minimal REST; run FP16 llama‑style model single‑request.
2. **M1 – Batching & KV**

   * Continuous batching, paged KV arena with eviction; SSE streaming; SQLite registry; GGUF loader.
3. **M2 – Quant & Kernels**

   * Q‑dequant matmul, fused attention/rope/norm kernels; latency target <80 ms/token on 7B class at 4‑bit.
4. **M3 – Speculative & Persistence**

   * Draft model verify path; KV/logits persistence default on; acceptance auto‑tuning.
5. **M4 – APIs & GUI**

   * OpenAI & Ollama shims; React tray/dock app; metrics dashboard.
6. **M5 – Adapters & Vision (optional)**

   * LoRA stacking; CLIP/ViT encoder path; image‑chat template.
7. **M6 – Polish & Release**

   * Sandboxing, Sparkle updates, codesign/notarize; docs and SDKs.

---

## Gathering Results

* **Functional**: API compatibility tests (OpenAI/Ollama), model zoo smoke tests (Llama/Mistral/Gemma/Qwen).
* **Performance**: tokens/s (prefill & decode), p50/p95 latency, peak memory, KV hit rate, speculative acceptance rate.
* **Stability**: long‑run soak tests (24–72h), memory leak checks, KV persistence correctness, recovery from sleep.
* **UX**: app startup time, quick‑switch model latency, UI responsiveness under load.
* **Acceptance criteria (MVP)**: chat works with 7B–8B at 4‑bit with <1 s first‑token and <80 ms/token steady‑state on M4; embeddings <20 ms/sample; OpenAI/Ollama clients work unchanged.
