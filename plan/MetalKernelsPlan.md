# Metal Kernel Implementation Plan (MLXR, Apple Silicon M4)

This plan lists the exact Metal compute shaders we will implement, their responsibilities, data layouts, variants, and performance targets for an Apple Silicon M4 GPU in a unified‑memory MacBook Pro. It’s aligned with our paged‑KV, continuous batching architecture and supports GGUF K‑quants and MLX tensors.

---

## Global Conventions

**Tensor layouts**

* Activations: row‑major unless noted. Shapes follow `[B, T, H, D]` for attention (Batch, Tokens, Heads, HeadDim), and `[B, T, Hidden]` for MLP.
* QKV projections: packed as contiguous tiles per head.
* KV cache: page‑blocked; each page stores `{K[V][block_tokens][H][D]}` with 16‑ or 32‑token blocks.

**DTypes & Quant**

* Compute: `half` (fp16). Optional fp32 accumulation in softmax/normalization reductions.
* Weights: fp16, fp8 (E4M3/E5M2), int4 K‑quants (Q2_K…Q8_K via unpack scales + zero‑points).
* Activations: fp16.

**Threading**

* Use **threadgroup memory** to stage K/V tiles and dequant scales; use **simdgroup reductions** for softmax and norms.
* Target threadgroup size 256–512 threads, tuned per kernel and head_dim.

**Argument buffers**

* All kernels use MSL argument buffers with colocated descriptors for weights, scales, page tables, and constants; align to 16 bytes.

---

## Kernel Catalog

### 1) `attention_prefill_fused`

**Purpose**: Fused prefill path: `X·Wqkv → [Q,K,V] → RoPE(Q,K) → scaled dot‑product attention → context`.

**Inputs**: `X[B,T,Hid]`, packed `Wqkv[Hid, 3*H*D]` (quantized or fp16), optional bias; RoPE tables; causal/prefix mask; KV page allocator handle.

**Outputs**: `Context[B,T,H*D]`, and writes `K,V` into KV cache pages for tokens `[0..T)`.

**Variants**:

* head_dim ∈ {64, 80, 96, 112, 128, 160, 192, 256}
* block_tokens ∈ {16, 32}
* weight dtype ∈ {fp16, fp8, int4_k}

**Notes**:

* Use tiled GEMM with on‑the‑fly dequantization; fuse RoPE before score matmul; numeric‑stable softmax with blockwise max/sum in fp32.

---

### 2) `attention_decode_fused`

**Purpose**: Decode path per step (or micro‑batch): read paged KV up to current position; compute attention against past and produce context for the next token.

**Inputs**: `q[Batch,H,D]`, KV page table per sequence, `K/V` pages (quantization already materialized as fp16).

**Outputs**: `ctx[Batch,H,D]`.

**Features**:

* Walk page tables to stitch K/V across non‑contiguous pages; mask future tokens; support sliding‑window attention.
* Softmax in chunks (e.g., 64‑token stripes) to bound memory.

**Variants**: head_dim set; block_tokens set.

---

### 3) `q_gemm_dequant`

**Purpose**: High‑throughput matmul for quantized weights used by attention/MLP projections: `Y = X · Wq + bias` with dequant in epilogue.

**Inputs**: `X[M,K] (fp16)`, `Wq[packed]`, `scales[groups]`, `zeros[groups]` (for K‑quants), `group_size`.

**Outputs**: `Y[M,N] (fp16)`.

**Variants**: int4 (Q2_K–Q8_K), int8, fp8 (E4M3/E5M2); tile sizes tuned per K/N.

**Notes**: Vectorized loads (uint4/uint8), unpack in registers, accumulate in fp16/fp32; optional activation fusion (SwiGLU gate).

---

### 4) `rope_apply`

**Purpose**: Rotary positional embedding for Q and K, interleaved layout.

**Inputs**: `Q/K[B,H,T,D]`, precomputed `cos,sin[T,D/2]`.

**Outputs**: in‑place Q/K (fp16).

**Variants**: base, NTK‑scaled, YaRN‑scaled params.

---

### 5) `rmsnorm_fused`

**Purpose**: RMSNorm + optional bias + residual add fusion.

**Inputs**: `X[B,T,Hid]`, `weight[Hid]`, optional `bias` and `residual`.

**Outputs**: `Y[B,T,Hid]`.

**Notes**: simdgroup reduction with Kahan compensation for stability; epsilon as compile‑time constant.

---

### 6) `swiglu_mlp_fused`

**Purpose**: Fused gated MLP: `(X·W_up) ⊗ σ(X·W_gate) → W_down` with optional dequant in each matmul.

**Inputs**: `X`, `W_up`, `W_gate`, `W_down`, quant params.

**Outputs**: `Y`.

**Variants**: fp16 weights, int4/int8/fp8 via `q_gemm_dequant` sub‑routine.

---

### 7) `kv_pack_store` / `kv_load_unpack`

**Purpose**: Efficiently write/read K,V blocks to/from KV pages.

**Inputs**: per‑sequence page ids, `K/V` tiles.

**Outputs**: page‑aligned writes to `device` buffers; or reads into on‑chip tiles.

**Notes**: Align to 128‑byte boundaries; coalesce with 2D threadgroup writes.

---

### 8) `kv_persist_copy`

**Purpose**: Async DMA‑style copy of KV blocks between GPU memory and CPU‑pinned buffers for persistence/eviction.

**Features**: batched copy with fence signaling back to runtime; optional compression (fp16→fp8) for cold blocks.

---

### 9) `softmax_block`

**Purpose**: Standalone numerically‑stable softmax for fallback and testing (prefill & decode stripes).

**Notes**: fp32 max/sum, rescale trick across stripes.

---

### 10) (Optional) `clip_patchify_proj`

**Purpose**: For VLMs: image patchify + linear projection; shared tiles, local memory reuse.

---

## Data Layouts & Memory

**QKV Packing**

* `Wqkv` packed by head: `[Hid, H*3*D]` with head‑major ordering for contiguous per‑head tiles.

**KV Cache Pages**

* Page size: `block_tokens * H * D * 2` (K and V) * sizeof(dtype). Recommend `block_tokens=32` for decode locality.
* Per‑sequence page table: compact 16‑bit page ids; upper bits store flags (dirty/persisted).

**Quant Groups**

* Group size `g ∈ {32, 64, 128}` along K‑dimension; `scales[g]`, `zeros[g]` contiguous; broadcast within tile.

---

## Kernel Interfaces (MSL Sketches)

```metal
struct MatmulArgs {
  device half* X;          // MxK
  device uchar* Wq;        // packed weights
  device half* Y;          // MxN
  device half* scales;     // per-group
  device char* zeros;      // per-group
  ushort M, N, K;
  ushort group_size;
  ushort ldX, ldY;         // leading dims
};

kernel void q_gemm_dequant(
  device MatmulArgs& args [[ buffer(0) ]],
  uint2 tidxy [[ thread_position_in_grid ]],
  uint2 lidxy [[ thread_position_in_threadgroup ]]) {
  // tile loads, unpack, dequant, accumulate, epilogue
}
```

```metal
struct AttnArgs {
  device half* X;          // input activations
  device half* Wqkv;       // or quantized + params
  device half* Ctx;        // output context
  device half* rope_cos;
  device half* rope_sin;
  device ushort* page_tbl; // per-seq page ids
  device half* K_pages;    // kv arena base
  device half* V_pages;
  ushort B, T, H, D;
  ushort block_tokens;
  bool causal;
};

kernel void attention_prefill_fused(
  device AttnArgs& args [[ buffer(0) ]],
  uint tg_id [[ threadgroup_position_in_grid ]],
  uint tid   [[ thread_position_in_threadgroup ]]) {
  // qkv proj (+dequant), rope, attention, ctx, kv store
}
```

---

## Variant Matrix & Build

**HeadDim**: 64, 80, 96, 112, 128, 160, 192, 256
**BlockTokens**: 16, 32
**Quant**: fp16, fp8(E4M3/E5M2), int4(Q2_K–Q8_K), int8
**Rope**: base, NTK, YaRN

**Build Artifacts**

* `attention_prefill_fused_hd{D}_bt{B}_{dtype}.metallib`
* `attention_decode_fused_hd{D}_bt{B}_{dtype}.metallib`
* `q_gemm_dequant_{quant}_{tile}.metallib`
* `rmsnorm_fused_{hid_tile}.metallib`
* `swiglu_mlp_fused_{quant}.metallib`

Scripts: `scripts/build_metal.sh` compiles all variants; runtime chooses best fit by shape & dtype.

---

## Scheduling & Occupancy Targets

* Aim ≥ 60% occupancy for attention kernels at D≤128; ≥ 50% at D≥192.
* Prefill bandwidth: ≥ 1.3× decode throughput.
* Decode kernel budget: < 0.6 ms/head for 7B models at D=128, block_tokens=32.

---

## Numerical Stability & Edge Cases

* Softmax: two‑pass (max, then exp/sum) with fp32 accumulation; subtract running max per stripe.
* RMSNorm: epsilon=1e‑5; clamp variance to avoid underflow.
* RoPE: guard large positions when using NTK/YaRN scaling; use sine/cos LUT in `half` with fp32 intermediate.
* Quant dequant: pre‑scale to `half` range; fuse bias add after matmul.

---

## Integration Hooks

* Expose kernels via C++ wrappers that accept MLX tensors (device allocations) and raw `id<MTLBuffer>` for fast path.
* Argument buffers cached by `(shape, variant)` key; support pipeline state caching across runs.
* KV persistence signals completion via shared event to the scheduler.

---

## Testing & Benchmarks

* **Correctness**: compare against MLX fp16 reference; max relative error <1e‑2 on attention/MLP outputs.
* **Perf micro‑benches**: sweep `(B,T,H,D)` across 7B/8B/13B shapes; report tokens/s, latency per token.
* **Stress**: fragmented page tables (simulate eviction); long‑context (≥32k) with RoPE scaling.

---

## Milestones (Kernels)

1. **K0**: `q_gemm_dequant` (int4/int8) + `rmsnorm_fused`.
2. **K1**: `attention_decode_fused` (fp16) + KV page walker.
3. **K2**: `attention_prefill_fused` + RoPE integration.
4. **K3**: Quant variants (fp8/int4) in attention paths.
5. **K4**: `swiglu_mlp_fused` with dequant fusion.
6. **K5**: `kv_pack_store/load` + `kv_persist_copy` + persistence signaling.
7. **K6**: Optional CLIP patchify.

**Exit criteria**: Meets latency targets in our Gathering Results; correctness within tolerances.
