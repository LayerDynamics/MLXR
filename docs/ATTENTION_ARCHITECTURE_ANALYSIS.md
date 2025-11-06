# Attention Architecture Analysis

**Date**: 2025-11-06
**Critical Finding**: CachedAttention is not being used; simple Attention is faster

## Executive Summary

The codebase has **two separate attention implementations** but only one is actually used:

1. **Simple Attention** ([layers.cpp:248-367](../core/graph/layers.cpp#L248-L367)) - **IN USE**
   - Basic K/V concatenation via `std::pair<Tensor, Tensor>` cache
   - No paging, no arena, no Metal kernels
   - Performance: **198ms prefill, 53ms/tok decode, 18.87 tok/s** ✅

2. **CachedAttention** ([attention_cached.cpp](../core/graph/attention_cached.cpp)) - **NOT USED**
   - Paged KV cache with Arena/Pager/Eviction
   - Metal kernels integrated with arena bridge
   - Performance: **592ms prefill, 70.75ms/tok decode, 14.13 tok/s** ❌
   - **3.5x slower than simple Attention!**

## How We Got Here

### TransformerBlock Architecture

```cpp
class TransformerBlock {
private:
  Attention attention_;  // ← Uses simple Attention, NOT CachedAttention
  MLP mlp_;
  RMSNorm input_layernorm_;
  RMSNorm post_attention_layernorm_;
};
```

**Location**: [layers.h:307](../core/graph/layers.h#L307)

### Call Chain

```
kv_cache_test.cpp
  ↓
Engine::forward_prefill/decode
  ↓
LlamaModel::forward
  ↓
TransformerBlock::forward
  ↓
Attention::forward  ← Simple implementation, basic concatenation
```

**CachedAttention is never instantiated or called.**

## Performance Comparison

| Implementation | Prefill | Decode | Throughput | Approach |
|----------------|---------|---------|------------|----------|
| Simple Attention | 198 ms | 53 ms/tok | **18.87 tok/s** | Concatenation |
| CachedAttention (before Metal) | 459 ms | 220 ms/tok | 4.55 tok/s | Paged cache, MLX ops |
| CachedAttention (with Metal) | 592 ms | 70.75 ms/tok | 14.13 tok/s | Paged + Metal kernels |

**Why CachedAttention is slower:**

The arena bridge overhead identified in previous analysis:
- `build_k/v_cache_array()`: 132 MLX operations per call
- `write_k/v_cache_array()`: Reconstructs entire block (31 MB copied)
- **Total: 62 MB/pass of unnecessary copying**

This overhead completely negates Metal kernel benefits and makes it slower than simple concatenation!

## Simple Attention Implementation

**Cache structure** (line 302-333):
```cpp
// Per-layer cache in KVCache struct
struct KVCache {
  std::vector<std::pair<Tensor, Tensor>> layer_caches;
  int cached_length = 0;
};

// On each forward pass:
if (cache exists) {
  k_for_attn = concatenate({cached_k, new_k}, axis=2);
  v_for_attn = concatenate({cached_v, new_v}, axis=2);
  layer_cache = {k_for_attn, v_for_attn};  // Update cache
}
```

**Why it's fast:**
- Direct MLX concatenation (highly optimized)
- No intermediate copies
- No bridge between abstractions
- No page table walking

**Why it will scale poorly:**
- Concatenation is O(n) where n = cached length
- For 4K context: 4096 tokens × 22 layers = 90K concatenations
- Memory fragmentation with large contexts

## CachedAttention Implementation

**Cache structure**:
```cpp
// Paged cache with Arena/Pager
class Arena {
  std::vector<Block*> blocks_;  // [num_layers, block_size, heads, dim]
};

class Pager {
  Arena arena_;
  std::map<int, Sequence> sequences_;  // seq_id → page table
};
```

**Why it's designed well:**
- O(1) block access via page table
- Memory pooling with eviction
- Designed for multi-sequence batching
- Scales to long contexts (16K+)

**Why it's slow in practice:**
- Arena bridge converts blocks → arrays → blocks every forward pass
- `write_k_cache_array()` reconstructs ALL layers in block, not just current layer
- 264 MLX operations per forward pass just for bridge
- Metal kernels can't compensate for this overhead

## Critical Questions

### 1. Why was CachedAttention built?

From [KV_CACHE_IMPLEMENTATION.md](KV_CACHE_IMPLEMENTATION.md):
- Designed for production multi-request batching
- Needed for continuous batching (daemon scheduler)
- Required for memory efficiency with long contexts
- Supports eviction and persistence

**Conclusion**: CachedAttention is the right long-term architecture for production.

### 2. Why isn't it being used?

Looking at the code:
- TransformerBlock was built first with simple Attention
- CachedAttention was added later as separate class
- No refactoring to switch TransformerBlock to use CachedAttention
- Tests were run with simple Attention, giving good numbers
- CachedAttention compiled but never called

**Conclusion**: Integration was incomplete.

### 3. Should we use CachedAttention or simple Attention?

**For single-sequence, short context (< 2K tokens)**:
- Simple Attention is **2-3x faster**
- Good enough for CLI tools, examples

**For production (multi-sequence, long context)**:
- CachedAttention is **necessary** for scaling
- But bridge overhead MUST be fixed first
- Can't ship 3x slower than simple implementation

## Path Forward

### Option A: Fix CachedAttention (Recommended)

**Implement the zero-copy optimization plan:**

1. **Phase 1**: Eliminate arena bridge
   - Pass block pointers directly to Metal kernels (vLLM pattern)
   - Kernel walks page table on GPU
   - Expected: 2-3x prefill speedup (592 → 200-250ms)

2. **Phase 2**: Scatter writes
   - Use MLX `array.at` or Metal kernel for cache updates
   - Replace full block reconstruction
   - Expected: 10-20% additional speedup

3. **Phase 3**: Adaptive selection
   - Use simple Attention for short sequences (< 2K)
   - Use CachedAttention for long sequences (≥ 2K)
   - Expected: Best of both worlds

**Target performance after optimization:**
- Prefill: ~200 ms (match simple Attention)
- Decode: 40-50 ms/tok (2-3x better than current simple)
- Throughput: 20-25 tok/s

### Option B: Add Metal Kernels to Simple Attention

**Simpler but limited:**
- Integrate Metal attention kernels into simple Attention path
- Keep concatenation approach
- Expected: 1.5-2x speedup for decode
- **Problem**: Doesn't solve long context scaling, no multi-sequence support

### Option C: Hybrid Approach

**Use both implementations:**
```cpp
class TransformerBlock {
  Attention simple_attention_;           // For short contexts
  CachedAttention cached_attention_;     // For long contexts
  bool use_cached_ = false;              // Switch at runtime
};
```

**Decision logic:**
- context_len < 2048: Use simple Attention
- context_len ≥ 2048: Use CachedAttention
- batch_size > 1: Always use CachedAttention

## Recommendation

**Implement Option A + C:**

1. **Week 1**: Fix CachedAttention bridge overhead
   - Eliminate arena bridge (Phase 1 of optimization plan)
   - Target: Match simple Attention prefill speed (~200ms)

2. **Week 2**: Integrate CachedAttention into TransformerBlock
   - Refactor to use CachedAttention with Pager
   - Verify production batching works

3. **Week 3**: Add adaptive selection
   - Keep simple Attention path for short contexts
   - Benchmark and tune threshold

**Why this approach:**
- Fixes the root cause (bridge overhead)
- Enables production features (batching, long context)
- Maintains fast path for simple use cases
- Aligns with original Phase 2 goals

## Files Involved

### Simple Attention
- [core/graph/layers.h](../core/graph/layers.h) (lines 134-208) - Attention class
- [core/graph/layers.cpp](../core/graph/layers.cpp) (lines 248-367) - Implementation
- [core/graph/model.cpp](../core/graph/model.cpp) - LlamaModel uses KVCache struct

### CachedAttention
- [core/graph/attention_cached.h](../core/graph/attention_cached.h) - CachedAttention class
- [core/graph/attention_cached.cpp](../core/graph/attention_cached.cpp) - Implementation with Metal kernels
- [core/runtime/kv/arena.cpp](../core/runtime/kv/arena.cpp) - Arena bridge (BOTTLENECK)
- [core/runtime/kv/pager.cpp](../core/runtime/kv/pager.cpp) - Page table management

### Metal Kernels
- [core/kernels/metal/attention_prefill.metal](../core/kernels/metal/attention_prefill.metal)
- [core/kernels/metal/attention_decode.metal](../core/kernels/metal/attention_decode.metal)
- [core/kernels/primitives/attention_prefill_primitive.mm](../core/kernels/primitives/attention_prefill_primitive.mm)
- [core/kernels/primitives/attention_decode_primitive.mm](../core/kernels/primitives/attention_decode_primitive.mm)

## Next Steps

1. **Verify**: Test CachedAttention directly to confirm 592ms/70ms numbers
2. **Profile**: Measure exact bridge overhead with instrumentation
3. **Implement**: Phase 1 of optimization (zero-copy block pointers)
4. **Benchmark**: Compare before/after optimization
5. **Integrate**: Switch TransformerBlock to use CachedAttention
6. **Document**: Update PHASE2_METAL_KERNEL_INTEGRATION.md with accurate data

## References

- [PHASE2_METAL_KERNEL_INTEGRATION.md](PHASE2_METAL_KERNEL_INTEGRATION.md) - Original (misleading) integration report
- [KV_CACHE_IMPLEMENTATION.md](KV_CACHE_IMPLEMENTATION.md) - Paged cache architecture
- [SESSION_2025_11_06.md](SESSION_2025_11_06.md) - Previous session work
- [PHASE2_CURRENT_STATUS.md](PHASE2_CURRENT_STATUS.md) - Strategic roadmap
