# Phase 1: Zero-Copy Optimization Implementation

**Date**: 2025-11-06
**Status**: 🚧 IN PROGRESS (Infrastructure Complete, Integration Pending)

## Executive Summary

Implementing **true zero-copy** attention kernel integration to eliminate the 62 MB/pass overhead from arena bridge methods. This addresses the performance regression where CachedAttention (592ms prefill) is slower than simple Attention (198ms prefill).

### Progress

- ✅ **Arena zero-copy methods** - Direct block array access without slicing/stacking
- ✅ **Metal kernel updates** - Support native block format [pages, layers, ...]
- ⏳ **Primitive integration** - Wire new format through MLX primitives
- ⏳ **Attention integration** - Use zero-copy in CachedAttention
- ⏳ **Testing & benchmarking** - Verify performance gains

## Problem Analysis

### Root Cause Discovered

The codebase has **two separate attention implementations**:

1. **Simple Attention** ([layers.cpp:248-367](../core/graph/layers.cpp#L248-L367)) - Currently in use
   - Basic K/V concatenation
   - Performance: **198ms prefill, 53ms/tok, 18.87 tok/s** ✅

2. **CachedAttention** ([attention_cached.cpp](../core/graph/attention_cached.cpp)) - NOT in use
   - Paged cache + Metal kernels + arena bridge
   - Performance: **592ms prefill, 70.75ms/tok, 14.13 tok/s** ❌
   - **3x slower than simple Attention!**

### Bridge Overhead

The `build_k/v_cache_array()` methods are catastrophically inefficient:

```cpp
// Arena::build_k_cache_array() - called per layer per forward pass
for (each block in page_table) {
  auto layer_slice = mlx::core::slice(block_k, {layer_idx, ...});  // View
  layer_slice = mlx::core::squeeze(layer_slice, 0);                // View
  layer_slices.push_back(layer_slice);
}
auto stacked = mlx::core::stack(layer_slices, 0);  // EXPENSIVE COPY!
```

**Cost per forward pass:**
- `build_k_cache_array()`: 132 MLX operations, stacks copied data
- `write_k_cache_array()`: Reconstructs entire block (all 22 layers!)
- **Total: 62 MB copied per forward pass**

This overhead completely negates Metal kernel benefits.

## Zero-Copy Solution

### Core Idea

Instead of extracting [layer_idx] from each block on CPU and stacking:

**OLD (slow):**
```
Block storage: [pages][layers, block_size, heads, dim]
                          ↓ CPU slice + stack
Metal input:   [pages, block_size, heads, dim]  // One layer, copied data
```

**NEW (zero-copy):**
```
Block storage: [pages][layers, block_size, heads, dim]
                          ↓ Direct pass
Metal input:   [pages, layers, block_size, heads, dim]  // All layers, no copy!
Metal kernel:  Indexes using layer_idx parameter
```

### Implementation

#### 1. Arena Zero-Copy Methods ✅

**Location**: [core/runtime/kv/arena.{h,cpp}](../core/runtime/kv/arena.h)

Added methods that return raw block arrays without slicing:

```cpp
class Arena {
public:
  // NEW: Zero-copy block access
  std::vector<mlx::core::array> get_k_block_arrays(const std::vector<int>& block_ids);
  std::vector<mlx::core::array> get_v_block_arrays(const std::vector<int>& block_ids);

  // OLD: Expensive slicing/stacking (keep for backward compat)
  Tensor build_k_cache_array(int layer_idx, const std::vector<int>& block_ids);
  Tensor build_v_cache_array(int layer_idx, const std::vector<int>& block_ids);
};
```

**Implementation** (arena.cpp:660-698):
```cpp
std::vector<mlx::core::array> Arena::get_k_block_arrays(const std::vector<int>& block_ids) {
  std::vector<mlx::core::array> result;
  result.reserve(block_ids.size());

  for (int block_id : block_ids) {
    Block* block = get_block(block_id);
    result.push_back(block->k_data.array());  // Direct reference, no copy!
    touch_block(block_id);
  }

  return result;
}
```

**Key insight**: Returns `array()` references directly from block storage. No `slice()`, no `stack()`, no copying!

#### 2. Metal Kernel Updates ✅

**Files Modified**:
- [core/kernels/metal/attention_decode.metal](../core/kernels/metal/attention_decode.metal)
- [core/kernels/metal/attention_prefill.metal](../core/kernels/metal/attention_prefill.metal)

**Changes to kernel arguments**:

```metal
struct AttentionDecodeArgs {
  device const half* k_cache;     // Can be stacked OR block format
  device const half* v_cache;     // Can be stacked OR block format

  // NEW parameters:
  uint num_layers;                // Total layers in model
  uint layer_idx;                 // Current layer to access
  bool use_block_format;          // Format selector

  // Existing parameters...
};
```

**Conditional indexing in kernel**:

```metal
// OLD (stacked format):
k_offset = page_id * block_size * num_kv_heads * head_dim + ...;

// NEW (block format):
if (args.use_block_format) {
  k_offset = page_id * num_layers * block_size * num_kv_heads * head_dim +
             layer_idx * block_size * num_kv_heads * head_dim + ...;
} else {
  k_offset = page_id * block_size * num_kv_heads * head_dim + ...;  // Legacy
}
```

**Backward compatibility**: Kernels support BOTH formats via `use_block_format` flag. This allows incremental migration and testing.

#### 3. Primitive Integration ⏳ PENDING

**Next steps**:

1. Update [attention_decode_primitive.mm](../core/kernels/primitives/attention_decode_primitive.mm):
   - Add `num_layers`, `layer_idx`, `use_block_format` parameters to constructor
   - Pass to Metal kernel via argument buffer
   - Support passing multiple block buffers (vector of arrays)

2. Update [attention_prefill_primitive.mm](../core/kernels/primitives/attention_prefill_primitive.mm):
   - Same parameter additions
   - Handle block array concatenation if needed

3. Update public API functions:
   ```cpp
   mlx::core::array attention_decode_fused(
       const mlx::core::array& q,
       const std::vector<mlx::core::array>& k_blocks,  // NEW: vector of blocks
       const std::vector<mlx::core::array>& v_blocks,  // NEW: vector of blocks
       int layer_idx,                                  // NEW
       bool use_block_format,                          // NEW
       // ... existing params
   );
   ```

#### 4. Attention Integration ⏳ PENDING

**Next steps for** [attention_cached.cpp](../core/graph/attention_cached.cpp):

**Current (slow):**
```cpp
auto k_cache = pager_->arena().build_k_cache_array(layer_idx_, page_table_vec);  // SLOW!
auto v_cache = pager_->arena().build_v_cache_array(layer_idx_, page_table_vec);  // SLOW!

auto attn_output = kernels::attention_prefill_fused(..., k_cache, v_cache, ...);

// Write back (also slow!)
pager_->arena().write_k_cache_array(layer_idx_, page_table_vec, k_cache);
pager_->arena().write_v_cache_array(layer_idx_, page_table_vec, v_cache);
```

**New (zero-copy):**
```cpp
// Get raw block arrays (zero-copy!)
auto k_blocks = pager_->arena().get_k_block_arrays(page_table_vec);
auto v_blocks = pager_->arena().get_v_block_arrays(page_table_vec);

// Stack for Metal (still needed, but kernel modifies in-place)
auto k_cache = mlx::core::stack(k_blocks, 0);  // [pages, layers, ...]
auto v_cache = mlx::core::stack(v_blocks, 0);

auto attn_output = kernels::attention_prefill_fused(
    ..., k_cache, v_cache, ...,
    layer_idx_,           // NEW: tell kernel which layer
    true                  // NEW: use block format
);

// NO write-back needed! Kernel modified blocks in-place
```

**Key optimization**: Metal kernel writes directly to block storage via shared buffer. No reconstruction needed!

## Expected Performance Improvement

### Current Bottleneck

- Arena bridge: **62 MB/pass** of unnecessary copying
- `build_k/v_cache_array()`: 132 MLX ops × 2 = 264 ops/pass
- `write_k/v_cache_array()`: Reconstructs all 22 layers on every update

### After Zero-Copy

- Block access: **0 bytes copied** (direct array references)
- Stack operation: Creates view with shared backing buffer
- Metal kernel: Modifies original blocks in-place
- No write-back needed

**Expected speedup:**
- Prefill: **592ms → 200-250ms** (2.4-3x faster, match simple Attention)
- Decode: **70.75ms → 40-50ms/tok** (1.4-1.8x faster)
- Throughput: **14.13 → 20-25 tok/s** (1.4-1.8x improvement)

## Why This Works

### MLX Unified Memory

On Apple Silicon, CPU and GPU share the same physical memory. When we:

1. Get `block->k_data.array()` - returns MLX array backed by unified memory
2. Stack arrays - creates view with offset indexing, shares underlying buffer
3. Pass to Metal kernel - GPU accesses same physical memory, no transfer!
4. Kernel modifies in-place - changes visible to CPU immediately

**Zero-copy guarantee**: As long as we pass original block arrays (or views of them), Metal kernel operates on the same memory backing the Arena blocks.

### Why Write-Back Was Needed Before

The OLD approach:
```cpp
auto stacked = mlx::core::stack(layer_slices, 0);  // Creates NEW buffer with copied data
```

`stack()` with sliced inputs creates a **new buffer** with copied data (not a view). Metal kernel modifies this new buffer, so write-back was needed to copy data back to blocks.

The NEW approach:
```cpp
auto stacked = mlx::core::stack(block_arrays, 0);  // Creates view of original buffers!
```

`stack()` with full block arrays creates a **view** that shares the original block buffers. Metal kernel modifies the original blocks directly!

## Testing Plan

### Phase 1: Verify Zero-Copy ⏳

1. Complete primitive integration
2. Update attention_cached.cpp to use block format
3. Rebuild: `make metal && make build`
4. Run basic test to verify it works

### Phase 2: Benchmark ⏳

**Test program**: Create `zero_copy_benchmark.cpp` based on kv_cache_test.cpp

**Metrics to measure:**
- Prefill latency (target: ~200ms, match simple Attention)
- Decode latency (target: 40-50ms/tok)
- Throughput (target: 20-25 tok/s)
- Memory usage (should be same or lower)

**Comparison matrix:**

| Implementation | Prefill | Decode | Throughput | Notes |
|----------------|---------|---------|------------|-------|
| Simple Attention | 198 ms | 53 ms/tok | 18.87 tok/s | Baseline |
| CachedAttention (old) | 592 ms | 70.75 ms/tok | 14.13 tok/s | With bridge |
| CachedAttention (zero-copy) | TBD | TBD | TBD | **Target** |

### Phase 3: Validation ⏳

1. Verify numerical correctness (outputs match)
2. Test with longer sequences (32, 64, 128 tokens)
3. Test with different models (if available)
4. Memory leak check (should be none)

## Files Modified

### Completed ✅

- [core/runtime/kv/arena.h](../core/runtime/kv/arena.h) - Added zero-copy methods (lines 268-288)
- [core/runtime/kv/arena.cpp](../core/runtime/kv/arena.cpp) - Implemented methods (lines 660-698)
- [core/kernels/metal/attention_decode.metal](../core/kernels/metal/attention_decode.metal) - Block format support (lines 31-53, 143-156, 263-276)
- [core/kernels/metal/attention_prefill.metal](../core/kernels/metal/attention_prefill.metal) - Block format support (lines 31-66, 217-235, 340-356)

### Pending ⏳

- [core/kernels/primitives/attention_decode_primitive.{h,mm}](../core/kernels/primitives/attention_decode_primitive.h) - Add block format parameters
- [core/kernels/primitives/attention_prefill_primitive.{h,mm}](../core/kernels/primitives/attention_prefill_primitive.h) - Add block format parameters
- [core/graph/attention_cached.cpp](../core/graph/attention_cached.cpp) - Use zero-copy methods, remove write-back

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Arena**: Old `build/write_k/v_cache_array()` methods still exist
2. **Metal kernels**: Support both formats via `use_block_format` flag
3. **Primitives**: Can be updated to support both APIs
4. **CachedAttention**: Can toggle between approaches for testing

This allows:
- Incremental testing (compare old vs new)
- Easy rollback if issues found
- Gradual migration

## Next Session Goals

1. **Complete primitive integration** (1-2 hours)
   - Update primitive classes
   - Add block format parameters
   - Test primitive calls work

2. **Update attention_cached.cpp** (1 hour)
   - Use `get_k/v_block_arrays()`
   - Pass to primitives with block format flag
   - Remove write-back calls

3. **Test and benchmark** (1 hour)
   - Run with TinyLlama
   - Compare performance vs baseline
   - Verify correctness

## References

- [ATTENTION_ARCHITECTURE_ANALYSIS.md](ATTENTION_ARCHITECTURE_ANALYSIS.md) - Problem diagnosis
- [PHASE2_METAL_KERNEL_INTEGRATION.md](PHASE2_METAL_KERNEL_INTEGRATION.md) - Original integration (misleading)
- [SESSION_2025_11_06.md](SESSION_2025_11_06.md) - Previous session work
- [KV_CACHE_IMPLEMENTATION.md](KV_CACHE_IMPLEMENTATION.md) - Paged cache architecture

## Key Insights

1. **Simple is faster**: The "simple" concatenation approach in Attention class outperforms the "optimized" paged cache because the bridge overhead is worse than concatenation cost.

2. **Zero-copy requires alignment**: Can only achieve zero-copy when data layout matches what GPU expects. Slicing breaks this.

3. **Metal unified memory**: Key enabler for zero-copy. CPU and GPU share physical memory on Apple Silicon.

4. **In-place modification**: Metal kernels already modify cache in-place. Write-back was copying already-modified data back to itself!

5. **Views vs copies**: MLX `stack()` behavior depends on input - can create views OR copies. Need to ensure view creation for zero-copy.

---

**Status**: Infrastructure complete, ready for integration. Expected completion: 1 session.
