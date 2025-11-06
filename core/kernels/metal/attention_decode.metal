/**
 * @file attention_decode.metal
 * @brief Fused attention decode kernel with paged KV cache
 *
 * This kernel implements the decode path for autoregressive generation:
 * 1. Read current query Q from input
 * 2. Walk paged KV cache to load all past K, V values
 * 3. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 * 4. Apply causal mask (if needed)
 * 5. Softmax over scores (numerically stable with fp32 accumulation)
 * 6. Compute context: context = softmax(scores) @ V
 *
 * Features:
 * - Paged KV cache walker for non-contiguous memory
 * - Grouped Query Attention (GQA) support
 * - Numerically stable softmax with fp32 accumulation
 * - Configurable head dimensions and block sizes
 * - Optional sliding window attention
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Attention decode kernel arguments (ZERO-COPY version)
 *
 * This structure is passed to the kernel via an argument buffer.
 * Blocks are passed in their native [num_layers, block_size, heads, dim] format.
 * Kernel indexes directly into blocks using layer_idx, avoiding CPU-side slicing.
 */
struct AttentionDecodeArgs {
  device const half* q;           // Query: [batch, num_heads, head_dim]
  device const half* k_cache;     // K cache pages: [num_pages, block_size, num_kv_heads, head_dim] OR
                                  // K cache blocks: [num_pages, num_layers, block_size, num_kv_heads, head_dim]
  device const half* v_cache;     // V cache pages: [num_pages, block_size, num_kv_heads, head_dim] OR
                                  // V cache blocks: [num_pages, num_layers, block_size, num_kv_heads, head_dim]
  device half* output;            // Output context: [batch, num_heads, head_dim]
  device const int* page_table;   // Page table: [batch, max_blocks_per_seq]
  device const int* seq_lengths;  // Sequence lengths: [batch]

  uint batch_size;
  uint num_heads;
  uint num_kv_heads;
  uint head_dim;
  uint block_size;
  uint max_blocks_per_seq;
  uint num_layers;                // NEW: Total layers in model (for block format)
  uint layer_idx;                 // NEW: Current layer index (for block format)
  bool use_block_format;          // NEW: If true, cache is [pages, layers, ...]; if false, [pages, ...]
  float scale;                    // 1/sqrt(head_dim)
  bool use_sliding_window;
  uint sliding_window_size;
};

/**
 * @brief Fused attention decode kernel (fp16 variant)
 *
 * Each threadgroup processes one query head. Threads within the threadgroup
 * cooperatively load KV cache, compute attention scores, perform softmax,
 * and compute the output context.
 *
 * Grid layout:
 *   - X: batch_size * num_heads (one threadgroup per query head)
 *   - Y: 1
 *   - Z: 1
 *
 * Threadgroup layout:
 *   - Threads: 256 (configurable, must divide evenly into head_dim)
 *   - Each thread processes multiple elements via striding
 *
 * Memory layout:
 *   - Threadgroup memory stores attention scores for current stripe
 *   - Threadgroup memory also used for softmax reduction (max, sum)
 */
kernel void attention_decode_fused(
    constant AttentionDecodeArgs& args [[buffer(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    threadgroup float* shared_scores [[threadgroup(0)]],
    threadgroup float* shared_reduce [[threadgroup(1)]]) {

  // Each threadgroup handles one query head
  if (gid >= args.batch_size * args.num_heads) {
    return;
  }

  // Decode batch and head indices
  const uint batch_idx = gid / args.num_heads;
  const uint head_idx = gid % args.num_heads;

  // For GQA: map query head to KV head
  const uint kv_head_idx = head_idx / (args.num_heads / args.num_kv_heads);

  // Get sequence length for this batch element
  const int seq_len = args.seq_lengths[batch_idx];

  if (seq_len <= 0) {
    // Empty sequence, write zeros
    for (uint d = lid; d < args.head_dim; d += threadgroup_size) {
      args.output[gid * args.head_dim + d] = 0.0h;
    }
    return;
  }

  // Load query into registers (each thread loads a subset of head_dim)
  // Q shape: [batch, num_heads, head_dim]
  const uint q_offset = batch_idx * args.num_heads * args.head_dim + head_idx * args.head_dim;
  device const half* q_ptr = args.q + q_offset;

  // Phase 1: Compute attention scores and find max (for numerical stability)
  // We'll process in stripes of 64 tokens at a time to bound memory usage
  const uint STRIPE_SIZE = 64;
  const uint seq_len_u = uint(seq_len);
  float max_score = -INFINITY;

  // Process all tokens in stripes
  for (uint stripe_start = 0; stripe_start < seq_len_u; stripe_start += STRIPE_SIZE) {
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, seq_len_u);
    const uint stripe_len = stripe_end - stripe_start;

    // Compute scores for this stripe
    for (uint tok = lid; tok < stripe_len; tok += threadgroup_size) {
      const uint token_pos = stripe_start + tok;

      // Determine which block and offset within block
      const uint block_idx = token_pos / args.block_size;
      const uint block_offset = token_pos % args.block_size;

      // Get page ID from page table
      const int page_id = args.page_table[batch_idx * args.max_blocks_per_seq + block_idx];

      if (page_id < 0) {
        // Invalid page, should not happen in valid sequence
        shared_scores[tok] = -INFINITY;
        continue;
      }

      // Calculate K pointer for this token
      // Two formats supported:
      // 1. Stacked: [num_pages, block_size, num_kv_heads, head_dim]
      // 2. Block: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (ZERO-COPY)
      uint k_offset;
      if (args.use_block_format) {
        // Block format: index with layer_idx
        k_offset = page_id * args.num_layers * args.block_size * args.num_kv_heads * args.head_dim +
                   args.layer_idx * args.block_size * args.num_kv_heads * args.head_dim +
                   block_offset * args.num_kv_heads * args.head_dim +
                   kv_head_idx * args.head_dim;
      } else {
        // Stacked format (legacy)
        k_offset = page_id * args.block_size * args.num_kv_heads * args.head_dim +
                   block_offset * args.num_kv_heads * args.head_dim +
                   kv_head_idx * args.head_dim;
      }
      device const half* k_ptr = args.k_cache + k_offset;

      // Compute Q @ K^T (dot product)
      float score = 0.0f;
      for (uint d = 0; d < args.head_dim; d++) {
        float q_val = float(q_ptr[d]);
        float k_val = float(k_ptr[d]);
        score += q_val * k_val;
      }

      // Scale by 1/sqrt(d_k)
      score *= args.scale;

      // Apply sliding window mask if enabled
      if (args.use_sliding_window) {
        const uint distance = seq_len - 1 - token_pos;  // Distance from current position
        if (distance >= args.sliding_window_size) {
          score = -INFINITY;
        }
      }

      shared_scores[tok] = score;
      max_score = max(max_score, score);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find max across threadgroup
    shared_reduce[lid] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        shared_reduce[lid] = max(shared_reduce[lid], shared_reduce[lid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    max_score = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Phase 2: Compute exp(score - max) and sum for softmax normalization
  float sum_exp = 0.0f;

  for (uint stripe_start = 0; stripe_start < seq_len_u; stripe_start += STRIPE_SIZE) {
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, seq_len_u);
    const uint stripe_len = stripe_end - stripe_start;

    for (uint tok = lid; tok < stripe_len; tok += threadgroup_size) {
      float score = shared_scores[tok];
      float exp_score = exp(score - max_score);
      shared_scores[tok] = exp_score;
      sum_exp += exp_score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Reduce sum across threadgroup
  shared_reduce[lid] = sum_exp;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      shared_reduce[lid] += shared_reduce[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  sum_exp = shared_reduce[0];
  const float inv_sum = 1.0f / (sum_exp + 1e-8f);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 3: Normalize scores (complete softmax)
  for (uint stripe_start = 0; stripe_start < seq_len_u; stripe_start += STRIPE_SIZE) {
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, seq_len_u);
    const uint stripe_len = stripe_end - stripe_start;

    for (uint tok = lid; tok < stripe_len; tok += threadgroup_size) {
      shared_scores[tok] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Phase 4: Compute context = softmax(scores) @ V
  // Each thread computes a subset of the head_dim output
  for (uint d = lid; d < args.head_dim; d += threadgroup_size) {
    float accum = 0.0f;

    for (uint token_pos = 0; token_pos < seq_len_u; token_pos++) {
      // Determine which block and offset
      const uint block_idx = token_pos / args.block_size;
      const uint block_offset = token_pos % args.block_size;

      // Get page ID
      const int page_id = args.page_table[batch_idx * args.max_blocks_per_seq + block_idx];

      if (page_id < 0) {
        continue;
      }

      // Calculate V pointer
      // Two formats supported (same as K cache):
      // 1. Stacked: [num_pages, block_size, num_kv_heads, head_dim]
      // 2. Block: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (ZERO-COPY)
      uint v_offset;
      if (args.use_block_format) {
        // Block format: index with layer_idx
        v_offset = page_id * args.num_layers * args.block_size * args.num_kv_heads * args.head_dim +
                   args.layer_idx * args.block_size * args.num_kv_heads * args.head_dim +
                   block_offset * args.num_kv_heads * args.head_dim +
                   kv_head_idx * args.head_dim;
      } else {
        // Stacked format (legacy)
        v_offset = page_id * args.block_size * args.num_kv_heads * args.head_dim +
                   block_offset * args.num_kv_heads * args.head_dim +
                   kv_head_idx * args.head_dim;
      }
      device const half* v_ptr = args.v_cache + v_offset;

      // Load attention weight from shared memory
      const uint stripe_idx = token_pos % STRIPE_SIZE;
      const float attn_weight = shared_scores[stripe_idx];

      // Accumulate: output[d] += attn_weight * V[token_pos, d]
      float v_val = float(v_ptr[d]);
      accum += attn_weight * v_val;
    }

    // Write output
    const uint out_offset = gid * args.head_dim + d;
    args.output[out_offset] = half(accum);
  }
}

// Note: Kernel variants for different head dimensions (small_head, large_head)
// will be added in Phase 2 with optimized implementations.
// For Phase 1, we use a single general-purpose kernel that works for all head dimensions.
