/**
 * @file attention_prefill.metal
 * @brief Fused attention prefill kernel with paged KV cache storage
 *
 * This kernel implements the prefill path for prompt processing:
 * 1. Project input X to Q, K, V (using provided projections)
 * 2. Apply RoPE (Rotary Positional Embeddings) to Q and K
 * 3. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 * 4. Apply causal mask
 * 5. Softmax over scores (numerically stable with fp32 accumulation)
 * 6. Compute context: context = softmax(scores) @ V
 * 7. Store K, V into paged KV cache for future decode steps
 *
 * Features:
 * - Fused QKV projection → RoPE → attention → KV storage
 * - Paged KV cache storage for non-contiguous memory
 * - Causal masking for autoregressive generation
 * - Numerically stable softmax with fp32 accumulation
 * - Grouped Query Attention (GQA) support
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Attention prefill kernel arguments (ZERO-COPY version)
 *
 * Blocks are passed in their native [num_layers, block_size, heads, dim] format.
 * Kernel indexes directly into blocks using layer_idx, avoiding CPU-side slicing.
 */
struct AttentionPrefillArgs {
  // Input and output
  device const half* input;       // Input tokens: [batch, seq_len, hidden_size]
  device half* output;            // Output context: [batch, seq_len, num_heads, head_dim]

  // QKV projection weights (already projected before kernel for Phase 1)
  device const half* q;           // Query:  [batch, seq_len, num_heads, head_dim]
  device const half* k;           // Key:    [batch, seq_len, num_kv_heads, head_dim]
  device const half* v;           // Value:  [batch, seq_len, num_kv_heads, head_dim]

  // RoPE tables
  device const half* rope_cos;    // cos: [max_seq_len, head_dim/2]
  device const half* rope_sin;    // sin: [max_seq_len, head_dim/2]

  // KV cache storage
  device half* k_cache;           // K cache pages: [num_pages, block_size, num_kv_heads, head_dim] OR
                                  // K cache blocks: [num_pages, num_layers, block_size, num_kv_heads, head_dim]
  device half* v_cache;           // V cache pages: [num_pages, block_size, num_kv_heads, head_dim] OR
                                  // V cache blocks: [num_pages, num_layers, block_size, num_kv_heads, head_dim]
  device const int* page_table;   // Page table: [batch, max_blocks_per_seq]

  // Dimensions
  uint batch_size;
  uint seq_len;
  uint num_heads;
  uint num_kv_heads;
  uint head_dim;
  uint hidden_size;
  uint block_size;
  uint max_blocks_per_seq;
  uint num_layers;                // NEW: Total layers in model (for block format)
  uint layer_idx;                 // NEW: Current layer index (for block format)
  bool use_block_format;          // NEW: If true, cache is [pages, layers, ...]; if false, [pages, ...]
  float scale;                    // 1/sqrt(head_dim)
  uint position_offset;           // Starting position for RoPE
};

/**
 * @brief Apply RoPE (Rotary Positional Embeddings) to Q or K (threadgroup version)
 *
 * RoPE applies rotations to pairs of dimensions:
 * For each pair (even, odd):
 *   x_out[even] = x[even] * cos(pos * theta) - x[odd] * sin(pos * theta)
 *   x_out[odd]  = x[odd]  * cos(pos * theta) + x[even] * sin(pos * theta)
 *
 * where theta depends on dimension index.
 */
inline void apply_rope_threadgroup(
    threadgroup half* input,
    threadgroup half* output,
    device const half* cos_table,
    device const half* sin_table,
    uint token_pos,
    uint head_dim) {

  // Process pairs of dimensions
  for (uint d = 0; d < head_dim / 2; d++) {
    uint even_idx = d * 2;
    uint odd_idx = d * 2 + 1;

    // Load cos and sin from precomputed tables
    uint rope_idx = token_pos * (head_dim / 2) + d;
    float cos_val = float(cos_table[rope_idx]);
    float sin_val = float(sin_table[rope_idx]);

    // Load input values
    float x_even = float(input[even_idx]);
    float x_odd = float(input[odd_idx]);

    // Apply rotation
    float y_even = x_even * cos_val - x_odd * sin_val;
    float y_odd = x_odd * cos_val + x_even * sin_val;

    // Store output
    output[even_idx] = half(y_even);
    output[odd_idx] = half(y_odd);
  }
}

/**
 * @brief Fused attention prefill kernel
 *
 * Each threadgroup processes one attention head for one token in the sequence.
 * We process tokens sequentially due to causal masking requirements.
 *
 * Grid layout:
 *   - X: batch_size * seq_len * num_heads (one threadgroup per head per token)
 *   - Y: 1
 *   - Z: 1
 *
 * Threadgroup layout:
 *   - Threads: 256 (configurable)
 *   - Threads cooperate to compute attention over past tokens
 */
kernel void attention_prefill_fused(
    constant AttentionPrefillArgs& args [[buffer(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    threadgroup float* shared_scores [[threadgroup(0)]],
    threadgroup float* shared_reduce [[threadgroup(1)]],
    threadgroup half* shared_q [[threadgroup(2)]],
    threadgroup half* shared_k [[threadgroup(3)]]) {

  // Decode indices
  const uint total_heads = args.batch_size * args.seq_len * args.num_heads;
  if (gid >= total_heads) {
    return;
  }

  const uint linear_idx = gid;
  const uint batch_idx = linear_idx / (args.seq_len * args.num_heads);
  const uint seq_token_idx = (linear_idx / args.num_heads) % args.seq_len;
  const uint head_idx = linear_idx % args.num_heads;

  // For GQA: map query head to KV head
  const uint kv_head_idx = head_idx / (args.num_heads / args.num_kv_heads);

  // Calculate absolute position for RoPE
  const uint abs_position = args.position_offset + seq_token_idx;

  // Load and apply RoPE to Q for this token
  // Q shape: [batch, seq_len, num_heads, head_dim]
  const uint q_offset = batch_idx * args.seq_len * args.num_heads * args.head_dim +
                       seq_token_idx * args.num_heads * args.head_dim +
                       head_idx * args.head_dim;
  device const half* q_in = args.q + q_offset;

  // Each thread loads a portion of Q into shared memory
  for (uint d = lid; d < args.head_dim; d += threadgroup_size) {
    shared_q[d] = q_in[d];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Apply RoPE to Q (only thread 0 for simplicity in Phase 1)
  // Phase 2 can parallelize this
  if (lid == 0) {
    apply_rope_threadgroup(shared_q, shared_q, args.rope_cos, args.rope_sin,
                          abs_position, args.head_dim);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 1: Compute attention scores for all tokens up to and including current
  // (causal masking: can only attend to past and current tokens)
  const uint num_context_tokens = seq_token_idx + 1;
  float max_score = -INFINITY;

  // Compute scores in stripes
  const uint STRIPE_SIZE = 64;
  const uint num_stripes = (num_context_tokens + STRIPE_SIZE - 1) / STRIPE_SIZE;

  for (uint stripe_idx = 0; stripe_idx < num_stripes; stripe_idx++) {
    const uint stripe_start = stripe_idx * STRIPE_SIZE;
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, num_context_tokens);
    const uint stripe_len = stripe_end - stripe_start;

    // Each thread computes scores for a subset of context tokens
    for (uint tok = lid; tok < stripe_len; tok += threadgroup_size) {
      const uint context_token = stripe_start + tok;

      // Load K for this context token (with RoPE already applied)
      // K shape: [batch, seq_len, num_kv_heads, head_dim]
      const uint k_offset = batch_idx * args.seq_len * args.num_kv_heads * args.head_dim +
                           context_token * args.num_kv_heads * args.head_dim +
                           kv_head_idx * args.head_dim;
      device const half* k_in = args.k + k_offset;

      // Load K into shared memory and apply RoPE
      for (uint d = 0; d < args.head_dim; d++) {
        shared_k[d] = k_in[d];
      }

      // Apply RoPE to K
      apply_rope_threadgroup(shared_k, shared_k, args.rope_cos, args.rope_sin,
                            args.position_offset + context_token, args.head_dim);

      // Store K into KV cache
      // Determine block and offset
      const uint block_idx = context_token / args.block_size;
      const uint block_offset = context_token % args.block_size;
      const int page_id = args.page_table[batch_idx * args.max_blocks_per_seq + block_idx];

      if (page_id >= 0) {
        // Two formats supported:
        // 1. Stacked: [num_pages, block_size, num_kv_heads, head_dim]
        // 2. Block: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (ZERO-COPY)
        uint k_cache_offset;
        if (args.use_block_format) {
          // Block format: index with layer_idx
          k_cache_offset = page_id * args.num_layers * args.block_size * args.num_kv_heads * args.head_dim +
                           args.layer_idx * args.block_size * args.num_kv_heads * args.head_dim +
                           block_offset * args.num_kv_heads * args.head_dim +
                           kv_head_idx * args.head_dim;
        } else {
          // Stacked format (legacy)
          k_cache_offset = page_id * args.block_size * args.num_kv_heads * args.head_dim +
                           block_offset * args.num_kv_heads * args.head_dim +
                           kv_head_idx * args.head_dim;
        }
        device half* k_cache_ptr = args.k_cache + k_cache_offset;

        for (uint d = 0; d < args.head_dim; d++) {
          k_cache_ptr[d] = shared_k[d];
        }
      }

      // Compute Q @ K^T (dot product)
      float score = 0.0f;
      for (uint d = 0; d < args.head_dim; d++) {
        float q_val = float(shared_q[d]);
        float k_val = float(shared_k[d]);
        score += q_val * k_val;
      }

      // Scale by 1/sqrt(d_k)
      score *= args.scale;

      shared_scores[tok] = score;
      max_score = max(max_score, score);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Reduce to find max score across threadgroup
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

  // Phase 2: Compute exp(score - max) and sum
  float sum_exp = 0.0f;

  for (uint stripe_idx = 0; stripe_idx < num_stripes; stripe_idx++) {
    const uint stripe_start = stripe_idx * STRIPE_SIZE;
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, num_context_tokens);
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
  for (uint stripe_idx = 0; stripe_idx < num_stripes; stripe_idx++) {
    const uint stripe_start = stripe_idx * STRIPE_SIZE;
    const uint stripe_end = min(stripe_start + STRIPE_SIZE, num_context_tokens);
    const uint stripe_len = stripe_end - stripe_start;

    for (uint tok = lid; tok < stripe_len; tok += threadgroup_size) {
      shared_scores[tok] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Phase 4: Compute context = softmax(scores) @ V
  // Also store V into KV cache
  for (uint d = lid; d < args.head_dim; d += threadgroup_size) {
    float accum = 0.0f;

    for (uint context_token = 0; context_token < num_context_tokens; context_token++) {
      // Load V for this context token
      // V shape: [batch, seq_len, num_kv_heads, head_dim]
      const uint v_offset = batch_idx * args.seq_len * args.num_kv_heads * args.head_dim +
                           context_token * args.num_kv_heads * args.head_dim +
                           kv_head_idx * args.head_dim;
      device const half* v_in = args.v + v_offset;

      half v_val = v_in[d];

      // Store V into KV cache (first token only to avoid redundant writes)
      if (context_token == seq_token_idx) {
        const uint block_idx = context_token / args.block_size;
        const uint block_offset = context_token % args.block_size;
        const int page_id = args.page_table[batch_idx * args.max_blocks_per_seq + block_idx];

        if (page_id >= 0) {
          // Two formats supported (same as K cache):
          // 1. Stacked: [num_pages, block_size, num_kv_heads, head_dim]
          // 2. Block: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (ZERO-COPY)
          uint v_cache_offset;
          if (args.use_block_format) {
            // Block format: index with layer_idx
            v_cache_offset = page_id * args.num_layers * args.block_size * args.num_kv_heads * args.head_dim +
                             args.layer_idx * args.block_size * args.num_kv_heads * args.head_dim +
                             block_offset * args.num_kv_heads * args.head_dim +
                             kv_head_idx * args.head_dim;
          } else {
            // Stacked format (legacy)
            v_cache_offset = page_id * args.block_size * args.num_kv_heads * args.head_dim +
                             block_offset * args.num_kv_heads * args.head_dim +
                             kv_head_idx * args.head_dim;
          }
          device half* v_cache_ptr = args.v_cache + v_cache_offset;
          v_cache_ptr[d] = v_val;
        }
      }

      // Accumulate: output[d] += attn_weight * V[context_token, d]
      const uint score_idx = context_token % STRIPE_SIZE;
      const float attn_weight = shared_scores[score_idx];
      accum += attn_weight * float(v_val);
    }

    // Write output context
    const uint out_offset = batch_idx * args.seq_len * args.num_heads * args.head_dim +
                           seq_token_idx * args.num_heads * args.head_dim +
                           head_idx * args.head_dim;
    args.output[out_offset + d] = half(accum);
  }
}
