/**
 * @file attention_cached.cpp
 * @brief Implementation of attention with KV cache support
 */

#include "attention_cached.h"

#include <iostream>

#ifdef USE_CUSTOM_KERNELS
#include "../kernels/primitives/attention_decode_primitive.h"
#include "../kernels/primitives/attention_prefill_primitive.h"
#endif

namespace mlxr {
namespace graph {

// ============================================================================
// CachedAttention Implementation
// ============================================================================

CachedAttention::CachedAttention(int hidden_size, int num_heads,
                                 int num_kv_heads, int max_seq_len,
                                 int layer_idx,
                                 std::shared_ptr<runtime::kv::Pager> pager)
    : attention_(hidden_size, num_heads, max_seq_len),
      hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(hidden_size / num_heads),
      layer_idx_(layer_idx),
      pager_(pager) {
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument("hidden_size must be divisible by num_heads");
  }
}

Tensor CachedAttention::forward(const Tensor& x, int seq_id, int start_pos,
                                const Tensor* mask) {
  // Determine if this is prefill or decode based on input shape
  auto x_shape = x.shape();
  int seq_len = x_shape[1];

  if (seq_len > 1 || start_pos == 0) {
    // Prefill path: multiple tokens
    return forward_prefill(x, seq_id, mask);
  } else {
    // Decode path: single token
    return forward_decode(x, seq_id, start_pos);
  }
}

Tensor CachedAttention::forward_prefill(const Tensor& x, int seq_id,
                                        const Tensor* mask) {
  auto x_shape = x.shape();
  int batch = x_shape[0];
  int seq_len = x_shape[1];

  // Project to Q, K, V
  auto q = attention_.q_proj().forward(x);
  auto k = attention_.k_proj().forward(x);
  auto v = attention_.v_proj().forward(x);

  // Reshape to [batch, seq_len, num_heads, head_dim]
  q = q.reshape({batch, seq_len, num_heads_, head_dim_});
  k = k.reshape({batch, seq_len, num_kv_heads_, head_dim_});
  v = v.reshape({batch, seq_len, num_kv_heads_, head_dim_});

#ifdef USE_CUSTOM_KERNELS
  // Use custom Metal kernel for fused RoPE + attention + KV storage
  if (is_cache_enabled() && seq_id >= 0) {
    std::cout << "[AttentionCached] PREFILL: Using Metal kernel path for layer "
              << layer_idx_ << ", seq_len=" << seq_len << std::endl;

    runtime::kv::Sequence* seq = pager_->get_sequence(seq_id);
    if (!seq) {
      throw std::runtime_error("Sequence not found: " + std::to_string(seq_id));
    }

    // Ensure sequence has enough blocks allocated
    pager_->allocate_blocks_for_sequence(seq_id, seq_len);

    // Get page table from sequence
    const auto& page_table_vec = seq->page_table();
    int max_blocks = page_table_vec.size();
    int block_size = pager_->arena().config().block_size_tokens;

    // Build page table array [batch, max_blocks]
    auto page_table = pager_->build_page_table_array(seq_id, max_blocks);

    // ZERO-COPY: Get raw block arrays without slicing/stacking
    auto k_block_arrays = pager_->arena().get_k_block_arrays(page_table_vec);
    auto v_block_arrays = pager_->arena().get_v_block_arrays(page_table_vec);

    // Stack block arrays to create format: [pages, layers, block_size, heads,
    // dim] This creates a view that shares the original block buffers
    // (zero-copy!)
    auto k_cache_arr = mlx::core::stack(k_block_arrays, 0);
    auto v_cache_arr = mlx::core::stack(v_block_arrays, 0);

    // Get RoPE cos/sin tables
    const auto& rope_cos = attention_.rope().cos_table();
    const auto& rope_sin = attention_.rope().sin_table();

    // Get model config for num_layers
    int num_layers = pager_->arena().config().num_layers;

    // Call fused Metal kernel with block format
    // Input: x [batch, seq_len, hidden_size]
    // Q, K, V: [batch, seq_len, num_heads/num_kv_heads, head_dim]
    // RoPE tables: [max_seq_len, head_dim/2]
    // Cache: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (BLOCK
    // FORMAT) Page table: [batch, max_blocks] NOTE: Kernel indexes with
    // layer_idx and modifies blocks IN-PLACE, no write-back needed!
    auto attn_output_arr = kernels::attention_prefill_fused(
        x.array(), q.array(), k.array(), v.array(), rope_cos.array(),
        rope_sin.array(),
        k_cache_arr,  // Modified in-place by kernel
        v_cache_arr,  // Modified in-place by kernel
        page_table.array(), num_heads_, num_kv_heads_, head_dim_, hidden_size_,
        block_size, max_blocks,
        num_layers,  // NEW: total layers in model
        layer_idx_,  // NEW: current layer index
        true,        // NEW: use_block_format=true
        0            // position_offset
    );

    // No write-back needed! Metal kernel modifies block buffers in-place via
    // unified memory

    // Convert output to Tensor [batch, seq_len, num_heads, head_dim]
    auto attn_output = Tensor(attn_output_arr);

    // Reshape to [batch, seq_len, hidden_size]
    attn_output = attn_output.reshape({batch, seq_len, hidden_size_});

    // Output projection
    auto output = attention_.o_proj().forward(attn_output);

    return output;
  }
#endif

  // Fallback path (no custom kernels or cache disabled)
  std::cout
      << "[AttentionCached] PREFILL: Using MLX fallback path (cache_enabled="
      << is_cache_enabled() << ", seq_id=" << seq_id << ")" << std::endl;
  // Apply rotary embeddings
  auto [q_rot, k_rot] = attention_.rope().forward(q, k, 0);

  // Store K, V in cache if enabled
  if (is_cache_enabled() && seq_id >= 0) {
    store_kv(k_rot, v, seq_id, 0);
  }

  // Expand KV heads if using Grouped Query Attention (GQA)
  if (num_kv_heads_ < num_heads_) {
    int num_groups = num_heads_ / num_kv_heads_;
    // Repeat K and V along head dimension
    auto k_shape = k_rot.shape();
    auto v_shape = v.shape();

    // K: [batch, seq_len, num_kv_heads, head_dim] -> [batch, seq_len,
    // num_heads, head_dim]
    std::vector<Tensor> k_repeated;
    std::vector<Tensor> v_repeated;

    for (int i = 0; i < num_kv_heads_; ++i) {
      for (int j = 0; j < num_groups; ++j) {
        // Slice out one KV head and repeat
        auto k_slice = mlx::core::slice(
            k_rot.array(), {0, 0, i, 0},
            {k_shape[0], k_shape[1], i + 1, k_shape[3]}, {1, 1, 1, 1});
        auto v_slice = mlx::core::slice(
            v.array(), {0, 0, i, 0},
            {v_shape[0], v_shape[1], i + 1, v_shape[3]}, {1, 1, 1, 1});
        k_repeated.push_back(Tensor(k_slice));
        v_repeated.push_back(Tensor(v_slice));
      }
    }

    k_rot = concatenate(k_repeated, 2);  // Concatenate along head dim
    v = concatenate(v_repeated, 2);
  }

  // Transpose to [batch, num_heads, seq_len, head_dim]
  q_rot = q_rot.transpose({0, 2, 1, 3});
  k_rot = k_rot.transpose({0, 2, 1, 3});
  v = v.transpose({0, 2, 1, 3});

  // Compute attention scores: Q @ K^T / sqrt(head_dim)
  auto k_rot_t = k_rot.transpose({0, 1, 3, 2});
  auto scores = matmul(q_rot, k_rot_t);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  scores = scores * scale;

  // Apply mask if provided
  if (mask != nullptr) {
    scores = scores + *mask;
  }

  // Apply softmax
  auto attn_weights = Tensor(mlx::core::softmax(scores.array(), /*axis=*/-1));

  // Apply attention to values
  auto attn_output = matmul(attn_weights, v);

  // Transpose back: [batch, seq_len, num_heads, head_dim]
  attn_output = attn_output.transpose({0, 2, 1, 3});

  // Reshape to [batch, seq_len, hidden_size]
  attn_output = attn_output.reshape({batch, seq_len, hidden_size_});

  // Output projection
  auto output = attention_.o_proj().forward(attn_output);

  return output;
}

Tensor CachedAttention::forward_decode(const Tensor& x, int seq_id, int pos) {
  auto x_shape = x.shape();
  int batch = x_shape[0];

  // Project to Q, K, V for current token
  auto q = attention_.q_proj().forward(x);
  auto k_cur = attention_.k_proj().forward(x);
  auto v_cur = attention_.v_proj().forward(x);

  // Reshape to [batch, 1, num_heads, head_dim]
  q = q.reshape({batch, 1, num_heads_, head_dim_});
  k_cur = k_cur.reshape({batch, 1, num_kv_heads_, head_dim_});
  v_cur = v_cur.reshape({batch, 1, num_kv_heads_, head_dim_});

  // Apply rotary embeddings with position offset
  auto [q_rot, k_rot] = attention_.rope().forward(q, k_cur, pos);

#ifdef USE_CUSTOM_KERNELS
  // Use custom Metal kernel for fused paged KV access + attention
  if (is_cache_enabled() && seq_id >= 0) {
    std::cout << "[AttentionCached] DECODE: Using Metal kernel path for layer "
              << layer_idx_ << ", pos=" << pos << std::endl;

    // Store current K, V in cache first
    store_kv(k_rot, v_cur, seq_id, pos);

    runtime::kv::Sequence* seq = pager_->get_sequence(seq_id);
    if (!seq) {
      throw std::runtime_error("Sequence not found: " + std::to_string(seq_id));
    }

    // Get page table from sequence
    const auto& page_table_vec = seq->page_table();
    int max_blocks = page_table_vec.size();
    int block_size = pager_->arena().config().block_size_tokens;

    // Build page table array [batch, max_blocks]
    auto page_table = pager_->build_page_table_array(seq_id, max_blocks);

    // ZERO-COPY: Get raw block arrays without slicing/stacking
    auto k_block_arrays = pager_->arena().get_k_block_arrays(page_table_vec);
    auto v_block_arrays = pager_->arena().get_v_block_arrays(page_table_vec);

    // Stack block arrays to create format: [pages, layers, block_size, heads,
    // dim] This creates a view that shares the original block buffers
    // (zero-copy!)
    auto k_cache_arr = mlx::core::stack(k_block_arrays, 0);
    auto v_cache_arr = mlx::core::stack(v_block_arrays, 0);

    // Squeeze Q to [batch, num_heads, head_dim]
    auto q_squeezed = mlx::core::squeeze(q_rot.array(), 1);

    // Create seq_lengths array [batch] = [pos + 1]
    std::vector<int> seq_lens(batch, pos + 1);
    auto seq_lengths =
        mlx::core::array(seq_lens.data(), {batch}, mlx::core::int32);

    // Get model config for num_layers
    int num_layers = pager_->arena().config().num_layers;

    // Call fused Metal kernel with block format
    // Q: [batch, num_heads, head_dim]
    // Cache: [num_pages, num_layers, block_size, num_kv_heads, head_dim] (BLOCK
    // FORMAT) Page table: [batch, max_blocks] Seq lengths: [batch] NOTE: Kernel
    // indexes with layer_idx and modifies blocks IN-PLACE, no write-back
    // needed!
    auto attn_output_arr = kernels::attention_decode_fused(
        q_squeezed, k_cache_arr, v_cache_arr, page_table.array(), seq_lengths,
        num_heads_, num_kv_heads_, head_dim_, block_size, max_blocks,
        num_layers,  // NEW: total layers in model
        layer_idx_,  // NEW: current layer index
        true,        // NEW: use_block_format=true
        false,       // use_sliding_window
        0            // sliding_window_size
    );

    // Expand dims back: [batch, num_heads, head_dim] -> [batch, 1, num_heads,
    // head_dim]
    attn_output_arr = mlx::core::expand_dims(attn_output_arr, 1);
    auto attn_output = Tensor(attn_output_arr);

    // Reshape to [batch, 1, hidden_size]
    attn_output = attn_output.reshape({batch, 1, hidden_size_});

    // Output projection
    auto output = attention_.o_proj().forward(attn_output);

    return output;
  }
#endif

  // Fallback path (no custom kernels or cache disabled)
  std::cout
      << "[AttentionCached] DECODE: Using MLX fallback path (cache_enabled="
      << is_cache_enabled() << ", seq_id=" << seq_id << ")" << std::endl;
  // Load cached K, V if available
  Tensor k_full, v_full;

  if (is_cache_enabled() && seq_id >= 0) {
    // Store current K, V
    store_kv(k_rot, v_cur, seq_id, pos);

    // Load full K, V from cache
    auto [k_cached, v_cached] = load_kv(seq_id, pos + 1);
    k_full = k_cached;
    v_full = v_cached;
  } else {
    // No cache - just use current tokens
    k_full = k_rot;
    v_full = v_cur;
  }

  // Expand KV heads if using GQA
  if (num_kv_heads_ < num_heads_) {
    int num_groups = num_heads_ / num_kv_heads_;
    auto k_shape = k_full.shape();
    auto v_shape = v_full.shape();

    std::vector<Tensor> k_repeated;
    std::vector<Tensor> v_repeated;

    for (int i = 0; i < num_kv_heads_; ++i) {
      for (int j = 0; j < num_groups; ++j) {
        auto k_slice = mlx::core::slice(
            k_full.array(), {0, 0, i, 0},
            {k_shape[0], k_shape[1], i + 1, k_shape[3]}, {1, 1, 1, 1});
        auto v_slice = mlx::core::slice(
            v_full.array(), {0, 0, i, 0},
            {v_shape[0], v_shape[1], i + 1, v_shape[3]}, {1, 1, 1, 1});
        k_repeated.push_back(Tensor(k_slice));
        v_repeated.push_back(Tensor(v_slice));
      }
    }

    k_full = concatenate(k_repeated, 2);
    v_full = concatenate(v_repeated, 2);
  }

  // Transpose to [batch, num_heads, seq_len, head_dim]
  q_rot = q_rot.transpose({0, 2, 1, 3});
  k_full = k_full.transpose({0, 2, 1, 3});
  v_full = v_full.transpose({0, 2, 1, 3});

  // Compute attention: Q @ K^T / sqrt(head_dim)
  auto k_full_t = k_full.transpose({0, 1, 3, 2});
  auto scores = matmul(q_rot, k_full_t);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  scores = scores * scale;

  // Apply softmax
  auto attn_weights = Tensor(mlx::core::softmax(scores.array(), /*axis=*/-1));

  // Apply attention to values
  auto attn_output = matmul(attn_weights, v_full);

  // Transpose back: [batch, 1, num_heads, head_dim]
  attn_output = attn_output.transpose({0, 2, 1, 3});

  // Reshape to [batch, 1, hidden_size]
  attn_output = attn_output.reshape({batch, 1, hidden_size_});

  // Output projection
  auto output = attention_.o_proj().forward(attn_output);

  return output;
}

void CachedAttention::store_kv(const Tensor& k, const Tensor& v, int seq_id,
                               int start_pos) {
  if (!is_cache_enabled()) {
    return;
  }

  // Get sequence
  runtime::kv::Sequence* seq = pager_->get_sequence(seq_id);
  if (!seq) {
    std::cerr << "Warning: Sequence " << seq_id << " not found in pager"
              << std::endl;
    return;
  }

  auto k_shape = k.shape();
  int seq_len = k_shape[1];

  // Ensure sequence has enough blocks allocated
  int target_tokens = start_pos + seq_len;
  pager_->allocate_blocks_for_sequence(seq_id, target_tokens);

  // Store K, V into blocks
  // For each token, determine which block it belongs to and update that block
  int block_size = pager_->arena().config().block_size_tokens;

  for (int t = 0; t < seq_len; ++t) {
    int token_pos = start_pos + t;
    int block_idx = token_pos / block_size;
    int offset_in_block = token_pos % block_size;

    int block_id = seq->get_block_id(block_idx);
    if (block_id < 0) {
      continue;  // Block not allocated
    }

    runtime::kv::Block* block = pager_->get_block(block_id);
    if (!block) {
      continue;
    }

    // Extract K, V for this token: [batch, 1, num_kv_heads, head_dim]
    auto k_token = mlx::core::slice(k.array(), {0, t, 0, 0},
                                    {k_shape[0], t + 1, k_shape[2], k_shape[3]},
                                    {1, 1, 1, 1});
    auto v_token = mlx::core::slice(v.array(), {0, t, 0, 0},
                                    {k_shape[0], t + 1, k_shape[2], k_shape[3]},
                                    {1, 1, 1, 1});

    // Update block storage at [layer_idx, offset_in_block, :, :]
    // Note: This is a simplified version - full implementation would use
    // efficient in-place updates or Metal kernels for KV packing

    // For now, we'll store the entire K, V tensors in the block
    // (In production, we'd do more efficient slice updates)
    block->k_data = k;
    block->v_data = v;
    block->dirty = true;

    // Touch block for LRU
    pager_->arena().touch_block(block_id);
  }
}

std::pair<Tensor, Tensor> CachedAttention::load_kv(int seq_id, int seq_len) {
  if (!is_cache_enabled()) {
    throw std::runtime_error("KV cache not enabled");
  }

  runtime::kv::Sequence* seq = pager_->get_sequence(seq_id);
  if (!seq) {
    throw std::runtime_error("Sequence not found: " + std::to_string(seq_id));
  }

  const auto& page_table = seq->page_table();
  if (page_table.empty()) {
    throw std::runtime_error("Empty page table for sequence");
  }

  // For simplified implementation, load K, V from first block
  // Full implementation would concatenate across all blocks

  int block_id = page_table[0];
  runtime::kv::Block* block = pager_->get_block(block_id);
  if (!block) {
    throw std::runtime_error("Block not found");
  }

  // Touch block for LRU
  pager_->arena().touch_block(block_id);

  // Return K, V (simplified - should slice to seq_len)
  return {block->k_data, block->v_data};
}

void CachedAttention::clear_cache(int seq_id) {
  if (!is_cache_enabled()) {
    return;
  }

  // This would be handled by the pager when sequence is deleted
  // For now, just a placeholder
}

// ============================================================================
// CachedTransformerBlock Implementation
// ============================================================================

CachedTransformerBlock::CachedTransformerBlock(
    int hidden_size, int num_heads, int num_kv_heads, int intermediate_size,
    int max_seq_len, int layer_idx, std::shared_ptr<runtime::kv::Pager> pager,
    float norm_eps)
    : hidden_size_(hidden_size),
      input_layernorm_(hidden_size, norm_eps),
      attention_(hidden_size, num_heads, num_kv_heads, max_seq_len, layer_idx,
                 pager),
      post_attention_layernorm_(hidden_size, norm_eps),
      mlp_(hidden_size, intermediate_size) {}

Tensor CachedTransformerBlock::forward(const Tensor& x, int seq_id,
                                       int start_pos, const Tensor* mask) {
  // Pre-norm architecture (like Llama)
  // x = x + attention(norm(x))
  // x = x + mlp(norm(x))

  // Attention block with residual
  auto normed = input_layernorm_.forward(x);
  auto attn_out = attention_.forward(normed, seq_id, start_pos, mask);
  auto x_after_attn = x + attn_out;

  // MLP block with residual
  normed = post_attention_layernorm_.forward(x_after_attn);
  auto mlp_out = mlp_.forward(normed);
  auto output = x_after_attn + mlp_out;

  return output;
}

}  // namespace graph
}  // namespace mlxr
