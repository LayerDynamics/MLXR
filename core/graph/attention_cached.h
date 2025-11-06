/**
 * @file attention_cached.h
 * @brief Attention layer with KV cache support
 *
 * Provides optimized attention implementation with:
 * - Paged KV cache integration
 * - Prefill and decode paths
 * - Support for multi-sequence batching
 */

#pragma once

#include <memory>
#include <vector>

#include "../runtime/kv/pager.h"
#include "layers.h"

namespace mlxr {
namespace graph {

/**
 * @brief Attention layer with KV cache
 *
 * Enhanced attention layer that uses paged KV cache for efficient
 * autoregressive generation. Supports both prefill (first tokens)
 * and decode (subsequent tokens) paths.
 */
class CachedAttention {
 public:
  /**
   * @brief Construct cached attention layer
   * @param hidden_size Hidden dimension
   * @param num_heads Number of attention heads
   * @param num_kv_heads Number of KV heads (for GQA)
   * @param max_seq_len Maximum sequence length
   * @param layer_idx Layer index in the model
   * @param pager KV cache pager (nullptr to disable caching)
   */
  CachedAttention(int hidden_size, int num_heads, int num_kv_heads,
                  int max_seq_len, int layer_idx,
                  std::shared_ptr<runtime::kv::Pager> pager = nullptr);

  /**
   * @brief Forward pass with KV caching
   * @param x Input tensor [batch, seq_len, hidden_size]
   * @param seq_id Sequence ID for KV cache lookup
   * @param start_pos Starting position in sequence (for decode)
   * @param mask Optional attention mask
   * @return Attention output [batch, seq_len, hidden_size]
   */
  Tensor forward(const Tensor& x, int seq_id = -1, int start_pos = 0,
                 const Tensor* mask = nullptr);

  /**
   * @brief Prefill forward pass (process multiple tokens at once)
   * @param x Input tensor [batch, seq_len, hidden_size]
   * @param seq_id Sequence ID
   * @param mask Optional causal mask
   * @return Attention output
   */
  Tensor forward_prefill(const Tensor& x, int seq_id,
                         const Tensor* mask = nullptr);

  /**
   * @brief Decode forward pass (process one token at a time)
   * @param x Input tensor [batch, 1, hidden_size]
   * @param seq_id Sequence ID
   * @param pos Current position in sequence
   * @return Attention output
   */
  Tensor forward_decode(const Tensor& x, int seq_id, int pos);

  /**
   * @brief Clear KV cache for a sequence
   * @param seq_id Sequence ID
   */
  void clear_cache(int seq_id);

  /**
   * @brief Get underlying attention layer
   */
  Attention& attention() { return attention_; }
  const Attention& attention() const { return attention_; }

  /**
   * @brief Get layer index
   */
  int layer_idx() const { return layer_idx_; }

 private:
  /**
   * @brief Store K, V tensors into KV cache
   * @param k Key tensor [batch, seq_len, num_kv_heads, head_dim]
   * @param v Value tensor [batch, seq_len, num_kv_heads, head_dim]
   * @param seq_id Sequence ID
   * @param start_pos Starting position
   */
  void store_kv(const Tensor& k, const Tensor& v, int seq_id, int start_pos);

  /**
   * @brief Load K, V tensors from KV cache
   * @param seq_id Sequence ID
   * @param seq_len Sequence length to load
   * @return Pair of (K, V) tensors
   */
  std::pair<Tensor, Tensor> load_kv(int seq_id, int seq_len);

  /**
   * @brief Check if KV cache is enabled
   */
  bool is_cache_enabled() const { return pager_ != nullptr; }

  Attention attention_;
  int hidden_size_;
  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
  int layer_idx_;
  std::shared_ptr<runtime::kv::Pager> pager_;
};

/**
 * @brief Transformer block with cached attention
 *
 * Enhanced transformer block using CachedAttention instead of
 * regular Attention layer.
 */
class CachedTransformerBlock {
 public:
  /**
   * @brief Construct cached transformer block
   * @param hidden_size Hidden dimension
   * @param num_heads Number of attention heads
   * @param num_kv_heads Number of KV heads (for GQA)
   * @param intermediate_size MLP intermediate dimension
   * @param max_seq_len Maximum sequence length
   * @param layer_idx Layer index
   * @param pager KV cache pager
   * @param norm_eps RMSNorm epsilon
   */
  CachedTransformerBlock(int hidden_size, int num_heads, int num_kv_heads,
                         int intermediate_size, int max_seq_len, int layer_idx,
                         std::shared_ptr<runtime::kv::Pager> pager,
                         float norm_eps = 1e-6f);

  /**
   * @brief Forward pass with KV caching
   * @param x Input tensor
   * @param seq_id Sequence ID
   * @param start_pos Starting position
   * @param mask Optional mask
   * @return Output tensor
   */
  Tensor forward(const Tensor& x, int seq_id = -1, int start_pos = 0,
                 const Tensor* mask = nullptr);

  /**
   * @brief Get cached attention layer
   */
  CachedAttention& attention() { return attention_; }
  const CachedAttention& attention() const { return attention_; }

  /**
   * @brief Get MLP layer
   */
  MLP& mlp() { return mlp_; }
  const MLP& mlp() const { return mlp_; }

  /**
   * @brief Get input layernorm
   */
  RMSNorm& input_layernorm() { return input_layernorm_; }
  const RMSNorm& input_layernorm() const { return input_layernorm_; }

  /**
   * @brief Get post-attention layernorm
   */
  RMSNorm& post_attention_layernorm() { return post_attention_layernorm_; }
  const RMSNorm& post_attention_layernorm() const {
    return post_attention_layernorm_;
  }

 private:
  int hidden_size_;
  RMSNorm input_layernorm_;
  CachedAttention attention_;
  RMSNorm post_attention_layernorm_;
  MLP mlp_;
};

}  // namespace graph
}  // namespace mlxr
