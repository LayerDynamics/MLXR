/**
 * @file layers.h
 * @brief Basic neural network layers for Llama models
 *
 * Provides layer implementations using MLX operations.
 */

#pragma once

#include <memory>
#include <vector>

#include "tensor.h"

namespace mlxr {
namespace graph {

/**
 * @brief RMS Normalization layer
 *
 * Implements Root Mean Square Layer Normalization as used in Llama models.
 */
class RMSNorm {
 public:
  /**
   * @brief Construct RMSNorm layer
   * @param dim Hidden dimension
   * @param eps Epsilon for numerical stability (default: 1e-6)
   */
  explicit RMSNorm(int dim, float eps = 1e-6f);

  /**
   * @brief Apply RMS normalization
   * @param x Input tensor
   * @return Normalized tensor
   */
  Tensor forward(const Tensor& x);

  /**
   * @brief Get weight parameter
   */
  Tensor& weight();
  const Tensor& weight() const;

 private:
  int dim_;
  float eps_;
  Tensor weight_;
};

/**
 * @brief Linear (fully connected) layer
 *
 * Implements y = xW^T + b
 */
class Linear {
 public:
  /**
   * @brief Construct linear layer
   * @param in_features Input feature dimension
   * @param out_features Output feature dimension
   * @param bias Whether to include bias term (default: false)
   */
  Linear(int in_features, int out_features, bool bias = false);

  /**
   * @brief Apply linear transformation
   * @param x Input tensor [..., in_features]
   * @return Output tensor [..., out_features]
   */
  Tensor forward(const Tensor& x);

  /**
   * @brief Get weight parameter
   */
  Tensor& weight();
  const Tensor& weight() const;

  /**
   * @brief Get bias parameter (if exists)
   */
  Tensor* bias();
  const Tensor* bias() const;

 private:
  int in_features_;
  int out_features_;
  bool has_bias_;
  Tensor weight_;
  Tensor bias_;
};

/**
 * @brief Rotary Position Embedding (RoPE)
 *
 * Implements rotary position embeddings as used in Llama models.
 */
class RotaryEmbedding {
 public:
  /**
   * @brief Construct rotary embedding
   * @param dim Dimension (must be even)
   * @param max_seq_len Maximum sequence length
   * @param base Base for frequency calculation (default: 10000)
   */
  RotaryEmbedding(int dim, int max_seq_len, float base = 10000.0f);

  /**
   * @brief Apply rotary embeddings
   * @param q Query tensor [batch, seq_len, num_heads, head_dim]
   * @param k Key tensor [batch, seq_len, num_heads, head_dim]
   * @param offset Position offset for cached sequences
   * @return Tuple of (rotated_q, rotated_k)
   */
  std::pair<Tensor, Tensor> forward(const Tensor& q, const Tensor& k,
                                    int offset = 0);

 private:
  void compute_freqs();

  int dim_;
  int max_seq_len_;
  float base_;
  Tensor cos_cached_;
  Tensor sin_cached_;
};

/**
 * @brief Multi-Head Attention layer
 *
 * Implements multi-head self-attention as used in Llama models.
 */
class Attention {
 public:
  /**
   * @brief Construct attention layer
   * @param hidden_size Hidden dimension
   * @param num_heads Number of attention heads
   * @param max_seq_len Maximum sequence length
   */
  Attention(int hidden_size, int num_heads, int max_seq_len);

  /**
   * @brief Apply attention
   * @param x Input tensor [batch, seq_len, hidden_size]
   * @param mask Optional attention mask
   * @return Attention output [batch, seq_len, hidden_size]
   */
  Tensor forward(const Tensor& x, const Tensor* mask = nullptr);

  /**
   * @brief Get query projection layer
   */
  Linear& q_proj();

  /**
   * @brief Get key projection layer
   */
  Linear& k_proj();

  /**
   * @brief Get value projection layer
   */
  Linear& v_proj();

  /**
   * @brief Get output projection layer
   */
  Linear& o_proj();

  /**
   * @brief Get rotary embedding
   */
  RotaryEmbedding& rope();

 private:
  int hidden_size_;
  int num_heads_;
  int head_dim_;

  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear o_proj_;
  RotaryEmbedding rope_;
};

/**
 * @brief Multi-Layer Perceptron (MLP) with SwiGLU activation
 *
 * Implements the feed-forward network used in Llama models.
 */
class MLP {
 public:
  /**
   * @brief Construct MLP layer
   * @param hidden_size Hidden dimension
   * @param intermediate_size Intermediate (expanded) dimension
   */
  MLP(int hidden_size, int intermediate_size);

  /**
   * @brief Apply MLP with SwiGLU activation
   * @param x Input tensor [batch, seq_len, hidden_size]
   * @return Output tensor [batch, seq_len, hidden_size]
   */
  Tensor forward(const Tensor& x);

  /**
   * @brief Get gate projection
   */
  Linear& gate_proj();

  /**
   * @brief Get up projection
   */
  Linear& up_proj();

  /**
   * @brief Get down projection
   */
  Linear& down_proj();

 private:
  int hidden_size_;
  int intermediate_size_;

  Linear gate_proj_;
  Linear up_proj_;
  Linear down_proj_;
};

/**
 * @brief Single transformer block (decoder layer)
 *
 * Combines attention, MLP, and layer normalization.
 */
class TransformerBlock {
 public:
  /**
   * @brief Construct transformer block
   * @param hidden_size Hidden dimension
   * @param num_heads Number of attention heads
   * @param intermediate_size MLP intermediate dimension
   * @param max_seq_len Maximum sequence length
   * @param norm_eps RMSNorm epsilon
   */
  TransformerBlock(int hidden_size, int num_heads, int intermediate_size,
                   int max_seq_len, float norm_eps = 1e-6f);

  /**
   * @brief Apply transformer block
   * @param x Input tensor [batch, seq_len, hidden_size]
   * @param mask Optional attention mask
   * @return Output tensor [batch, seq_len, hidden_size]
   */
  Tensor forward(const Tensor& x, const Tensor* mask = nullptr);

  /**
   * @brief Get attention sublayer
   */
  Attention& attention();

  /**
   * @brief Get MLP sublayer
   */
  MLP& mlp();

  /**
   * @brief Get input layer norm
   */
  RMSNorm& input_layernorm();

  /**
   * @brief Get post-attention layer norm
   */
  RMSNorm& post_attention_layernorm();

 private:
  int hidden_size_;

  RMSNorm input_layernorm_;
  Attention attention_;
  RMSNorm post_attention_layernorm_;
  MLP mlp_;
};

}  // namespace graph
}  // namespace mlxr
