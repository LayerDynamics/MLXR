/**
 * @file layers.cpp
 * @brief Implementation of neural network layers for Llama models
 */

#include "layers.h"

#include <cmath>
#include <stdexcept>

#include "mlx/mlx.h"
#include "model.h"  // For KVCache definition

#ifdef USE_CUSTOM_KERNELS
#include "primitives/rmsnorm_primitive.h"
#endif

namespace mlxr {
namespace graph {

// ============================================================================
// RMSNorm Implementation
// ============================================================================

RMSNorm::RMSNorm(int dim, float eps) : dim_(dim), eps_(eps) {
  // Initialize weight to ones
  weight_ = mlxr::graph::ones({dim}, mlx::core::float32);
}

Tensor RMSNorm::forward(const Tensor& x) {
#ifdef USE_CUSTOM_KERNELS
  // Use MLX Primitive-based custom Metal kernel with proper buffer access
  auto result_arr =
      mlxr::kernels::rmsnorm_fused(x.array(), weight_.array(), eps_);
  return Tensor(result_arr);
#else
  // Fallback to MLX implementation
  // RMS normalization: x * rsqrt(mean(x^2) + eps) * weight
  auto x_arr = x.array();

  // Compute x^2
  auto x_sq = mlx::core::multiply(x_arr, x_arr);

  // Compute mean over last dimension
  std::vector<int> axes_vec = {-1};
  auto mean_sq = mlx::core::mean(x_sq, axes_vec, /*keepdims=*/true);

  // Compute rsqrt(mean(x^2) + eps)
  auto rms = mlx::core::rsqrt(mlx::core::add(mean_sq, mlx::core::array(eps_)));

  // Normalize: x * rms * weight
  auto normalized = mlx::core::multiply(x_arr, rms);
  auto result = mlx::core::multiply(normalized, weight_.array());

  return Tensor(result);
#endif
}

Tensor& RMSNorm::weight() { return weight_; }

const Tensor& RMSNorm::weight() const { return weight_; }

// ============================================================================
// Linear Layer Implementation
// ============================================================================

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
  // Initialize weight with Xavier/Glorot initialization
  // weight shape: [out_features, in_features]
  // Xavier: W ~ Uniform(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in +
  // fan_out)))
  float limit = std::sqrt(6.0f / (in_features + out_features));

  // Use MLX random uniform initialization
  auto weight_arr = mlx::core::random::uniform(
      -limit, limit, {out_features, in_features}, mlx::core::float32);
  weight_ = Tensor(weight_arr);

  if (has_bias_) {
    // Initialize bias to zeros (standard practice)
    bias_ = mlxr::graph::zeros({out_features}, mlx::core::float32);
  }
}

Tensor Linear::forward(const Tensor& x) {
  // Compute y = xW^T + b
  // x shape: [..., in_features]
  // weight shape: [out_features, in_features]
  // output shape: [..., out_features]

  auto result = matmul(x, weight_.transpose());

  if (has_bias_) {
    result = result + bias_;
  }

  return result;
}

Tensor& Linear::weight() { return weight_; }

const Tensor& Linear::weight() const { return weight_; }

Tensor* Linear::bias() { return has_bias_ ? &bias_ : nullptr; }

const Tensor* Linear::bias() const { return has_bias_ ? &bias_ : nullptr; }

// ============================================================================
// Rotary Embedding Implementation
// ============================================================================

RotaryEmbedding::RotaryEmbedding(int dim, int max_seq_len, float base)
    : dim_(dim), max_seq_len_(max_seq_len), base_(base) {
  if (dim % 2 != 0) {
    throw std::invalid_argument("RotaryEmbedding dimension must be even");
  }
  compute_freqs();
}

void RotaryEmbedding::compute_freqs() {
  // Compute frequency bands
  // freqs = 1.0 / (base^(torch.arange(0, dim, 2) / dim))
  std::vector<float> freqs_data;
  for (int i = 0; i < dim_ / 2; ++i) {
    float freq = 1.0f / std::pow(base_, (2.0f * i) / dim_);
    freqs_data.push_back(freq);
  }

  auto freqs = from_data(freqs_data.data(), {dim_ / 2}, mlx::core::float32);

  // Compute position indices: [0, 1, 2, ..., max_seq_len-1]
  std::vector<float> pos_data;
  for (int i = 0; i < max_seq_len_; ++i) {
    pos_data.push_back(static_cast<float>(i));
  }
  auto positions =
      from_data(pos_data.data(), {max_seq_len_}, mlx::core::float32);

  // Compute outer product: positions[:, None] * freqs[None, :]
  // Result shape: [max_seq_len, dim/2]
  auto pos_reshaped = positions.reshape({max_seq_len_, 1});
  auto freqs_reshaped = freqs.reshape({1, dim_ / 2});

  auto angles_arr =
      mlx::core::multiply(pos_reshaped.array(), freqs_reshaped.array());
  Tensor angles(angles_arr);

  // Compute cos and sin
  cos_cached_ = Tensor(mlx::core::cos(angles.array()));
  sin_cached_ = Tensor(mlx::core::sin(angles.array()));
}

std::pair<Tensor, Tensor> RotaryEmbedding::forward(const Tensor& q,
                                                   const Tensor& k,
                                                   int offset) {
  // q, k shape: [batch, seq_len, num_heads, head_dim]
  auto q_shape = q.shape();
  auto k_shape = k.shape();

  if (q_shape.size() != 4 || k_shape.size() != 4) {
    throw std::invalid_argument(
        "RotaryEmbedding expects 4D tensors [batch, seq_len, num_heads, "
        "head_dim]");
  }

  // Extract dimensions (batch and num_heads not used in computation but kept
  // for clarity)
  (void)q_shape[0];  // batch - suppress unused warning
  int seq_len = q_shape[1];
  (void)q_shape[2];  // num_heads - suppress unused warning
  int head_dim = q_shape[3];

  if (head_dim != dim_) {
    throw std::invalid_argument("head_dim must match RotaryEmbedding dim");
  }

  // Extract cos and sin for current positions
  // Shape: [seq_len, dim/2]
  auto cos_slice = mlx::core::slice(cos_cached_.array(), {offset, 0},
                                    {offset + seq_len, dim_ / 2}, {1, 1});
  auto sin_slice = mlx::core::slice(sin_cached_.array(), {offset, 0},
                                    {offset + seq_len, dim_ / 2}, {1, 1});

  // Reshape for broadcasting: [1, seq_len, 1, dim/2]
  Tensor cos_for_broadcast(
      mlx::core::reshape(cos_slice, {1, seq_len, 1, dim_ / 2}));
  Tensor sin_for_broadcast(
      mlx::core::reshape(sin_slice, {1, seq_len, 1, dim_ / 2}));

  // Split q and k into two halves for rotation
  // q1, q2: [batch, seq_len, num_heads, head_dim/2]
  std::vector<int> split_indices = {dim_ / 2};
  auto q_splits = split(q, split_indices, 3);  // Split along last dimension
  auto k_splits = split(k, split_indices, 3);

  Tensor q1 = q_splits[0];
  Tensor q2 = q_splits[1];
  Tensor k1 = k_splits[0];
  Tensor k2 = k_splits[1];

  // Apply rotation: rotate_half(x) = [-x2, x1]
  // q_rotated = q1 * cos - q2 * sin, q2 * cos + q1 * sin
  auto q1_cos = q1 * cos_for_broadcast;
  auto q2_sin = q2 * sin_for_broadcast;
  auto q2_cos = q2 * cos_for_broadcast;
  auto q1_sin = q1 * sin_for_broadcast;

  auto q_rot_1 = q1_cos - q2_sin;
  auto q_rot_2 = q2_cos + q1_sin;

  auto k1_cos = k1 * cos_for_broadcast;
  auto k2_sin = k2 * sin_for_broadcast;
  auto k2_cos = k2 * cos_for_broadcast;
  auto k1_sin = k1 * sin_for_broadcast;

  auto k_rot_1 = k1_cos - k2_sin;
  auto k_rot_2 = k2_cos + k1_sin;

  // Concatenate back
  auto q_rotated = concatenate({q_rot_1, q_rot_2}, 3);
  auto k_rotated = concatenate({k_rot_1, k_rot_2}, 3);

  return {q_rotated, k_rotated};
}

// ============================================================================
// Attention Implementation
// ============================================================================

Attention::Attention(int hidden_size, int num_heads, int max_seq_len,
                     int num_kv_heads)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads < 0 ? num_heads
                                     : num_kv_heads),  // Default to MHA
      head_dim_(hidden_size / num_heads),
      q_proj_(hidden_size, hidden_size, false),
      k_proj_(hidden_size,
              (num_kv_heads < 0 ? num_heads : num_kv_heads) *
                  (hidden_size / num_heads),
              false),  // GQA
      v_proj_(hidden_size,
              (num_kv_heads < 0 ? num_heads : num_kv_heads) *
                  (hidden_size / num_heads),
              false),  // GQA
      o_proj_(hidden_size, hidden_size, false),
      rope_(head_dim_, max_seq_len) {
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument("hidden_size must be divisible by num_heads");
  }
  if (num_heads % num_kv_heads_ != 0) {
    throw std::invalid_argument(
        "num_heads must be divisible by num_kv_heads for GQA");
  }
}

Tensor Attention::forward(const Tensor& x, const Tensor* mask,
                          KVCache* kv_cache, int layer_idx) {
  // x shape: [batch, seq_len, hidden_size]
  auto x_shape = x.shape();
  int batch = x_shape[0];
  int seq_len = x_shape[1];

  // Project to Q, K, V
  auto q = q_proj_.forward(x);  // [batch, seq_len, hidden_size]
  auto k = k_proj_.forward(x);  // [batch, seq_len, num_kv_heads * head_dim]
  auto v = v_proj_.forward(x);  // [batch, seq_len, num_kv_heads * head_dim]

  // Reshape: Q uses num_heads, K/V use num_kv_heads
  q = q.reshape({batch, seq_len, num_heads_, head_dim_});
  k = k.reshape({batch, seq_len, num_kv_heads_, head_dim_});
  v = v.reshape({batch, seq_len, num_kv_heads_, head_dim_});

  // Apply rotary embeddings with offset for cached positions
  int rope_offset =
      (kv_cache && kv_cache->is_initialized()) ? kv_cache->cached_length : 0;
  auto [q_rot, k_rot] = rope_.forward(q, k, rope_offset);

  // Transpose: Q to [batch, num_heads, seq_len, head_dim]
  //            K/V to [batch, num_kv_heads, seq_len, head_dim]
  q_rot = q_rot.transpose({0, 2, 1, 3});
  k_rot = k_rot.transpose({0, 2, 1, 3});
  v = v.transpose({0, 2, 1, 3});

  // For GQA: Repeat K/V heads to match number of Q heads
  // Each KV head is repeated (num_heads / num_kv_heads) times
  Tensor k_for_attn = k_rot;
  Tensor v_for_attn = v;

  if (num_kv_heads_ < num_heads_) {
    // GQA: repeat each KV head
    int repeat_factor = num_heads_ / num_kv_heads_;

    // Method: repeat_interleave along head dimension
    // k_rot shape: [batch, num_kv_heads, seq_len, head_dim]
    // Result shape: [batch, num_heads, seq_len, head_dim]
    auto k_arr = k_rot.array();
    auto v_arr = v.array();

    // Repeat each head: [b, kv_h, s, d] -> [b, kv_h*repeat, s, d]
    // IMPORTANT: Force evaluation after repeat to ensure contiguous memory
    // layout This prevents "Cannot reshape" errors in subsequent operations
    auto k_repeated = mlx::core::repeat(k_arr, repeat_factor, 1);
    auto v_repeated = mlx::core::repeat(v_arr, repeat_factor, 1);
    mlx::core::eval(k_repeated);
    mlx::core::eval(v_repeated);
    k_for_attn = Tensor(k_repeated);
    v_for_attn = Tensor(v_repeated);
  }

  // Handle KV cache (cache already-repeated K/V for efficiency)
  if (kv_cache != nullptr) {
    // Ensure cache has space for this layer
    if (layer_idx >= static_cast<int>(kv_cache->layer_caches.size())) {
      kv_cache->layer_caches.resize(layer_idx + 1);
    }

    auto& layer_cache = kv_cache->layer_caches[layer_idx];

    if (kv_cache->is_initialized() && !layer_cache.first.empty()) {
      // Cache exists - concatenate new K,V with cached K,V
      // Cached shape: [batch, num_heads, cached_length, head_dim]
      // New shape: [batch, num_heads, seq_len, head_dim]

      // IMPORTANT: Evaluate cached and new tensors before concatenation
      // This ensures both tensors are contiguous, preventing reshape errors
      auto cached_k = layer_cache.first.array();
      auto cached_v = layer_cache.second.array();
      auto new_k = k_for_attn.array();
      auto new_v = v_for_attn.array();
      mlx::core::eval(cached_k);
      mlx::core::eval(cached_v);
      mlx::core::eval(new_k);
      mlx::core::eval(new_v);

      k_for_attn = concatenate({Tensor(cached_k), Tensor(new_k)}, /*axis=*/2);
      v_for_attn = concatenate({Tensor(cached_v), Tensor(new_v)}, /*axis=*/2);
    }

    // Update cache with concatenated K,V (already repeated for GQA)
    layer_cache.first = k_for_attn;
    layer_cache.second = v_for_attn;
  }

  // Compute attention scores: Q @ K^T / sqrt(head_dim)
  // k_for_attn transposed: [batch, num_heads, head_dim, total_seq_len]
  auto k_rot_t = k_for_attn.transpose({0, 1, 3, 2});
  auto scores =
      matmul(q_rot, k_rot_t);  // [batch, num_heads, seq_len, total_seq_len]

  // Scale by sqrt(head_dim)
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  scores = scores * scale;

  // Apply mask if provided
  if (mask != nullptr) {
    // mask shape should be broadcastable to scores shape
    scores = scores + *mask;
  }

  // Apply softmax
  auto attn_weights = Tensor(mlx::core::softmax(scores.array(), /*axis=*/-1));

  // Apply attention to values: attn_weights @ V
  // [batch, num_heads, seq_len, total_seq_len] @ [batch, num_heads,
  // total_seq_len, head_dim]
  // -> [batch, num_heads, seq_len, head_dim]
  auto attn_output = matmul(attn_weights, v_for_attn);

  // Transpose back: [batch, seq_len, num_heads, head_dim]
  attn_output = attn_output.transpose({0, 2, 1, 3});

  // Reshape to [batch, seq_len, hidden_size]
  attn_output = attn_output.reshape({batch, seq_len, hidden_size_});

  // Output projection
  auto output = o_proj_.forward(attn_output);

  return output;
}

Linear& Attention::q_proj() { return q_proj_; }
Linear& Attention::k_proj() { return k_proj_; }
Linear& Attention::v_proj() { return v_proj_; }
Linear& Attention::o_proj() { return o_proj_; }
RotaryEmbedding& Attention::rope() { return rope_; }

// ============================================================================
// MLP Implementation
// ============================================================================

MLP::MLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      gate_proj_(hidden_size, intermediate_size, false),
      up_proj_(hidden_size, intermediate_size, false),
      down_proj_(intermediate_size, hidden_size, false) {}

Tensor MLP::forward(const Tensor& x) {
  // SwiGLU activation: swish(gate(x)) * up(x)
  // where swish(x) = x * sigmoid(x)

  // Compute gate and up projections
  auto gate = gate_proj_.forward(x);  // [batch, seq_len, intermediate_size]
  auto up = up_proj_.forward(x);      // [batch, seq_len, intermediate_size]

  // Apply SiLU (Swish) activation to gate: x * sigmoid(x)
  auto gate_arr = gate.array();
  auto gate_sigmoid = mlx::core::sigmoid(gate_arr);
  auto gate_silu = mlx::core::multiply(gate_arr, gate_sigmoid);

  // Multiply with up projection
  auto activated = Tensor(mlx::core::multiply(gate_silu, up.array()));

  // Down projection
  auto output = down_proj_.forward(activated);

  return output;
}

Linear& MLP::gate_proj() { return gate_proj_; }
Linear& MLP::up_proj() { return up_proj_; }
Linear& MLP::down_proj() { return down_proj_; }

// ============================================================================
// TransformerBlock Implementation
// ============================================================================

TransformerBlock::TransformerBlock(int hidden_size, int num_heads,
                                   int intermediate_size, int max_seq_len,
                                   float norm_eps, int num_kv_heads)
    : hidden_size_(hidden_size),
      input_layernorm_(hidden_size, norm_eps),
      attention_(hidden_size, num_heads, max_seq_len, num_kv_heads),
      post_attention_layernorm_(hidden_size, norm_eps),
      mlp_(hidden_size, intermediate_size) {}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor* mask,
                                 KVCache* kv_cache, int layer_idx) {
  // Pre-norm architecture (like Llama)
  // x = x + attention(norm(x))
  // x = x + mlp(norm(x))

  // Attention block with residual
  auto normed = input_layernorm_.forward(x);
  auto attn_out = attention_.forward(normed, mask, kv_cache, layer_idx);
  auto x_after_attn = x + attn_out;

  // MLP block with residual
  normed = post_attention_layernorm_.forward(x_after_attn);
  auto mlp_out = mlp_.forward(normed);
  auto output = x_after_attn + mlp_out;

  return output;
}

Attention& TransformerBlock::attention() { return attention_; }
MLP& TransformerBlock::mlp() { return mlp_; }
RMSNorm& TransformerBlock::input_layernorm() { return input_layernorm_; }
RMSNorm& TransformerBlock::post_attention_layernorm() {
  return post_attention_layernorm_;
}

}  // namespace graph
}  // namespace mlxr
