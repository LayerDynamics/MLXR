// Copyright © 2025 MLXR Development
// MLX Primitive-based custom attention prefill kernel

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * AttentionPrefill primitive using custom fused Metal kernel
 *
 * Implements attention prefill path with paged KV cache storage:
 * 1. Apply RoPE to Q and K
 * 2. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 * 3. Apply causal masking
 * 4. Softmax with fp32 accumulation for numerical stability
 * 5. Compute context: context = softmax(scores) @ V
 * 6. Store K, V into paged KV cache for future decode steps
 *
 * Input shapes:
 *   input:      [batch, seq_len, hidden_size]
 *   q:          [batch, seq_len, num_heads, head_dim]
 *   k:          [batch, seq_len, num_kv_heads, head_dim]
 *   v:          [batch, seq_len, num_kv_heads, head_dim]
 *   rope_cos:   [max_seq_len, head_dim/2]
 *   rope_sin:   [max_seq_len, head_dim/2]
 *   k_cache:    [num_pages, block_size, num_kv_heads, head_dim]
 *   v_cache:    [num_pages, block_size, num_kv_heads, head_dim]
 *   page_table: [batch, max_blocks_per_seq] (int32)
 *
 * Output shape:
 *   context:    [batch, seq_len, num_heads, head_dim]
 *
 * Features:
 * - Fused RoPE → attention → KV storage
 * - Paged KV cache with non-contiguous memory access
 * - Grouped Query Attention (GQA) support
 * - Numerically stable softmax (fp32 accumulation)
 * - Causal masking for autoregressive generation
 * - Configurable block sizes (16, 32 tokens per block)
 */
class AttentionPrefillPrimitive : public mlx::core::Primitive {
 public:
  /**
   * Constructor
   *
   * @param stream MLX stream for execution
   * @param num_heads Number of query heads
   * @param num_kv_heads Number of key/value heads (for GQA)
   * @param head_dim Dimension per attention head
   * @param hidden_size Hidden size
   * @param block_size Number of tokens per KV cache block
   * @param max_blocks_per_seq Maximum blocks per sequence
   * @param num_layers Total number of layers in model (for block format)
   * @param layer_idx Current layer index (for block format indexing)
   * @param use_block_format If true, cache is [pages, layers, ...]; if false,
   * [pages, ...]
   * @param position_offset Starting position for RoPE (default 0)
   */
  AttentionPrefillPrimitive(mlx::core::Stream stream, int num_heads,
                            int num_kv_heads, int head_dim, int hidden_size,
                            int block_size, int max_blocks_per_seq,
                            int num_layers = 0, int layer_idx = 0,
                            bool use_block_format = false,
                            int position_offset = 0);

  ~AttentionPrefillPrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Uses MLX operations for CPU execution
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [input, q, k, v, rope_cos, rope_sin, k_cache, v_cache, page_table]
   * Outputs: [context]
   *
   * Note: k_cache and v_cache are modified in-place
   */
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Vectorization support (vmap)
   * Maps over batch dimensions
   */
  std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
      const std::vector<mlx::core::array>& inputs,
      const std::vector<int>& axes) override;

  /**
   * Forward-mode autodiff (optional for Phase 1)
   */
  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& tangents,
      const std::vector<int>& argnums) override;

  /**
   * Reverse-mode autodiff (optional for Phase 1)
   */
  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

  /**
   * Compute output shapes from input shapes
   */
  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override {
    // Output shape: [batch, seq_len, num_heads, head_dim]
    return {inputs[1].shape()};  // Same as Q shape
  }

  /**
   * Primitive identification
   */
  const char* name() const override { return "attention_prefill_fused"; }

  /**
   * Check equivalence with another primitive
   */
  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /**
   * Get configuration for equivalence checking
   */
  int num_heads() const { return num_heads_; }
  int num_kv_heads() const { return num_kv_heads_; }
  int head_dim() const { return head_dim_; }
  int hidden_size() const { return hidden_size_; }
  int block_size() const { return block_size_; }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int num_layers() const { return num_layers_; }
  int layer_idx() const { return layer_idx_; }
  bool use_block_format() const { return use_block_format_; }
  int position_offset() const { return position_offset_; }

 private:
  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
  int hidden_size_;
  int block_size_;
  int max_blocks_per_seq_;
  int num_layers_;
  int layer_idx_;
  bool use_block_format_;
  int position_offset_;

  // Metal library (loaded lazily on first GPU eval)
  void* library_;  // Stores MTL::Library*

  /**
   * Load Metal library containing custom kernels
   * Returns MTL::Library* (cast from void*)
   */
  void* load_metal_library();
};

/**
 * Public API function for attention prefill
 *
 * @param input Input tokens [batch, seq_len, hidden_size]
 * @param q Query tensor [batch, seq_len, num_heads, head_dim]
 * @param k Key tensor [batch, seq_len, num_kv_heads, head_dim]
 * @param v Value tensor [batch, seq_len, num_kv_heads, head_dim]
 * @param rope_cos RoPE cosine table [max_seq_len, head_dim/2]
 * @param rope_sin RoPE sine table [max_seq_len, head_dim/2]
 * @param k_cache K cache pages [num_pages, block_size, num_kv_heads, head_dim]
 * or [num_pages, num_layers, block_size, num_kv_heads, head_dim]
 * @param v_cache V cache pages [num_pages, block_size, num_kv_heads, head_dim]
 * or [num_pages, num_layers, block_size, num_kv_heads, head_dim]
 * @param page_table Page table [batch, max_blocks_per_seq] (int32)
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param head_dim Head dimension
 * @param hidden_size Hidden size
 * @param block_size Tokens per block
 * @param max_blocks_per_seq Max blocks per sequence
 * @param num_layers Total number of layers (for block format)
 * @param layer_idx Current layer index (for block format)
 * @param use_block_format If true, cache has layer dimension
 * @param position_offset Starting position for RoPE
 * @param s Stream or device for execution
 * @return Context tensor [batch, seq_len, num_heads, head_dim]
 */
mlx::core::array attention_prefill_fused(
    const mlx::core::array& input, const mlx::core::array& q,
    const mlx::core::array& k, const mlx::core::array& v,
    const mlx::core::array& rope_cos, const mlx::core::array& rope_sin,
    mlx::core::array& k_cache,  // Modified in-place
    mlx::core::array& v_cache,  // Modified in-place
    const mlx::core::array& page_table, int num_heads, int num_kv_heads,
    int head_dim, int hidden_size, int block_size, int max_blocks_per_seq,
    int num_layers = 0, int layer_idx = 0, bool use_block_format = false,
    int position_offset = 0, mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
