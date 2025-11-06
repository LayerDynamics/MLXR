// Copyright © 2025 MLXR Development
// MLX Primitive-based custom attention decode kernel

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * AttentionDecode primitive using custom fused Metal kernel
 *
 * Implements attention decode path with paged KV cache:
 * 1. Load query Q for current token
 * 2. Walk paged KV cache to gather all past K, V
 * 3. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 * 4. Softmax with fp32 accumulation for numerical stability
 * 5. Compute context: context = softmax(scores) @ V
 *
 * Input shapes:
 *   q:           [batch, num_heads, head_dim]
 *   k_cache:     [num_pages, block_size, num_kv_heads, head_dim]
 *   v_cache:     [num_pages, block_size, num_kv_heads, head_dim]
 *   page_table:  [batch, max_blocks_per_seq] (int32)
 *   seq_lengths: [batch] (int32)
 *
 * Output shape:
 *   context:     [batch, num_heads, head_dim]
 *
 * Features:
 * - Paged KV cache with non-contiguous memory access
 * - Grouped Query Attention (GQA) support
 * - Numerically stable softmax (fp32 accumulation)
 * - Optional sliding window attention
 * - Configurable block sizes (16, 32 tokens per block)
 */
class AttentionDecodePrimitive : public mlx::core::Primitive {
 public:
  /**
   * Constructor
   *
   * @param stream MLX stream for execution
   * @param num_heads Number of query heads
   * @param num_kv_heads Number of key/value heads (for GQA)
   * @param head_dim Dimension per attention head
   * @param block_size Number of tokens per KV cache block
   * @param max_blocks_per_seq Maximum blocks per sequence
   * @param num_layers Total layers in model (for block format)
   * @param layer_idx Current layer index (for block format)
   * @param use_block_format Use native block format [pages, layers, ...]
   * (zero-copy)
   * @param use_sliding_window Enable sliding window attention
   * @param sliding_window_size Sliding window size (if enabled)
   */
  AttentionDecodePrimitive(mlx::core::Stream stream, int num_heads,
                           int num_kv_heads, int head_dim, int block_size,
                           int max_blocks_per_seq, int num_layers = 0,
                           int layer_idx = 0, bool use_block_format = false,
                           bool use_sliding_window = false,
                           int sliding_window_size = 0);

  ~AttentionDecodePrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Uses MLX operations for CPU execution
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [q, k_cache, v_cache, page_table, seq_lengths]
   * Outputs: [context]
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
    // Output shape: [batch, num_heads, head_dim]
    return {inputs[0].shape()};
  }

  /**
   * Primitive identification
   */
  const char* name() const override { return "attention_decode_fused"; }

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
  int block_size() const { return block_size_; }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int num_layers() const { return num_layers_; }
  int layer_idx() const { return layer_idx_; }
  bool use_block_format() const { return use_block_format_; }
  bool use_sliding_window() const { return use_sliding_window_; }
  int sliding_window_size() const { return sliding_window_size_; }

 private:
  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
  int block_size_;
  int max_blocks_per_seq_;
  int num_layers_;
  int layer_idx_;
  bool use_block_format_;
  bool use_sliding_window_;
  int sliding_window_size_;

  // Metal library (loaded lazily on first GPU eval)
  void* library_;  // Stores MTL::Library*

  /**
   * Load Metal library containing custom kernels
   * Returns MTL::Library* (cast from void*)
   */
  void* load_metal_library();
};

/**
 * Public API function for attention decode
 *
 * @param q Query tensor [batch, num_heads, head_dim]
 * @param k_cache K cache pages [num_pages, block_size, num_kv_heads, head_dim]
 * OR blocks [num_pages, num_layers, block_size, num_kv_heads, head_dim]
 * @param v_cache V cache pages [num_pages, block_size, num_kv_heads, head_dim]
 * OR blocks [num_pages, num_layers, block_size, num_kv_heads, head_dim]
 * @param page_table Page table [batch, max_blocks_per_seq] (int32)
 * @param seq_lengths Sequence lengths [batch] (int32)
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads (for GQA)
 * @param head_dim Head dimension
 * @param block_size Tokens per block
 * @param max_blocks_per_seq Max blocks per sequence
 * @param num_layers Total layers (for block format)
 * @param layer_idx Current layer index (for block format)
 * @param use_block_format Use native block format (zero-copy)
 * @param use_sliding_window Enable sliding window attention
 * @param sliding_window_size Sliding window size
 * @param s Stream or device for execution
 * @return Context tensor [batch, num_heads, head_dim]
 */
mlx::core::array attention_decode_fused(
    const mlx::core::array& q, const mlx::core::array& k_cache,
    const mlx::core::array& v_cache, const mlx::core::array& page_table,
    const mlx::core::array& seq_lengths, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq, int num_layers = 0,
    int layer_idx = 0, bool use_block_format = false,
    bool use_sliding_window = false, int sliding_window_size = 0,
    mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
