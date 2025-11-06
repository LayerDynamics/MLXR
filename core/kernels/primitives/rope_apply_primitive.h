// Copyright © 2025 MLXR Development
// MLX Primitive-based RoPE (Rotary Positional Embeddings) kernel

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * RoPE scaling modes
 */
enum class RoPEScalingMode {
  BASE = 0,    // Standard RoPE
  NTK = 1,     // NTK-aware interpolation
  YARN = 2,    // YaRN scaling
  LINEAR = 3,  // Linear interpolation
};

/**
 * RoPEApply primitive using custom Metal kernel
 *
 * Applies Rotary Positional Embeddings (RoPE) to query/key tensors.
 * RoPE encodes positional information by rotating pairs of dimensions:
 *
 * For each pair (even, odd):
 *   x_out[even] = x[even] * cos(θ) - x[odd] * sin(θ)
 *   x_out[odd]  = x[odd] * cos(θ) + x[even] * sin(θ)
 *
 * where θ = position * base^(-2i/d) for dimension pair i
 *
 * Input shapes:
 *   input:      [batch, seq_len, num_heads, head_dim] or [tokens, num_heads,
 * head_dim] cos_table:  [max_seq_len, head_dim/2] sin_table:  [max_seq_len,
 * head_dim/2] positions:  [batch, seq_len] or [tokens] (int32 position indices)
 *
 * Output shape:
 *   output:     Same shape as input
 *
 * Features:
 * - Multiple scaling modes: BASE, NTK, YaRN, LINEAR
 * - Precomputed cos/sin tables for efficiency
 * - Support for head_dim: 64, 80, 96, 112, 128, 160, 192, 256
 * - Handles both contiguous and strided tensors
 * - Optional in-place modification
 * - FP16/FP32 precision variants
 *
 * RoPE Variants:
 * 1. BASE:   Standard RoPE with θ = position * base^(-2i/d)
 * 2. NTK:    NTK-scaled for extended context windows
 * 3. YARN:   YaRN scaling with temperature and attention reweighting
 * 4. LINEAR: Simple linear interpolation scaling
 */
class RoPEApplyPrimitive : public mlx::core::Primitive {
 public:
  /**
   * Constructor
   *
   * @param stream MLX stream for execution
   * @param batch_size Number of sequences (or total tokens if flattened)
   * @param seq_len Sequence length (1 if flattened)
   * @param num_heads Number of attention heads
   * @param head_dim Dimension per head (must be even)
   * @param scaling_mode RoPE scaling mode
   * @param scale_factor Linear scaling factor (only for LINEAR mode)
   * @param position_offset Offset to add to positions (default 0)
   * @param inplace Whether to modify input in-place (default false)
   */
  RoPEApplyPrimitive(mlx::core::Stream stream, int batch_size, int seq_len,
                     int num_heads, int head_dim,
                     RoPEScalingMode scaling_mode = RoPEScalingMode::BASE,
                     float scale_factor = 1.0f, int position_offset = 0,
                     bool inplace = false);

  ~RoPEApplyPrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Uses MLX operations for CPU execution
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [input, cos_table, sin_table, positions]
   * Outputs: [output]
   *
   * Note: If inplace=true, output references input
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
    // Output shape: same as input
    return {inputs[0].shape()};
  }

  /**
   * Primitive identification
   */
  const char* name() const override {
    return inplace_ ? "rope_apply_inplace" : "rope_apply";
  }

  /**
   * Check equivalence with another primitive
   */
  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /**
   * Get configuration for equivalence checking
   */
  int batch_size() const { return batch_size_; }
  int seq_len() const { return seq_len_; }
  int num_heads() const { return num_heads_; }
  int head_dim() const { return head_dim_; }
  RoPEScalingMode scaling_mode() const { return scaling_mode_; }
  float scale_factor() const { return scale_factor_; }
  int position_offset() const { return position_offset_; }
  bool inplace() const { return inplace_; }

 private:
  int batch_size_;
  int seq_len_;
  int num_heads_;
  int head_dim_;
  RoPEScalingMode scaling_mode_;
  float scale_factor_;
  int position_offset_;
  bool inplace_;

  // Metal library (loaded lazily on first GPU eval)
  void* library_;  // Stores MTL::Library*

  /**
   * Load Metal library containing custom kernels
   * Returns MTL::Library* (cast from void*)
   */
  void* load_metal_library();
};

/**
 * Public API function for RoPE application
 *
 * @param input Input tensor [batch, seq_len, num_heads, head_dim] or [tokens,
 * num_heads, head_dim]
 * @param cos_table Precomputed cosine table [max_seq_len, head_dim/2]
 * @param sin_table Precomputed sine table [max_seq_len, head_dim/2]
 * @param positions Token position indices [batch, seq_len] or [tokens] (int32)
 * @param scaling_mode RoPE scaling mode
 * @param scale_factor Linear scaling factor (for LINEAR mode)
 * @param position_offset Offset to add to positions
 * @param inplace Whether to modify input in-place
 * @param s Stream or device for execution
 * @return Output tensor (same shape as input)
 */
mlx::core::array rope_apply(
    const mlx::core::array& input, const mlx::core::array& cos_table,
    const mlx::core::array& sin_table, const mlx::core::array& positions,
    RoPEScalingMode scaling_mode = RoPEScalingMode::BASE,
    float scale_factor = 1.0f, int position_offset = 0, bool inplace = false,
    mlx::core::StreamOrDevice s = {});

/**
 * Helper function to precompute RoPE cos/sin tables
 *
 * @param max_seq_len Maximum sequence length
 * @param head_dim Head dimension (must be even)
 * @param base RoPE base frequency (default 10000.0)
 * @param scaling_mode RoPE scaling mode
 * @param scale_factor Scaling factor (for LINEAR or NTK modes)
 * @param orig_context Original context length (for NTK scaling)
 * @return Pair of (cos_table, sin_table) arrays [max_seq_len, head_dim/2]
 */
std::pair<mlx::core::array, mlx::core::array> compute_rope_tables(
    int max_seq_len, int head_dim, float base = 10000.0f,
    RoPEScalingMode scaling_mode = RoPEScalingMode::BASE,
    float scale_factor = 1.0f, int orig_context = 2048);

/**
 * Helper function to compute scaled RoPE base (for NTK mode)
 *
 * @param base Original RoPE base
 * @param scale Context length scaling factor
 * @param head_dim Head dimension
 * @param orig_context Original context length
 * @return Scaled base frequency
 */
float compute_ntk_scaled_base(float base, float scale, int head_dim,
                              int orig_context);

/**
 * Get scaling mode name for debugging
 */
const char* rope_scaling_mode_name(RoPEScalingMode mode);

}  // namespace kernels
}  // namespace mlxr
