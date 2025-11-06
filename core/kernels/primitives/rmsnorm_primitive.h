// Copyright © 2025 MLXR Development
// MLX Primitive-based custom RMSNorm kernel

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * RMSNorm primitive using custom fused Metal kernel
 *
 * Implements Root Mean Square Layer Normalization with a single
 * kernel dispatch, fusing:
 * 1. Square computation
 * 2. Mean reduction
 * 3. RMS calculation
 * 4. Normalization
 * 5. Weight scaling
 *
 * Input shapes:
 *   input:  [batch, seq_len, hidden_size] or [seq_len, hidden_size]
 *   weight: [hidden_size]
 * Output shape: same as input
 */
class RMSNormPrimitive : public mlx::core::Primitive {
 public:
  explicit RMSNormPrimitive(mlx::core::Stream stream, float eps);

  ~RMSNormPrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Uses MLX operations for CPU execution
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [input, weight]
   * Outputs: [normalized_output]
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
    // Output has same shape as input
    return {inputs[0].shape()};
  }

  /**
   * Primitive identification
   */
  const char* name() const override { return "rmsnorm_fused"; }

  /**
   * Check equivalence with another primitive
   */
  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /**
   * Get state for equivalence checking
   */
  float eps() const { return eps_; }

 private:
  float eps_;  // Epsilon for numerical stability

  // Metal library (loaded lazily on first GPU eval)
  void* library_;  // Stores MTL::Library*

  /**
   * Load Metal library containing custom kernels
   * Returns MTL::Library* (cast from void*)
   */
  void* load_metal_library();  // Returns MTL::Library* as void*
};

/**
 * Public API function for RMSNorm
 *
 * @param input Input tensor [batch, seq_len, hidden_size]
 * @param weight Scale weights [hidden_size]
 * @param eps Epsilon for numerical stability
 * @param s Stream or device for execution
 * @return Normalized output tensor with same shape as input
 */
mlx::core::array rmsnorm_fused(const mlx::core::array& input,
                               const mlx::core::array& weight,
                               float eps = 1e-6f,
                               mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
