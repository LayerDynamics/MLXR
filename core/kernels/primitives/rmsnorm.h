/**
 * @file rmsnorm.h
 * @brief RMSNorm primitive for MLX integration
 */

#pragma once

#include <mlx/mlx.h>

#include <vector>

namespace mlxr {
namespace kernels {

/**
 * @brief RMSNorm primitive implementation
 *
 * Implements RMSNorm as an MLX custom primitive for proper Metal integration.
 * This ensures correct buffer handling and stream synchronization.
 */
class RMSNormPrimitive : public mlx::core::Primitive {
 public:
  explicit RMSNormPrimitive(mlx::core::Stream stream, float eps)
      : Primitive(stream), eps_(eps) {}

  /**
   * @brief GPU evaluation (Metal backend)
   */
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                mlx::core::array& out) override;

  /**
   * @brief CPU evaluation (fallback)
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                mlx::core::array& out) override;

  /**
   * @brief Print primitive info
   */
  void print(std::ostream& os) override { os << "RMSNorm(eps=" << eps_ << ")"; }

 private:
  float eps_;

  // Metal kernel dispatch helper
  void eval_metal(const std::vector<mlx::core::array>& inputs,
                  mlx::core::array& out);
};

/**
 * @brief Apply RMSNorm operation
 * @param input Input tensor [..., hidden_size]
 * @param weight Weight tensor [hidden_size]
 * @param eps Epsilon for numerical stability
 * @param stream MLX stream to execute on
 * @return Normalized tensor
 */
mlx::core::array rmsnorm(const mlx::core::array& input,
                         const mlx::core::array& weight, float eps = 1e-6f,
                         mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
