/**
 * @file rmsnorm_kernel.h
 * @brief RMSNorm Metal kernel wrapper
 */

#pragma once

#include "graph/tensor.h"
#include "metal_kernel_base.h"

namespace mlxr {
namespace kernels {

/**
 * @brief Fused RMSNorm Metal kernel wrapper
 *
 * Provides optimized RMSNorm computation using custom Metal kernels.
 * Fuses multiple operations (x², mean, rsqrt, normalize, weight scale)
 * into a single kernel dispatch.
 */
class RMSNormKernel : public MetalKernelBase {
 public:
  /**
   * @brief Construct RMSNorm kernel wrapper
   * @param library_path Path to Metal library containing RMSNorm kernels
   */
  explicit RMSNormKernel(const std::string& library_path);

  /**
   * @brief Destructor
   */
  ~RMSNormKernel();

  // Disable copy and move
  RMSNormKernel(const RMSNormKernel&) = delete;
  RMSNormKernel& operator=(const RMSNormKernel&) = delete;
  RMSNormKernel(RMSNormKernel&&) = delete;
  RMSNormKernel& operator=(RMSNormKernel&&) = delete;

  /**
   * @brief Apply RMSNorm
   * @param input Input tensor [..., hidden_size]
   * @param weight Weight tensor [hidden_size]
   * @param eps Epsilon for numerical stability
   * @return Normalized tensor with same shape as input
   */
  graph::Tensor forward(const graph::Tensor& input, const graph::Tensor& weight,
                        float eps = 1e-6f);

  /**
   * @brief Apply RMSNorm with fused residual connection
   * @param input Input tensor [..., hidden_size]
   * @param weight Weight tensor [hidden_size]
   * @param residual Residual tensor (same shape as input)
   * @param eps Epsilon for numerical stability
   * @return Normalized tensor with same shape as input
   */
  graph::Tensor forward_with_residual(const graph::Tensor& input,
                                      const graph::Tensor& weight,
                                      const graph::Tensor& residual,
                                      float eps = 1e-6f);

 private:
  // Opaque pointers to Metal objects (actual types hidden in .mm file)
  void* library_;
  void* pipeline_fp32_;
  void* pipeline_fp16_;
  void* pipeline_residual_;
};

/**
 * @brief Convenience function for fused RMSNorm
 * @param input Input tensor
 * @param weight Weight tensor
 * @param eps Epsilon value
 * @return Normalized tensor
 *
 * Uses a singleton RMSNormKernel instance initialized on first use.
 */
graph::Tensor fused_rmsnorm(const graph::Tensor& input,
                            const graph::Tensor& weight, float eps = 1e-6f);

}  // namespace kernels
}  // namespace mlxr
