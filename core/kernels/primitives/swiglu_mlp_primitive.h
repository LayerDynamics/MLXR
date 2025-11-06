// Copyright © 2025 MLXR Development
// MLX Primitive-based SwiGLU MLP kernel

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * SwiGLUMLP primitive using custom fused Metal kernel
 *
 * Implements the SwiGLU MLP layer used in Llama and other modern LLMs:
 * MLP(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down
 *
 * where:
 * - Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * - ⊙ denotes element-wise multiplication (gating)
 * - W_gate, W_up, W_down are learned weight matrices
 *
 * Input shapes:
 *   input:    [M, hidden_size] where M = batch * seq_len
 *   w_gate:   [intermediate_size, hidden_size]
 *   w_up:     [intermediate_size, hidden_size]
 *   w_down:   [hidden_size, intermediate_size]
 *   bias_gate: [intermediate_size] (optional)
 *   bias_up:   [intermediate_size] (optional)
 *   bias_down: [hidden_size] (optional)
 *
 * Output shape:
 *   output:   [M, hidden_size]
 *
 * For Llama models:
 * - Llama 7B:  hidden_size=4096,  intermediate_size=11008  (2.69x expansion)
 * - Llama 13B: hidden_size=5120,  intermediate_size=13824  (2.70x expansion)
 * - Llama 70B: hidden_size=8192,  intermediate_size=28672  (3.50x expansion)
 *
 * Features:
 * - Fused gate and up projections (computed in parallel)
 * - In-kernel SwiGLU activation (avoids memory traffic)
 * - Optional bias addition for all three projections
 * - FP32 accumulation for numerical stability
 * - Optimized tiled matrix multiplication
 * - Support for quantized weights (Phase 2)
 *
 * Performance:
 * - Single kernel dispatch for entire MLP layer
 * - Minimizes memory bandwidth by fusing operations
 * - Cooperative threadgroup execution
 * - Vectorized memory access patterns
 */
class SwiGLUMLPPrimitive : public mlx::core::Primitive {
 public:
  /**
   * Constructor
   *
   * @param stream MLX stream for execution
   * @param M Number of tokens (batch_size * seq_len)
   * @param hidden_size Hidden dimension
   * @param intermediate_size Intermediate dimension (FFN hidden)
   * @param has_bias Whether bias terms are present
   */
  SwiGLUMLPPrimitive(mlx::core::Stream stream, int M, int hidden_size,
                     int intermediate_size, bool has_bias = false);

  ~SwiGLUMLPPrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Uses MLX operations for CPU execution
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [input, w_gate, w_up, w_down, bias_gate (opt), bias_up (opt),
   * bias_down (opt)] Outputs: [output]
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
    // Output shape: [M, hidden_size] (same as input)
    return {inputs[0].shape()};
  }

  /**
   * Primitive identification
   */
  const char* name() const override { return "swiglu_mlp_fused"; }

  /**
   * Check equivalence with another primitive
   */
  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /**
   * Get configuration for equivalence checking
   */
  int M() const { return M_; }
  int hidden_size() const { return hidden_size_; }
  int intermediate_size() const { return intermediate_size_; }
  bool has_bias() const { return has_bias_; }

 private:
  int M_;
  int hidden_size_;
  int intermediate_size_;
  bool has_bias_;

  // Metal library (loaded lazily on first GPU eval)
  void* library_;  // Stores MTL::Library*

  /**
   * Load Metal library containing custom kernels
   * Returns MTL::Library* (cast from void*)
   */
  void* load_metal_library();
};

/**
 * Public API function for SwiGLU MLP
 *
 * @param input Input tensor [M, hidden_size]
 * @param w_gate Gate weight matrix [intermediate_size, hidden_size]
 * @param w_up Up weight matrix [intermediate_size, hidden_size]
 * @param w_down Down weight matrix [hidden_size, intermediate_size]
 * @param bias_gate Optional gate bias [intermediate_size]
 * @param bias_up Optional up bias [intermediate_size]
 * @param bias_down Optional down bias [hidden_size]
 * @param s Stream or device for execution
 * @return Output tensor [M, hidden_size]
 */
mlx::core::array swiglu_mlp(const mlx::core::array& input,
                            const mlx::core::array& w_gate,
                            const mlx::core::array& w_up,
                            const mlx::core::array& w_down,
                            const mlx::core::array* bias_gate = nullptr,
                            const mlx::core::array* bias_up = nullptr,
                            const mlx::core::array* bias_down = nullptr,
                            mlx::core::StreamOrDevice s = {});

/**
 * Helper function: Swish activation
 * swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
mlx::core::array swish(const mlx::core::array& x,
                       mlx::core::StreamOrDevice s = {});

/**
 * Helper function: SwiGLU activation
 * swiglu(gate, up) = swish(gate) * up
 */
mlx::core::array swiglu(const mlx::core::array& gate,
                        const mlx::core::array& up,
                        mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
