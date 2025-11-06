// Copyright © 2025 MLXR Development
// MLX Primitive-based quantized GEMM with dequantization

#pragma once

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlxr {
namespace kernels {

/**
 * Quantization format types
 */
enum class QuantType {
  Q4_0 = 0,      // 4-bit weights, shared scale per 32 elements
  Q4_1 = 1,      // 4-bit weights, scale + min per 32 elements
  Q8_0 = 2,      // 8-bit weights, shared scale per 32 elements
  Q4_K = 3,      // 4-bit K-quant with super-block structure (256 elements)
  Q6_K = 4,      // 6-bit K-quant with super-block structure
  Q5_0 = 5,      // 5-bit weights, shared scale per 32 elements
  Q5_1 = 6,      // 5-bit weights, scale + min per 32 elements
  Q8_1 = 7,      // 8-bit weights, scale + min per 32 elements
  Q2_K = 8,      // 2-bit K-quant
  Q3_K = 9,      // 3-bit K-quant
  Q5_K = 10,     // 5-bit K-quant
  Q8_K = 11,     // 8-bit K-quant
  IQ2_XXS = 12,  // IQ2_XXS (2.06 bpw)
  IQ2_XS = 13,   // IQ2_XS (2.31 bpw)
  IQ3_XXS = 14,  // IQ3_XXS (3.06 bpw)
  IQ3_S = 15,    // IQ3_S (3.44 bpw)
};

/**
 * QGemmDequant primitive using custom fused Metal kernel
 *
 * Implements quantized matrix multiplication with on-the-fly dequantization:
 * Y = X @ W^T + bias (optional)
 *
 * Input shapes:
 *   input:   [M, K]  (fp16) - Input activations
 *   weights: Variable size depending on quantization format - Quantized weight
 * matrix bias:    [N] (fp16, optional) - Bias vector
 *
 * Output shape:
 *   output:  [M, N]  (fp16)
 *
 * Where:
 *   M = batch_size * seq_len (number of tokens)
 *   K = input features (hidden_size)
 *   N = output features (hidden_size or vocab_size)
 *
 * Features:
 * - On-the-fly weight dequantization (no staging buffer needed)
 * - Vectorized loads for quantized data
 * - FP32 accumulation for numerical stability
 * - Optional fused bias addition
 * - Supports multiple quantization formats (GGUF K-quants, standard quants)
 * - Optimized for grouped quantization (32-256 elements per group)
 * - Cooperative threadgroup execution
 *
 * Quantization formats:
 * - Q4_0:  4-bit, 32-elem groups, scale only (4.5 bpw)
 * - Q4_1:  4-bit, 32-elem groups, scale + min (5.0 bpw)
 * - Q8_0:  8-bit, 32-elem groups, scale only (8.5 bpw)
 * - Q4_K:  4-bit K-quant, 256-elem super-blocks, 8 sub-blocks (4.5 bpw)
 * - Q6_K:  6-bit K-quant, 256-elem super-blocks (6.5 bpw)
 *
 * Performance:
 * - Tiled execution: 32x64 output tile per threadgroup
 * - Threadgroup size: 512 threads
 * - Each thread computes 4 output elements
 * - K-dimension reduction over quantization groups
 */
class QGemmDequantPrimitive : public mlx::core::Primitive {
 public:
  /**
   * Constructor
   *
   * @param stream MLX stream for execution
   * @param M Number of input rows (batch_size * seq_len)
   * @param N Number of output columns (output features)
   * @param K Number of input columns (input features)
   * @param quant_type Quantization format
   * @param group_size Elements per quantization group
   * @param has_bias Whether bias is present
   */
  QGemmDequantPrimitive(mlx::core::Stream stream, int M, int N, int K,
                        QuantType quant_type, int group_size,
                        bool has_bias = false);

  ~QGemmDequantPrimitive() override;

  /**
   * Evaluate on CPU (fallback)
   * Dequantizes weights and performs standard matrix multiplication
   */
  void eval_cpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;

  /**
   * Evaluate on GPU using custom Metal kernel
   * Inputs: [input, weights, bias (optional)]
   * Outputs: [output]
   *
   * Note: weights are in quantized format
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
    // Output shape: [M, N]
    return {{M_, N_}};
  }

  /**
   * Primitive identification
   */
  const char* name() const override { return "q_gemm_dequant"; }

  /**
   * Check equivalence with another primitive
   */
  bool is_equivalent(const mlx::core::Primitive& other) const override;

  /**
   * Get configuration for equivalence checking
   */
  int M() const { return M_; }
  int N() const { return N_; }
  int K() const { return K_; }
  QuantType quant_type() const { return quant_type_; }
  int group_size() const { return group_size_; }
  bool has_bias() const { return has_bias_; }

 public:
  /**
   * Get bytes per weight for quantization format
   */
  static float get_bytes_per_weight(QuantType type);

  /**
   * Get default group size for quantization format
   */
  static int get_default_group_size(QuantType type);

 private:
  int M_;
  int N_;
  int K_;
  QuantType quant_type_;
  int group_size_;
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
 * Public API function for quantized GEMM with dequantization
 *
 * @param input Input tensor [M, K] (fp16)
 * @param weights Quantized weight tensor (format-specific size)
 * @param M Number of input rows
 * @param N Number of output columns
 * @param K Number of input columns
 * @param quant_type Quantization format
 * @param group_size Elements per quantization group
 * @param bias Optional bias tensor [N] (fp16)
 * @param s Stream or device for execution
 * @return Output tensor [M, N] (fp16)
 */
mlx::core::array q_gemm_dequant(const mlx::core::array& input,
                                const mlx::core::array& weights, int M, int N,
                                int K, QuantType quant_type, int group_size,
                                const mlx::core::array* bias = nullptr,
                                mlx::core::StreamOrDevice s = {});

/**
 * Helper function to compute quantized weight buffer size
 *
 * @param N Number of output features
 * @param K Number of input features
 * @param quant_type Quantization format
 * @return Size in bytes for quantized weight buffer
 */
size_t compute_quantized_weight_size(int N, int K, QuantType quant_type);

/**
 * Helper function to get quantization format name
 */
const char* quant_type_name(QuantType type);

}  // namespace kernels
}  // namespace mlxr
