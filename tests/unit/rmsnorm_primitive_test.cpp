/**
 * @file rmsnorm_primitive_test.cpp
 * @brief Unit tests for RMSNorm Primitive with proper Metal buffer access
 */

#include <gtest/gtest.h>

#ifdef USE_CUSTOM_KERNELS

#include <cmath>
#include <vector>

#include "graph/tensor.h"
#include "mlx/mlx.h"
#include "primitives/rmsnorm_primitive.h"

namespace mlxr {
namespace kernels {
namespace test {

// Helper function to compute reference RMSNorm on CPU
mlx::core::array reference_rmsnorm(const mlx::core::array& input,
                                   const mlx::core::array& weight, float eps) {
  auto x_sq = mlx::core::multiply(input, input);
  std::vector<int> axes = {-1};
  auto mean_sq = mlx::core::mean(x_sq, axes, /*keepdims=*/true);
  auto rms_inv =
      mlx::core::rsqrt(mlx::core::add(mean_sq, mlx::core::array(eps)));
  auto normalized = mlx::core::multiply(input, rms_inv);
  auto result = mlx::core::multiply(normalized, weight);
  mlx::core::eval(result);
  return result;
}

// Helper to check array equality with tolerance
bool arrays_close(const mlx::core::array& a, const mlx::core::array& b,
                  float rtol = 1e-5f, float atol = 1e-6f) {
  if (a.shape() != b.shape()) return false;

  mlx::core::eval(a);
  mlx::core::eval(b);

  auto diff = mlx::core::abs(mlx::core::subtract(a, b));
  auto threshold = mlx::core::add(
      mlx::core::array(atol),
      mlx::core::multiply(mlx::core::array(rtol), mlx::core::abs(b)));

  auto close = mlx::core::less_equal(diff, threshold);
  mlx::core::eval(close);

  return mlx::core::all(close).item<bool>();
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, BasicForward) {
  // Test basic forward pass with simple input
  int hidden_size = 128;
  auto input = mlx::core::random::normal({1, 4, hidden_size});
  auto weight = mlx::core::ones({hidden_size});
  float eps = 1e-6f;

  // Compute with custom primitive
  auto output = rmsnorm_fused(input, weight, eps);
  mlx::core::eval(output);

  // Compute reference
  auto expected = reference_rmsnorm(input, weight, eps);

  // Check shapes match
  EXPECT_EQ(output.shape(), expected.shape());

  // Check values are close
  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

TEST(RMSNormPrimitiveTest, SingleSequence) {
  // Test with single sequence (2D input)
  int seq_len = 16;
  int hidden_size = 256;

  auto input = mlx::core::random::normal({seq_len, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  auto expected = reference_rmsnorm(input, weight, 1e-6f);

  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

TEST(RMSNormPrimitiveTest, BatchedSequences) {
  // Test with batched sequences (3D input)
  int batch_size = 4;
  int seq_len = 8;
  int hidden_size = 512;

  auto input = mlx::core::random::normal({batch_size, seq_len, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  auto expected = reference_rmsnorm(input, weight, 1e-6f);

  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

// ============================================================================
// Different Dtypes
// ============================================================================

TEST(RMSNormPrimitiveTest, Float32Dtype) {
  // Explicitly test float32
  auto input = mlx::core::random::normal({2, 64}, mlx::core::float32);
  auto weight = mlx::core::ones({64}, mlx::core::float32);

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  EXPECT_EQ(output.dtype(), mlx::core::float32);
  EXPECT_EQ(output.shape(), input.shape());
}

TEST(RMSNormPrimitiveTest, Float16Dtype) {
  // Test with float16 (uses different kernel)
  auto input = mlx::core::random::normal({2, 64}, mlx::core::float16);
  auto weight = mlx::core::ones({64}, mlx::core::float16);

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  EXPECT_EQ(output.dtype(), mlx::core::float16);
  EXPECT_EQ(output.shape(), input.shape());

  // Check numerical correctness (with relaxed tolerance for fp16)
  auto expected = reference_rmsnorm(input, weight, 1e-6f);
  EXPECT_TRUE(arrays_close(output, expected, 1e-2f, 1e-3f));
}

// ============================================================================
// Weight Scaling
// ============================================================================

TEST(RMSNormPrimitiveTest, NonUniformWeights) {
  // Test with non-uniform weights
  int hidden_size = 128;
  auto input = mlx::core::random::normal({2, 4, hidden_size});
  auto weight = mlx::core::random::uniform(
      mlx::core::array(0.5f), mlx::core::array(1.5f), {hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  auto expected = reference_rmsnorm(input, weight, 1e-6f);

  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

TEST(RMSNormPrimitiveTest, ZeroWeights) {
  // Test with zero weights (should produce zero output)
  int hidden_size = 64;
  auto input = mlx::core::random::normal({1, hidden_size});
  auto weight = mlx::core::zeros({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  // Output should be all zeros
  auto zero = mlx::core::zeros_like(output);
  EXPECT_TRUE(arrays_close(output, zero, 1e-6f, 1e-7f));
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, LargeValues) {
  // Test with large input values
  int hidden_size = 128;
  auto input = mlx::core::multiply(mlx::core::random::normal({2, hidden_size}),
                                   mlx::core::array(1000.0f));
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  // Should not produce NaN or Inf
  auto is_finite = mlx::core::isfinite(output);
  mlx::core::eval(is_finite);
  EXPECT_TRUE(mlx::core::all(is_finite).item<bool>());

  // Check against reference
  auto expected = reference_rmsnorm(input, weight, 1e-6f);
  EXPECT_TRUE(arrays_close(output, expected, 1e-3f, 1e-4f));
}

TEST(RMSNormPrimitiveTest, SmallValues) {
  // Test with small input values
  int hidden_size = 128;
  auto input = mlx::core::multiply(mlx::core::random::normal({2, hidden_size}),
                                   mlx::core::array(0.001f));
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  auto expected = reference_rmsnorm(input, weight, 1e-6f);
  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

TEST(RMSNormPrimitiveTest, DifferentEpsilonValues) {
  // Test with different epsilon values
  int hidden_size = 128;
  auto input = mlx::core::random::normal({2, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  std::vector<float> epsilons = {1e-8f, 1e-6f, 1e-4f, 1e-2f};

  for (float eps : epsilons) {
    auto output = rmsnorm_fused(input, weight, eps);
    mlx::core::eval(output);

    auto expected = reference_rmsnorm(input, weight, eps);
    EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f))
        << "Failed for epsilon = " << eps;
  }
}

// ============================================================================
// Shape and Dimension Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, DifferentHiddenSizes) {
  // Test with various hidden sizes
  std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

  for (int hidden_size : sizes) {
    auto input = mlx::core::random::normal({1, 4, hidden_size});
    auto weight = mlx::core::ones({hidden_size});

    auto output = rmsnorm_fused(input, weight, 1e-6f);
    mlx::core::eval(output);

    EXPECT_EQ(output.shape(), input.shape())
        << "Failed for hidden_size = " << hidden_size;

    auto expected = reference_rmsnorm(input, weight, 1e-6f);
    EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f))
        << "Failed for hidden_size = " << hidden_size;
  }
}

TEST(RMSNormPrimitiveTest, DifferentBatchSizes) {
  // Test with various batch sizes
  int hidden_size = 128;
  std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};

  for (int batch : batch_sizes) {
    auto input = mlx::core::random::normal({batch, 8, hidden_size});
    auto weight = mlx::core::ones({hidden_size});

    auto output = rmsnorm_fused(input, weight, 1e-6f);
    mlx::core::eval(output);

    EXPECT_EQ(output.shape(), input.shape())
        << "Failed for batch_size = " << batch;
  }
}

TEST(RMSNormPrimitiveTest, SingleToken) {
  // Test with single token (minimal case)
  int hidden_size = 128;
  auto input = mlx::core::random::normal({1, 1, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  auto expected = reference_rmsnorm(input, weight, 1e-6f);
  EXPECT_TRUE(arrays_close(output, expected, 1e-4f, 1e-5f));
}

// ============================================================================
// Memory and Buffer Access Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, NonContiguousInput) {
  // Test with non-contiguous input (should trigger copy)
  int hidden_size = 128;
  auto input = mlx::core::random::normal({4, 8, hidden_size});

  // Transpose to make non-contiguous
  auto input_transposed = mlx::core::transpose(input, {1, 0, 2});
  auto weight = mlx::core::ones({hidden_size});

  // Should handle non-contiguous arrays correctly
  auto output = rmsnorm_fused(input_transposed, weight, 1e-6f);
  mlx::core::eval(output);

  EXPECT_EQ(output.shape(), input_transposed.shape());
}

TEST(RMSNormPrimitiveTest, MultipleEvaluations) {
  // Test multiple evaluations (checks Metal resource management)
  int hidden_size = 128;
  auto input = mlx::core::random::normal({2, 4, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  for (int i = 0; i < 10; i++) {
    auto output = rmsnorm_fused(input, weight, 1e-6f);
    mlx::core::eval(output);

    EXPECT_EQ(output.shape(), input.shape()) << "Failed on iteration " << i;
  }
}

TEST(RMSNormPrimitiveTest, ConcurrentEvaluations) {
  // Test that multiple operations can be scheduled
  int hidden_size = 128;
  std::vector<mlx::core::array> inputs;
  std::vector<mlx::core::array> outputs;

  auto weight = mlx::core::ones({hidden_size});

  // Create multiple operations
  for (int i = 0; i < 5; i++) {
    auto input = mlx::core::random::normal({2, 4, hidden_size});
    inputs.push_back(input);

    auto output = rmsnorm_fused(input, weight, 1e-6f);
    outputs.push_back(output);
  }

  // Evaluate all at once
  mlx::core::eval(outputs);

  // Check all completed
  for (size_t i = 0; i < outputs.size(); i++) {
    EXPECT_EQ(outputs[i].shape(), inputs[i].shape());
  }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, InvalidInputDimensions) {
  // Test with 0D input (should throw)
  auto input = mlx::core::array(1.0f);
  auto weight = mlx::core::ones({1});

  EXPECT_THROW({ rmsnorm_fused(input, weight, 1e-6f); }, std::invalid_argument);
}

TEST(RMSNormPrimitiveTest, WeightSizeMismatch) {
  // Test with mismatched weight size (should throw)
  auto input = mlx::core::random::normal({2, 128});
  auto weight = mlx::core::ones({64});  // Wrong size!

  EXPECT_THROW({ rmsnorm_fused(input, weight, 1e-6f); }, std::invalid_argument);
}

TEST(RMSNormPrimitiveTest, NonVectorWeight) {
  // Test with 2D weight (should throw)
  auto input = mlx::core::random::normal({2, 128});
  auto weight = mlx::core::ones({64, 2});  // 2D!

  EXPECT_THROW({ rmsnorm_fused(input, weight, 1e-6f); }, std::invalid_argument);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, IntegrationWithTensor) {
  // Test integration with graph::Tensor wrapper
  using namespace mlxr::graph;

  int hidden_size = 128;
  auto input_arr = mlx::core::random::normal({2, 4, hidden_size});
  auto weight_arr = mlx::core::ones({hidden_size});

  // Wrap in Tensor
  Tensor input(input_arr);
  Tensor weight(weight_arr);

  // Call through Tensor API (simulates RMSNorm::forward)
  auto output_arr = rmsnorm_fused(input.array(), weight.array(), 1e-6f);
  Tensor output(output_arr);

  mlx::core::eval(output.array());

  EXPECT_EQ(output.shape(), input.shape());
  EXPECT_EQ(output.dtype(), input.dtype());
}

TEST(RMSNormPrimitiveTest, ChainedOperations) {
  // Test that RMSNorm can be part of a computation graph
  int hidden_size = 128;
  auto input = mlx::core::random::normal({2, 4, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  // Chain: input -> rmsnorm -> add -> rmsnorm
  auto norm1 = rmsnorm_fused(input, weight, 1e-6f);
  auto added = mlx::core::add(norm1, mlx::core::array(1.0f));
  auto norm2 = rmsnorm_fused(added, weight, 1e-6f);

  mlx::core::eval(norm2);

  EXPECT_EQ(norm2.shape(), input.shape());
}

// ============================================================================
// Performance Sanity Tests
// ============================================================================

TEST(RMSNormPrimitiveTest, LargeInput) {
  // Test with large input to ensure Metal dispatch works
  int batch_size = 16;
  int seq_len = 512;
  int hidden_size = 4096;

  auto input = mlx::core::random::normal({batch_size, seq_len, hidden_size});
  auto weight = mlx::core::ones({hidden_size});

  auto output = rmsnorm_fused(input, weight, 1e-6f);
  mlx::core::eval(output);

  EXPECT_EQ(output.shape(), input.shape());

  // Verify a subset matches reference (full comparison would be slow)
  auto input_subset = mlx::core::reshape(
      mlx::core::slice(input, {0, 0, 0}, {1, 4, hidden_size}),
      {1, 4, hidden_size});
  auto output_subset = mlx::core::reshape(
      mlx::core::slice(output, {0, 0, 0}, {1, 4, hidden_size}),
      {1, 4, hidden_size});

  auto expected_subset = reference_rmsnorm(input_subset, weight, 1e-6f);
  EXPECT_TRUE(arrays_close(output_subset, expected_subset, 1e-4f, 1e-5f));
}

}  // namespace test
}  // namespace kernels
}  // namespace mlxr

#endif  // USE_CUSTOM_KERNELS
