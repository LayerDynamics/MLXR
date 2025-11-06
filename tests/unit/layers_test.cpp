/**
 * @file layers_test.cpp
 * @brief Unit tests for neural network layers
 */

#include "graph/layers.h"

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <cmath>

#include "graph/model.h"  // For KVCache definition
#include "graph/tensor.h"

using namespace mlxr::graph;

// Test fixture for layer tests
class LayersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common setup
  }

  void TearDown() override {
    // Common teardown
  }

  // Helper to check if shapes are equal
  bool shapes_equal(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  // Helper to check if tensor values are close
  bool values_close(const Tensor& t, float expected, float atol = 1e-4f) {
    auto arr = t.array();
    mlx::core::eval(arr);

    // Compute mean absolute difference from expected
    auto diff =
        mlx::core::abs(mlx::core::subtract(arr, mlx::core::array(expected)));
    auto mean_diff = mlx::core::mean(diff);
    mlx::core::eval(mean_diff);

    return mean_diff.item<float>() < atol;
  }
};

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST_F(LayersTest, RMSNormConstruction) {
  RMSNorm norm(128, 1e-6f);

  // Check weight is initialized
  EXPECT_FALSE(norm.weight().empty());
  EXPECT_TRUE(shapes_equal(norm.weight().shape(), {128}));
}

TEST_F(LayersTest, RMSNormForward) {
  RMSNorm norm(4, 1e-6f);

  // Create input tensor [2, 4] with known values
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  Tensor input =
      mlxr::graph::from_data(data.data(), {2, 4}, mlx::core::float32);

  // Apply normalization
  Tensor output = norm.forward(input);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {2, 4}));

  // Verify output is not all zeros
  auto arr = output.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(mlx::core::abs(arr));
  mlx::core::eval(sum);
  EXPECT_GT(sum.item<float>(), 0.0f);
}

TEST_F(LayersTest, RMSNormNumericalStability) {
  RMSNorm norm(4, 1e-6f);

  // Test with very small values
  std::vector<float> data = {1e-8f, 2e-8f, 3e-8f, 4e-8f};
  Tensor input =
      mlxr::graph::from_data(data.data(), {1, 4}, mlx::core::float32);

  // Should not crash or produce NaN
  Tensor output = norm.forward(input);

  auto arr = output.array();
  mlx::core::eval(arr);

  // Check for NaN
  bool has_nan = false;
  for (int i = 0; i < 4; ++i) {
    if (std::isnan(arr.data<float>()[i])) {
      has_nan = true;
      break;
    }
  }
  EXPECT_FALSE(has_nan);
}

TEST_F(LayersTest, RMSNormWeightAccess) {
  RMSNorm norm(8);

  // Test mutable access
  Tensor& weight = norm.weight();
  EXPECT_FALSE(weight.empty());

  // Test const access
  const RMSNorm& norm_const = norm;
  const Tensor& weight_const = norm_const.weight();
  EXPECT_FALSE(weight_const.empty());
}

// ============================================================================
// Linear Layer Tests
// ============================================================================

TEST_F(LayersTest, LinearConstruction) {
  Linear layer(128, 256, false);

  // Check weight is initialized
  EXPECT_FALSE(layer.weight().empty());
  EXPECT_TRUE(shapes_equal(layer.weight().shape(), {256, 128}));

  // Check no bias
  EXPECT_EQ(layer.bias(), nullptr);
}

TEST_F(LayersTest, LinearConstructionWithBias) {
  Linear layer(128, 256, true);

  // Check weight and bias are initialized
  EXPECT_FALSE(layer.weight().empty());
  EXPECT_TRUE(shapes_equal(layer.weight().shape(), {256, 128}));

  EXPECT_NE(layer.bias(), nullptr);
  EXPECT_TRUE(shapes_equal(layer.bias()->shape(), {256}));
}

TEST_F(LayersTest, LinearXavierInitialization) {
  int in_features = 100;
  int out_features = 200;
  Linear layer(in_features, out_features, false);

  // Xavier limit: sqrt(6 / (in + out)) = sqrt(6 / 300) ≈ 0.1414
  float expected_limit = std::sqrt(6.0f / (in_features + out_features));

  // Get weight values
  auto weight_arr = layer.weight().array();
  mlx::core::eval(weight_arr);

  // Check max and min are within expected range
  auto max_val = mlx::core::max(weight_arr);
  auto min_val = mlx::core::min(weight_arr);
  mlx::core::eval(max_val);
  mlx::core::eval(min_val);

  EXPECT_LE(max_val.item<float>(), expected_limit);
  EXPECT_GE(min_val.item<float>(), -expected_limit);
}

TEST_F(LayersTest, LinearForward) {
  Linear layer(4, 3, false);

  // Create input [2, 4]
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Tensor input =
      mlxr::graph::from_data(data.data(), {2, 4}, mlx::core::float32);

  // Apply linear transformation
  Tensor output = layer.forward(input);

  // Check output shape [2, 3]
  EXPECT_TRUE(shapes_equal(output.shape(), {2, 3}));
}

TEST_F(LayersTest, LinearForwardWithBias) {
  Linear layer(4, 3, true);

  // Set bias to ones for testing
  *layer.bias() = mlxr::graph::ones({3}, mlx::core::float32);

  // Create input
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor input =
      mlxr::graph::from_data(data.data(), {1, 4}, mlx::core::float32);

  // Apply linear transformation
  Tensor output = layer.forward(input);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {1, 3}));

  // Output should not be zero (has bias)
  auto arr = output.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(mlx::core::abs(arr));
  mlx::core::eval(sum);
  EXPECT_GT(sum.item<float>(), 0.0f);
}

TEST_F(LayersTest, LinearBatchedInput) {
  Linear layer(8, 16, false);

  // Batched input [32, 8]
  Tensor input = mlxr::graph::ones({32, 8}, mlx::core::float32);

  // Apply linear transformation
  Tensor output = layer.forward(input);

  // Check output shape [32, 16]
  EXPECT_TRUE(shapes_equal(output.shape(), {32, 16}));
}

// ============================================================================
// RotaryEmbedding Tests
// ============================================================================

TEST_F(LayersTest, RotaryEmbeddingConstruction) {
  // Valid construction (even dimension)
  EXPECT_NO_THROW({ RotaryEmbedding rope(64, 2048); });

  // Invalid construction (odd dimension)
  EXPECT_THROW({ RotaryEmbedding rope(63, 2048); }, std::invalid_argument);
}

TEST_F(LayersTest, RotaryEmbeddingForward) {
  int head_dim = 8;
  int seq_len = 4;
  int num_heads = 2;

  RotaryEmbedding rope(head_dim, 128);

  // Create query and key tensors [1, seq_len, num_heads, head_dim]
  Tensor q =
      mlxr::graph::ones({1, seq_len, num_heads, head_dim}, mlx::core::float32);
  Tensor k =
      mlxr::graph::ones({1, seq_len, num_heads, head_dim}, mlx::core::float32);

  // Apply RoPE
  auto [q_out, k_out] = rope.forward(q, k, 0);

  // Check shapes are preserved
  EXPECT_TRUE(shapes_equal(q_out.shape(), {1, seq_len, num_heads, head_dim}));
  EXPECT_TRUE(shapes_equal(k_out.shape(), {1, seq_len, num_heads, head_dim}));

  // Outputs should exist and have correct shape (rotary embeddings applied)
  EXPECT_FALSE(q_out.empty());
  EXPECT_FALSE(k_out.empty());
}

TEST_F(LayersTest, RotaryEmbeddingWithOffset) {
  int head_dim = 8;
  int seq_len = 4;
  int num_heads = 2;

  RotaryEmbedding rope(head_dim, 128);

  // Create query and key tensors
  Tensor q =
      mlxr::graph::ones({1, seq_len, num_heads, head_dim}, mlx::core::float32);
  Tensor k =
      mlxr::graph::ones({1, seq_len, num_heads, head_dim}, mlx::core::float32);

  // Apply RoPE with offset
  auto [q_out1, k_out1] = rope.forward(q, k, 0);
  auto [q_out2, k_out2] = rope.forward(q, k, 10);

  // Outputs with different offsets should differ
  auto q1_arr = q_out1.array();
  auto q2_arr = q_out2.array();
  mlx::core::eval(q1_arr);
  mlx::core::eval(q2_arr);

  EXPECT_NE(q1_arr.data<float>()[0], q2_arr.data<float>()[0]);
}

// ============================================================================
// Attention Layer Tests
// ============================================================================

TEST_F(LayersTest, AttentionConstruction) {
  int hidden_size = 128;
  int num_heads = 8;
  int max_seq_len = 512;

  Attention attn(hidden_size, num_heads, max_seq_len);

  // Check projections are initialized
  EXPECT_FALSE(attn.q_proj().weight().empty());
  EXPECT_FALSE(attn.k_proj().weight().empty());
  EXPECT_FALSE(attn.v_proj().weight().empty());
  EXPECT_FALSE(attn.o_proj().weight().empty());

  // Check projection dimensions
  EXPECT_TRUE(
      shapes_equal(attn.q_proj().weight().shape(), {hidden_size, hidden_size}));
}

TEST_F(LayersTest, AttentionForward) {
  int hidden_size = 64;
  int num_heads = 4;
  int seq_len = 8;
  int batch_size = 2;

  Attention attn(hidden_size, num_heads, 512);

  // Create input [batch, seq_len, hidden_size]
  Tensor input =
      mlxr::graph::ones({batch_size, seq_len, hidden_size}, mlx::core::float32);

  // Apply attention (no mask)
  Tensor output = attn.forward(input);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {batch_size, seq_len, hidden_size}));
}

TEST_F(LayersTest, AttentionWithMask) {
  int hidden_size = 64;
  int num_heads = 4;
  int seq_len = 8;

  Attention attn(hidden_size, num_heads, 512);

  // Create input [1, seq_len, hidden_size]
  Tensor input =
      mlxr::graph::ones({1, seq_len, hidden_size}, mlx::core::float32);

  // Create causal mask [seq_len, seq_len]
  auto mask_arr = mlx::core::triu(
      mlx::core::full({seq_len, seq_len}, -1e9f, mlx::core::float32),
      1  // diagonal offset
  );
  Tensor mask(mask_arr);

  // Apply attention with mask
  Tensor output = attn.forward(input, &mask);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {1, seq_len, hidden_size}));
}

TEST_F(LayersTest, AttentionSingleToken) {
  int hidden_size = 64;
  int num_heads = 4;

  Attention attn(hidden_size, num_heads, 512);

  // Single token input [1, 1, hidden_size]
  Tensor input = mlxr::graph::ones({1, 1, hidden_size}, mlx::core::float32);

  // Apply attention
  Tensor output = attn.forward(input);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {1, 1, hidden_size}));
}

// ============================================================================
// MLP Layer Tests
// ============================================================================

TEST_F(LayersTest, MLPConstruction) {
  int hidden_size = 128;
  int intermediate_size = 512;

  MLP mlp(hidden_size, intermediate_size);

  // Check projections are initialized
  EXPECT_FALSE(mlp.gate_proj().weight().empty());
  EXPECT_FALSE(mlp.up_proj().weight().empty());
  EXPECT_FALSE(mlp.down_proj().weight().empty());

  // Check dimensions
  EXPECT_TRUE(shapes_equal(mlp.gate_proj().weight().shape(),
                           {intermediate_size, hidden_size}));
  EXPECT_TRUE(shapes_equal(mlp.down_proj().weight().shape(),
                           {hidden_size, intermediate_size}));
}

TEST_F(LayersTest, MLPForward) {
  int hidden_size = 64;
  int intermediate_size = 256;
  int seq_len = 8;

  MLP mlp(hidden_size, intermediate_size);

  // Create input [1, seq_len, hidden_size]
  Tensor input =
      mlxr::graph::ones({1, seq_len, hidden_size}, mlx::core::float32);

  // Apply MLP
  Tensor output = mlp.forward(input);

  // Check output shape (should match input)
  EXPECT_TRUE(shapes_equal(output.shape(), {1, seq_len, hidden_size}));

  // Output should not be zero
  auto arr = output.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(mlx::core::abs(arr));
  mlx::core::eval(sum);
  EXPECT_GT(sum.item<float>(), 0.0f);
}

TEST_F(LayersTest, MLPSwiGLUActivation) {
  int hidden_size = 32;
  int intermediate_size = 128;

  MLP mlp(hidden_size, intermediate_size);

  // Create two different inputs
  std::vector<float> data1(hidden_size, 1.0f);
  std::vector<float> data2(hidden_size, 2.0f);

  Tensor input1 = mlxr::graph::from_data(data1.data(), {1, 1, hidden_size},
                                         mlx::core::float32);
  Tensor input2 = mlxr::graph::from_data(data2.data(), {1, 1, hidden_size},
                                         mlx::core::float32);

  // Apply MLP
  Tensor output1 = mlp.forward(input1);
  Tensor output2 = mlp.forward(input2);

  // Outputs should be different (non-linear activation)
  auto arr1 = output1.array();
  auto arr2 = output2.array();
  mlx::core::eval(arr1);
  mlx::core::eval(arr2);

  // They shouldn't be exactly scaled versions of each other
  // (which would be the case for linear activation)
  float val1 = arr1.data<float>()[0];
  float val2 = arr2.data<float>()[0];
  EXPECT_NE(val2, val1 * 2.0f);
}

// ============================================================================
// TransformerBlock Tests
// ============================================================================

TEST_F(LayersTest, TransformerBlockConstruction) {
  int hidden_size = 128;
  int num_heads = 8;
  int intermediate_size = 512;
  int max_seq_len = 512;

  TransformerBlock block(hidden_size, num_heads, intermediate_size,
                         max_seq_len);

  // Check components are initialized
  EXPECT_FALSE(block.input_layernorm().weight().empty());
  EXPECT_FALSE(block.post_attention_layernorm().weight().empty());
}

TEST_F(LayersTest, TransformerBlockForward) {
  int hidden_size = 64;
  int num_heads = 4;
  int intermediate_size = 256;
  int seq_len = 8;

  TransformerBlock block(hidden_size, num_heads, intermediate_size, 512);

  // Create input [1, seq_len, hidden_size]
  Tensor input =
      mlxr::graph::ones({1, seq_len, hidden_size}, mlx::core::float32);

  // Apply transformer block
  Tensor output = block.forward(input);

  // Check output shape matches input
  EXPECT_TRUE(shapes_equal(output.shape(), {1, seq_len, hidden_size}));

  // Output should be different from input
  auto in_arr = input.array();
  auto out_arr = output.array();
  mlx::core::eval(in_arr);
  mlx::core::eval(out_arr);

  EXPECT_NE(out_arr.data<float>()[0], 1.0f);
}

TEST_F(LayersTest, TransformerBlockWithMask) {
  int hidden_size = 64;
  int num_heads = 4;
  int intermediate_size = 256;
  int seq_len = 8;

  TransformerBlock block(hidden_size, num_heads, intermediate_size, 512);

  // Create input
  Tensor input =
      mlxr::graph::ones({1, seq_len, hidden_size}, mlx::core::float32);

  // Create causal mask
  auto mask_arr = mlx::core::triu(
      mlx::core::full({seq_len, seq_len}, -1e9f, mlx::core::float32), 1);
  Tensor mask(mask_arr);

  // Apply transformer block with mask
  Tensor output = block.forward(input, &mask);

  // Check output shape
  EXPECT_TRUE(shapes_equal(output.shape(), {1, seq_len, hidden_size}));
}

TEST_F(LayersTest, TransformerBlockResidualConnection) {
  int hidden_size = 64;
  int num_heads = 4;
  int intermediate_size = 256;

  TransformerBlock block(hidden_size, num_heads, intermediate_size, 512);

  // Create input [1, 1, hidden_size]
  Tensor input = mlxr::graph::ones({1, 1, hidden_size}, mlx::core::float32);

  // Apply transformer block
  Tensor output = block.forward(input);

  // Output magnitude should be significantly different from input
  // due to residual connections and normalization
  auto out_arr = output.array();
  mlx::core::eval(out_arr);
  auto out_sum = mlx::core::sum(mlx::core::abs(out_arr));
  mlx::core::eval(out_sum);

  EXPECT_GT(out_sum.item<float>(), 0.0f);
}

// ============================================================================
// Grouped Query Attention (GQA) Tests
// ============================================================================

TEST_F(LayersTest, GQAAttentionConstruction) {
  int hidden_size = 2048;
  int num_heads = 32;
  int num_kv_heads = 4;  // GQA: 4 KV heads, 32 query heads
  int max_seq_len = 2048;

  Attention attn(hidden_size, num_heads, max_seq_len, num_kv_heads);

  // Check Q projection uses full hidden_size
  EXPECT_TRUE(
      shapes_equal(attn.q_proj().weight().shape(), {hidden_size, hidden_size}));

  // Check K and V projections use num_kv_heads * head_dim
  int head_dim = hidden_size / num_heads;  // 64
  int kv_dim = num_kv_heads * head_dim;    // 256
  EXPECT_TRUE(
      shapes_equal(attn.k_proj().weight().shape(), {kv_dim, hidden_size}));
  EXPECT_TRUE(
      shapes_equal(attn.v_proj().weight().shape(), {kv_dim, hidden_size}));
}

TEST_F(LayersTest, GQAAttentionForwardNoReshapeError) {
  // Test case that previously triggered: "[reshape] Cannot reshape array of
  // size 2304 into shape (1,9,32,64)" This validates the fix for MLX lazy
  // evaluation creating non-contiguous tensors

  int hidden_size = 2048;
  int num_heads = 32;
  int num_kv_heads = 4;  // GQA configuration like TinyLlama
  int seq_len = 9;       // Same sequence length that caused the original error
  int batch_size = 1;

  Attention attn(hidden_size, num_heads, 2048, num_kv_heads);

  // Create input [batch, seq_len, hidden_size]
  Tensor input =
      mlxr::graph::ones({batch_size, seq_len, hidden_size}, mlx::core::float32);

  // This should NOT throw a reshape error
  EXPECT_NO_THROW({
    Tensor output = attn.forward(input);

    // Verify output shape is correct
    EXPECT_TRUE(
        shapes_equal(output.shape(), {batch_size, seq_len, hidden_size}));

    // Verify output is not all zeros
    auto arr = output.array();
    mlx::core::eval(arr);
    auto sum = mlx::core::sum(mlx::core::abs(arr));
    mlx::core::eval(sum);
    EXPECT_GT(sum.item<float>(), 0.0f);
  });
}

TEST_F(LayersTest, GQAAttentionWithKVCache) {
  // Test that GQA attention works correctly with KV cache (multi-step
  // generation)
  int hidden_size = 2048;
  int num_heads = 32;
  int num_kv_heads = 4;
  int prefill_len = 9;
  int batch_size = 1;

  Attention attn(hidden_size, num_heads, 2048, num_kv_heads);

  // Step 1: Prefill with multiple tokens
  Tensor prefill_input = mlxr::graph::ones(
      {batch_size, prefill_len, hidden_size}, mlx::core::float32);

  // Create KV cache
  mlxr::graph::KVCache kv_cache;
  kv_cache.cached_length = 0;

  // First forward pass (prefill)
  Tensor prefill_output;
  EXPECT_NO_THROW({
    prefill_output = attn.forward(prefill_input, nullptr, &kv_cache, 0);
    EXPECT_TRUE(shapes_equal(prefill_output.shape(),
                             {batch_size, prefill_len, hidden_size}));
  });

  // Step 2: Decode single token using cache
  Tensor decode_input =
      mlxr::graph::ones({batch_size, 1, hidden_size}, mlx::core::float32);

  // Mark cache as initialized
  kv_cache.cached_length = prefill_len;

  // Second forward pass (decode with cache)
  Tensor decode_output;
  EXPECT_NO_THROW({
    decode_output = attn.forward(decode_input, nullptr, &kv_cache, 0);
    EXPECT_TRUE(
        shapes_equal(decode_output.shape(), {batch_size, 1, hidden_size}));
  });

  // Verify cache was updated (should have prefill_len + 1 tokens now)
  // Cache stores repeated K/V (32 heads instead of 4)
  ASSERT_FALSE(kv_cache.layer_caches.empty());
  ASSERT_FALSE(kv_cache.layer_caches[0].first.empty());

  auto cached_k_shape = kv_cache.layer_caches[0].first.shape();
  // Shape should be [batch, num_heads, total_seq_len, head_dim]
  EXPECT_EQ(cached_k_shape[0], batch_size);
  EXPECT_EQ(cached_k_shape[1], num_heads);  // 32, not 4 (already repeated)
  EXPECT_EQ(cached_k_shape[2], prefill_len + 1);  // prefill + decode token
  EXPECT_EQ(cached_k_shape[3], hidden_size / num_heads);  // head_dim = 64
}

TEST_F(LayersTest, GQAAttentionMultipleDecodeSteps) {
  // Test multiple decode steps to ensure cache concatenation works correctly
  int hidden_size = 256;  // Smaller for faster tests
  int num_heads = 8;
  int num_kv_heads = 2;
  int batch_size = 1;

  Attention attn(hidden_size, num_heads, 512, num_kv_heads);

  // Prefill
  Tensor prefill_input =
      mlxr::graph::ones({batch_size, 5, hidden_size}, mlx::core::float32);
  mlxr::graph::KVCache kv_cache;
  kv_cache.cached_length = 0;

  attn.forward(prefill_input, nullptr, &kv_cache, 0);
  kv_cache.cached_length = 5;

  // Decode 3 tokens sequentially
  for (int i = 0; i < 3; ++i) {
    Tensor decode_input =
        mlxr::graph::ones({batch_size, 1, hidden_size}, mlx::core::float32);

    EXPECT_NO_THROW({
      Tensor output = attn.forward(decode_input, nullptr, &kv_cache, 0);
      EXPECT_TRUE(shapes_equal(output.shape(), {batch_size, 1, hidden_size}));
    });

    kv_cache.cached_length += 1;

    // Verify cache length increases
    auto cached_k_shape = kv_cache.layer_caches[0].first.shape();
    EXPECT_EQ(cached_k_shape[2], 5 + i + 1);  // prefill + decode tokens
  }

  // Final cache should have 8 tokens (5 prefill + 3 decode)
  auto final_k_shape = kv_cache.layer_caches[0].first.shape();
  EXPECT_EQ(final_k_shape[2], 8);
}

TEST_F(LayersTest, GQAAttentionHeadGroupRatio) {
  // Test different GQA configurations (num_heads must be divisible by
  // num_kv_heads)

  // Valid: 32 query heads, 4 KV heads (8:1 ratio)
  EXPECT_NO_THROW({ Attention attn(2048, 32, 2048, 4); });

  // Valid: 64 query heads, 8 KV heads (8:1 ratio) - Llama-2-70B
  EXPECT_NO_THROW({ Attention attn(8192, 64, 4096, 8); });

  // Valid: 32 query heads, 8 KV heads (4:1 ratio) - Mistral
  EXPECT_NO_THROW({ Attention attn(4096, 32, 8192, 8); });

  // Invalid: num_heads not divisible by num_kv_heads
  EXPECT_THROW(
      {
        Attention attn(512, 7, 512, 4);  // 7 % 4 != 0
      },
      std::invalid_argument);
}

TEST_F(LayersTest, GQATensorEvaluationFix) {
  // This test specifically validates the fix for non-contiguous tensors
  // The fix adds mlx::core::eval() calls after repeat operations

  int hidden_size = 2048;
  int num_heads = 32;
  int num_kv_heads = 4;
  int seq_len = 9;

  Attention attn(hidden_size, num_heads, 2048, num_kv_heads);
  Tensor input =
      mlxr::graph::ones({1, seq_len, hidden_size}, mlx::core::float32);

  // Without the eval() fix, this would fail with:
  // "[reshape] Cannot reshape array of size 2304 into shape (1,9,32,64)"
  //
  // The fix ensures:
  // 1. After repeat(k, 8, 1): k goes from [1,4,9,64] to [1,32,9,64]
  // 2. mlx::core::eval() forces materialization to contiguous memory
  // 3. Subsequent operations work correctly

  Tensor output;
  ASSERT_NO_THROW({ output = attn.forward(input); });

  // Verify the output is valid and has correct shape
  EXPECT_FALSE(output.empty());
  EXPECT_TRUE(shapes_equal(output.shape(), {1, seq_len, hidden_size}));

  // Verify output values are finite (no NaN or Inf)
  auto arr = output.array();
  mlx::core::eval(arr);

  bool all_finite = true;
  for (int i = 0; i < std::min(100, seq_len * hidden_size); ++i) {
    if (!std::isfinite(arr.data<float>()[i])) {
      all_finite = false;
      break;
    }
  }
  EXPECT_TRUE(all_finite);
}
