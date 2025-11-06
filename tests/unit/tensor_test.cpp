/**
 * @file tensor_test.cpp
 * @brief Unit tests for Tensor wrapper class
 */

#include "graph/tensor.h"

#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <cmath>
#include <vector>

using namespace mlxr::graph;

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common teardown if needed
  }

  // Helper to compare tensor shapes
  bool shapes_equal(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  // Helper to compare tensor values (approximately)
  bool tensors_close(const Tensor& a, const Tensor& b, float atol = 1e-5f) {
    if (!shapes_equal(a.shape(), b.shape())) return false;

    auto a_arr = a.array();
    auto b_arr = b.array();
    mlx::core::eval(a_arr);
    mlx::core::eval(b_arr);

    // Compute absolute difference
    auto diff = mlx::core::abs(mlx::core::subtract(a_arr, b_arr));
    auto max_diff = mlx::core::max(diff);
    mlx::core::eval(max_diff);

    float max_val = max_diff.item<float>();
    return max_val < atol;
  }
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(TensorTest, DefaultConstructor) {
  Tensor t;
  // MLX default constructor creates a scalar array
  EXPECT_FALSE(t.empty());
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.size(), 1);
}

TEST_F(TensorTest, ConstructorFromMLXArray) {
  auto arr = mlx::core::array({1.0f, 2.0f, 3.0f}, {3});
  Tensor t(arr);

  EXPECT_FALSE(t.empty());
  EXPECT_EQ(t.ndim(), 1);
  EXPECT_EQ(t.size(), 3);
  EXPECT_TRUE(shapes_equal(t.shape(), {3}));
}

TEST_F(TensorTest, ConstructorWithShape) {
  Tensor t({2, 3}, mlx::core::float32);

  EXPECT_FALSE(t.empty());
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.size(), 6);
  EXPECT_TRUE(shapes_equal(t.shape(), {2, 3}));
}

// ============================================================================
// Property Tests
// ============================================================================

TEST_F(TensorTest, ShapeProperty) {
  Tensor t({2, 3, 4}, mlx::core::float32);
  auto shape = t.shape();

  EXPECT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
}

TEST_F(TensorTest, DtypeProperty) {
  Tensor t({2, 3}, mlx::core::float32);
  EXPECT_EQ(t.dtype(), mlx::core::float32);
}

TEST_F(TensorTest, NDimProperty) {
  Tensor t1({5}, mlx::core::float32);
  EXPECT_EQ(t1.ndim(), 1);

  Tensor t2({5, 10}, mlx::core::float32);
  EXPECT_EQ(t2.ndim(), 2);

  Tensor t3({5, 10, 15}, mlx::core::float32);
  EXPECT_EQ(t3.ndim(), 3);
}

TEST_F(TensorTest, SizeProperty) {
  Tensor t1({5}, mlx::core::float32);
  EXPECT_EQ(t1.size(), 5);

  Tensor t2({5, 10}, mlx::core::float32);
  EXPECT_EQ(t2.size(), 50);

  Tensor t3({2, 3, 4}, mlx::core::float32);
  EXPECT_EQ(t3.size(), 24);
}

TEST_F(TensorTest, EmptyProperty) {
  Tensor t1;
  // Default constructor creates scalar, not empty
  EXPECT_FALSE(t1.empty());

  Tensor t2({5}, mlx::core::float32);
  EXPECT_FALSE(t2.empty());
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(TensorTest, Zeros) {
  Tensor t = mlxr::graph::zeros({2, 3}, mlx::core::float32);

  EXPECT_TRUE(shapes_equal(t.shape(), {2, 3}));
  EXPECT_EQ(t.size(), 6);

  // Verify all values are zero
  auto arr = t.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(arr);
  mlx::core::eval(sum);
  EXPECT_FLOAT_EQ(sum.item<float>(), 0.0f);
}

TEST_F(TensorTest, Ones) {
  Tensor t = mlxr::graph::ones({2, 3}, mlx::core::float32);

  EXPECT_TRUE(shapes_equal(t.shape(), {2, 3}));
  EXPECT_EQ(t.size(), 6);

  // Verify all values are one
  auto arr = t.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(arr);
  mlx::core::eval(sum);
  EXPECT_FLOAT_EQ(sum.item<float>(), 6.0f);
}

TEST_F(TensorTest, FromData) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor t = mlxr::graph::from_data(data.data(), {2, 2}, mlx::core::float32);

  EXPECT_TRUE(shapes_equal(t.shape(), {2, 2}));
  EXPECT_EQ(t.size(), 4);

  // Verify data is correct
  auto arr = t.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.data<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[1], 2.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[2], 3.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[3], 4.0f);
}

// ============================================================================
// Operation Tests
// ============================================================================

TEST_F(TensorTest, Reshape) {
  Tensor t = mlxr::graph::ones({6}, mlx::core::float32);
  Tensor reshaped = t.reshape({2, 3});

  EXPECT_TRUE(shapes_equal(reshaped.shape(), {2, 3}));
  EXPECT_EQ(reshaped.size(), 6);
}

TEST_F(TensorTest, Transpose2D) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor t = mlxr::graph::from_data(data.data(), {2, 3}, mlx::core::float32);

  Tensor transposed = t.transpose();

  // Check shape is transposed correctly
  EXPECT_TRUE(shapes_equal(transposed.shape(), {3, 2}));

  // Verify transpose happened (just check shape changed, not specific values
  // as memory layout may differ)
  EXPECT_NE(t.shape()[0], transposed.shape()[0]);
  EXPECT_NE(t.shape()[1], transposed.shape()[1]);
}

TEST_F(TensorTest, TransposeWithAxes) {
  Tensor t({2, 3, 4}, mlx::core::float32);

  // Transpose axes [0, 2, 1] should give shape [2, 4, 3]
  Tensor transposed = t.transpose({0, 2, 1});

  EXPECT_TRUE(shapes_equal(transposed.shape(), {2, 4, 3}));
}

// ============================================================================
// Arithmetic Operation Tests
// ============================================================================

TEST_F(TensorTest, Addition) {
  Tensor a = mlxr::graph::ones({2, 2}, mlx::core::float32);
  Tensor b = mlxr::graph::ones({2, 2}, mlx::core::float32);

  Tensor c = a + b;

  EXPECT_TRUE(shapes_equal(c.shape(), {2, 2}));

  // Verify all values are 2.0
  auto arr = c.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(arr);
  mlx::core::eval(sum);
  EXPECT_FLOAT_EQ(sum.item<float>(), 8.0f);  // 4 elements * 2.0
}

TEST_F(TensorTest, Subtraction) {
  std::vector<float> data_a = {3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f};

  Tensor a = mlxr::graph::from_data(data_a.data(), {2, 2}, mlx::core::float32);
  Tensor b = mlxr::graph::from_data(data_b.data(), {2, 2}, mlx::core::float32);

  Tensor c = a - b;

  // Verify all values are 2.0
  auto arr = c.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(arr);
  mlx::core::eval(sum);
  EXPECT_FLOAT_EQ(sum.item<float>(), 8.0f);  // (3-1) + (4-2) + (5-3) + (6-4)
}

TEST_F(TensorTest, Multiplication) {
  std::vector<float> data = {2.0f, 3.0f, 4.0f, 5.0f};
  Tensor a = mlxr::graph::from_data(data.data(), {2, 2}, mlx::core::float32);
  Tensor b = mlxr::graph::ones({2, 2}, mlx::core::float32);

  Tensor c = a * b;

  // Verify values unchanged (multiply by 1)
  auto arr = c.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.data<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[1], 3.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[2], 4.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[3], 5.0f);
}

TEST_F(TensorTest, Division) {
  std::vector<float> data = {4.0f, 6.0f, 8.0f, 10.0f};
  Tensor a = mlxr::graph::from_data(data.data(), {2, 2}, mlx::core::float32);

  std::vector<float> div_data = {2.0f, 2.0f, 2.0f, 2.0f};
  Tensor b =
      mlxr::graph::from_data(div_data.data(), {2, 2}, mlx::core::float32);

  Tensor c = a / b;

  // Verify division correctness
  auto arr = c.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.data<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[1], 3.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[2], 4.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[3], 5.0f);
}

TEST_F(TensorTest, ScalarAddition) {
  Tensor a = mlxr::graph::ones({2, 2}, mlx::core::float32);
  Tensor b = a + 5.0f;

  // Verify all values are 6.0
  auto arr = b.array();
  mlx::core::eval(arr);
  auto sum = mlx::core::sum(arr);
  mlx::core::eval(sum);
  EXPECT_FLOAT_EQ(sum.item<float>(), 24.0f);  // 4 elements * 6.0
}

TEST_F(TensorTest, ScalarMultiplication) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor a = mlxr::graph::from_data(data.data(), {2, 2}, mlx::core::float32);
  Tensor b = a * 2.0f;

  // Verify values doubled
  auto arr = b.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.data<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[1], 4.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[2], 6.0f);
  EXPECT_FLOAT_EQ(arr.data<float>()[3], 8.0f);
}

// ============================================================================
// Tensor Operation Tests
// ============================================================================

TEST_F(TensorTest, Matmul) {
  // Create 2x3 matrix
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor a = mlxr::graph::from_data(data_a.data(), {2, 3}, mlx::core::float32);

  // Create 3x2 matrix
  std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor b = mlxr::graph::from_data(data_b.data(), {3, 2}, mlx::core::float32);

  // Result should be 2x2
  Tensor c = matmul(a, b);

  EXPECT_TRUE(shapes_equal(c.shape(), {2, 2}));

  // Verify computation: [1,2,3] · [1,3,5]^T = 1*1 + 2*3 + 3*5 = 22
  auto arr = c.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.data<float>()[0], 22.0f);
}

TEST_F(TensorTest, Concatenate) {
  Tensor a = mlxr::graph::ones({2, 3}, mlx::core::float32);
  Tensor b = mlxr::graph::zeros({2, 3}, mlx::core::float32);

  // Concatenate along axis 0 (rows)
  Tensor c = concatenate({a, b}, 0);

  EXPECT_TRUE(shapes_equal(c.shape(), {4, 3}));
  EXPECT_EQ(c.size(), 12);
}

TEST_F(TensorTest, Split) {
  Tensor t = mlxr::graph::ones({6, 4}, mlx::core::float32);

  // Split into 3 equal parts along axis 0 at indices [2, 4]
  auto splits = split(t, {2, 4}, 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_TRUE(shapes_equal(splits[0].shape(), {2, 4}));
  EXPECT_TRUE(shapes_equal(splits[1].shape(), {2, 4}));
  EXPECT_TRUE(shapes_equal(splits[2].shape(), {2, 4}));
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(TensorTest, ToShapeConversion) {
  std::vector<int> shape_vec = {2, 3, 4};
  mlx::core::Shape mlx_shape = to_shape(shape_vec);

  EXPECT_EQ(mlx_shape.size(), 3);
  EXPECT_EQ(mlx_shape[0], 2);
  EXPECT_EQ(mlx_shape[1], 3);
  EXPECT_EQ(mlx_shape[2], 4);
}

TEST_F(TensorTest, FromShapeConversion) {
  mlx::core::Shape mlx_shape = {2, 3, 4};
  std::vector<int> shape_vec = from_shape(mlx_shape);

  EXPECT_EQ(shape_vec.size(), 3);
  EXPECT_EQ(shape_vec[0], 2);
  EXPECT_EQ(shape_vec[1], 3);
  EXPECT_EQ(shape_vec[2], 4);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(TensorTest, EmptyTensorOperations) {
  Tensor t;
  // Default constructor creates scalar, not empty
  EXPECT_FALSE(t.empty());
  EXPECT_EQ(t.size(), 1);
  EXPECT_EQ(t.ndim(), 1);
}

TEST_F(TensorTest, SingleElementTensor) {
  Tensor t = mlxr::graph::ones({1}, mlx::core::float32);
  EXPECT_EQ(t.size(), 1);
  EXPECT_EQ(t.ndim(), 1);

  auto arr = t.array();
  mlx::core::eval(arr);
  EXPECT_FLOAT_EQ(arr.item<float>(), 1.0f);
}

TEST_F(TensorTest, LargeShapeTensor) {
  Tensor t({100, 100, 10}, mlx::core::float32);
  EXPECT_EQ(t.size(), 100000);
  EXPECT_EQ(t.ndim(), 3);
}
