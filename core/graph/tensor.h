/**
 * @file tensor.h
 * @brief C++ wrapper for MLX arrays
 *
 * Provides a simple C++ interface to MLX tensor operations for the MLXR
 * inference engine.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace mlxr {
namespace graph {

// Helper function to convert std::vector to mlx::core::Shape
inline mlx::core::Shape to_shape(const std::vector<int>& vec) {
  return mlx::core::Shape(vec.begin(), vec.end());
}

// Helper function to convert mlx::core::Shape to std::vector
inline std::vector<int> from_shape(const mlx::core::Shape& shape) {
  return std::vector<int>(shape.begin(), shape.end());
}

/**
 * @brief Tensor wrapper around MLX array
 *
 * This class provides a convenient C++ interface to MLX arrays,
 * handling memory management and common operations.
 */
class Tensor {
 public:
  /**
   * @brief Construct an empty tensor
   */
  Tensor();

  /**
   * @brief Construct a tensor from an MLX array
   * @param array The MLX array to wrap
   */
  explicit Tensor(const mlx::core::array& array);

  /**
   * @brief Construct a tensor with given shape and dtype
   * @param shape Shape of the tensor
   * @param dtype Data type of the tensor
   */
  Tensor(const std::vector<int>& shape, mlx::core::Dtype dtype);

  /**
   * @brief Get the underlying MLX array
   */
  mlx::core::array& array();
  const mlx::core::array& array() const;

  /**
   * @brief Get tensor shape
   */
  std::vector<int> shape() const;

  /**
   * @brief Get tensor dtype
   */
  mlx::core::Dtype dtype() const;

  /**
   * @brief Get number of dimensions
   */
  int ndim() const;

  /**
   * @brief Get total number of elements
   */
  size_t size() const;

  /**
   * @brief Check if tensor is empty
   */
  bool empty() const;

  /**
   * @brief Reshape tensor
   * @param new_shape New shape
   * @return Reshaped tensor
   */
  Tensor reshape(const std::vector<int>& new_shape) const;

  /**
   * @brief Transpose tensor
   * @param axes Axes permutation (if empty, reverses all axes)
   * @return Transposed tensor
   */
  Tensor transpose(const std::vector<int>& axes = {}) const;

  /**
   * @brief Convert tensor to string representation
   */
  std::string to_string() const;

  /**
   * @brief Evaluate the tensor (force computation)
   */
  void eval();

  // Arithmetic operations
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  // Scalar operations
  Tensor operator+(float scalar) const;
  Tensor operator-(float scalar) const;
  Tensor operator*(float scalar) const;
  Tensor operator/(float scalar) const;

 private:
  mlx::core::array array_;
};

/**
 * @brief Create a tensor filled with zeros
 * @param shape Shape of the tensor
 * @param dtype Data type
 * @return Tensor filled with zeros
 */
Tensor zeros(const std::vector<int>& shape,
             mlx::core::Dtype dtype = mlx::core::float32);

/**
 * @brief Create a tensor filled with ones
 * @param shape Shape of the tensor
 * @param dtype Data type
 * @return Tensor filled with ones
 */
Tensor ones(const std::vector<int>& shape,
            mlx::core::Dtype dtype = mlx::core::float32);

/**
 * @brief Create a tensor from raw data
 * @param data Pointer to data
 * @param shape Shape of the tensor
 * @param dtype Data type
 * @return Tensor containing the data
 */
Tensor from_data(const void* data, const std::vector<int>& shape,
                 mlx::core::Dtype dtype);

/**
 * @brief Matrix multiplication
 * @param a First tensor
 * @param b Second tensor
 * @return Result of a @ b
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Concatenate tensors along an axis
 * @param tensors Vector of tensors to concatenate
 * @param axis Axis along which to concatenate
 * @return Concatenated tensor
 */
Tensor concatenate(const std::vector<Tensor>& tensors, int axis = 0);

/**
 * @brief Split tensor along an axis
 * @param tensor Tensor to split
 * @param indices_or_sections Split indices or number of sections
 * @param axis Axis along which to split
 * @return Vector of split tensors
 */
std::vector<Tensor> split(const Tensor& tensor,
                          const std::vector<int>& indices_or_sections,
                          int axis = 0);

}  // namespace graph
}  // namespace mlxr
