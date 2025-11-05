/**
 * @file tensor.cpp
 * @brief Implementation of MLX tensor wrapper
 */

#include "tensor.h"

#include <sstream>

namespace mlxr {
namespace graph {

Tensor::Tensor() : array_(mlx::core::array()) {}

Tensor::Tensor(const mlx::core::array& array) : array_(array) {}

Tensor::Tensor(const std::vector<int>& shape, mlx::core::Dtype dtype)
    : array_(mlx::core::zeros(shape, dtype)) {}

mlx::core::array& Tensor::array() { return array_; }

const mlx::core::array& Tensor::array() const { return array_; }

std::vector<int> Tensor::shape() const { return array_.shape(); }

mlx::core::Dtype Tensor::dtype() const { return array_.dtype(); }

int Tensor::ndim() const { return array_.ndim(); }

size_t Tensor::size() const { return array_.size(); }

bool Tensor::empty() const { return array_.size() == 0; }

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
  return Tensor(mlx::core::reshape(array_, new_shape));
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
  if (axes.empty()) {
    return Tensor(mlx::core::transpose(array_));
  } else {
    return Tensor(mlx::core::transpose(array_, axes));
  }
}

std::string Tensor::to_string() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  auto s = shape();
  for (size_t i = 0; i < s.size(); ++i) {
    oss << s[i];
    if (i < s.size() - 1) oss << ", ";
  }
  oss << "], dtype=" << dtype() << ")";
  return oss.str();
}

void Tensor::eval() { mlx::core::eval(array_); }

// Arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const {
  return Tensor(mlx::core::add(array_, other.array_));
}

Tensor Tensor::operator-(const Tensor& other) const {
  return Tensor(mlx::core::subtract(array_, other.array_));
}

Tensor Tensor::operator*(const Tensor& other) const {
  return Tensor(mlx::core::multiply(array_, other.array_));
}

Tensor Tensor::operator/(const Tensor& other) const {
  return Tensor(mlx::core::divide(array_, other.array_));
}

// Scalar operations
Tensor Tensor::operator+(float scalar) const {
  return Tensor(mlx::core::add(array_, mlx::core::array(scalar)));
}

Tensor Tensor::operator-(float scalar) const {
  return Tensor(mlx::core::subtract(array_, mlx::core::array(scalar)));
}

Tensor Tensor::operator*(float scalar) const {
  return Tensor(mlx::core::multiply(array_, mlx::core::array(scalar)));
}

Tensor Tensor::operator/(float scalar) const {
  return Tensor(mlx::core::divide(array_, mlx::core::array(scalar)));
}

// Factory functions
Tensor zeros(const std::vector<int>& shape, mlx::core::Dtype dtype) {
  return Tensor(mlx::core::zeros(shape, dtype));
}

Tensor ones(const std::vector<int>& shape, mlx::core::Dtype dtype) {
  return Tensor(mlx::core::ones(shape, dtype));
}

Tensor from_data(const void* data, const std::vector<int>& shape,
                 mlx::core::Dtype dtype) {
  return Tensor(mlx::core::array(data, shape, dtype));
}

// Operations
Tensor matmul(const Tensor& a, const Tensor& b) {
  return Tensor(mlx::core::matmul(a.array(), b.array()));
}

Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
  std::vector<mlx::core::array> arrays;
  arrays.reserve(tensors.size());
  for (const auto& t : tensors) {
    arrays.push_back(t.array());
  }
  return Tensor(mlx::core::concatenate(arrays, axis));
}

std::vector<Tensor> split(const Tensor& tensor,
                          const std::vector<int>& indices_or_sections,
                          int axis) {
  auto arrays = mlx::core::split(tensor.array(), indices_or_sections, axis);
  std::vector<Tensor> result;
  result.reserve(arrays.size());
  for (const auto& arr : arrays) {
    result.push_back(Tensor(arr));
  }
  return result;
}

}  // namespace graph
}  // namespace mlxr
