// Copyright © 2025 MLXR Development
// SwiGLU MLP Primitive Implementation

#include "swiglu_mlp_primitive.h"

#include <cmath>
#include <stdexcept>
#include <sstream>

// Metal-cpp headers
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// MLX device and computation
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/ops.h>

namespace mlxr {
namespace kernels {

//------------------------------------------------------------------------------
// SwiGLUMLPPrimitive Implementation
//------------------------------------------------------------------------------

SwiGLUMLPPrimitive::SwiGLUMLPPrimitive(
    mlx::core::Stream stream,
    int M,
    int hidden_size,
    int intermediate_size,
    bool has_bias)
    : mlx::core::Primitive(stream),
      M_(M),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      has_bias_(has_bias),
      library_(nullptr) {}

SwiGLUMLPPrimitive::~SwiGLUMLPPrimitive() {
  if (library_) {
    auto* lib = static_cast<MTL::Library*>(library_);
    lib->release();
  }
}

void SwiGLUMLPPrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Inputs: [input, w_gate, w_up, w_down, bias_gate (opt), bias_up (opt), bias_down (opt)]
  const auto& input = inputs[0];       // [M, hidden_size]
  const auto& w_gate = inputs[1];      // [intermediate_size, hidden_size]
  const auto& w_up = inputs[2];        // [intermediate_size, hidden_size]
  const auto& w_down = inputs[3];      // [hidden_size, intermediate_size]

  // Gate projection: [M, intermediate_size]
  auto gate = mlx::core::matmul(input, mlx::core::transpose(w_gate));
  if (has_bias_ && inputs.size() > 4) {
    gate = mlx::core::add(gate, inputs[4]);  // Add bias_gate
  }

  // Up projection: [M, intermediate_size]
  auto up = mlx::core::matmul(input, mlx::core::transpose(w_up));
  if (has_bias_ && inputs.size() > 5) {
    up = mlx::core::add(up, inputs[5]);  // Add bias_up
  }

  // Apply SwiGLU: swish(gate) * up
  // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
  auto gate_sigmoid = mlx::core::sigmoid(gate);
  auto gate_swish = mlx::core::multiply(gate, gate_sigmoid);
  auto activated = mlx::core::multiply(gate_swish, up);

  // Down projection: [M, hidden_size]
  auto output = mlx::core::matmul(activated, mlx::core::transpose(w_down));
  if (has_bias_ && inputs.size() > 6) {
    output = mlx::core::add(output, inputs[6]);  // Add bias_down
  }

  outputs[0] = output;
}

void SwiGLUMLPPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Inputs
  const auto& input = inputs[0];       // [M, hidden_size]
  const auto& w_gate = inputs[1];      // [intermediate_size, hidden_size]
  const auto& w_up = inputs[2];        // [intermediate_size, hidden_size]
  const auto& w_down = inputs[3];      // [hidden_size, intermediate_size]
  const mlx::core::array* bias_gate = (has_bias_ && inputs.size() > 4) ? &inputs[4] : nullptr;
  const mlx::core::array* bias_up = (has_bias_ && inputs.size() > 5) ? &inputs[5] : nullptr;
  const mlx::core::array* bias_down = (has_bias_ && inputs.size() > 6) ? &inputs[6] : nullptr;
  auto& output = outputs[0];

  // Allocate output
  output = mlx::core::array({M_, hidden_size_}, mlx::core::float16);

  // Load Metal library
  auto* mtl_lib = static_cast<MTL::Library*>(load_metal_library());

  // Get device and kernel
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  std::string kernel_name = has_bias_ ? "swiglu_mlp_fused_bias" : "swiglu_mlp_fused";

  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Create command encoder
  auto& compute_encoder = d.get_command_encoder(stream().index);

  // Set kernel
  compute_encoder.set_compute_pipeline_state(kernel);

  // Build arguments struct
  struct SwiGLUArgs {
    const void* input;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    void* output;
    const void* bias_gate;
    const void* bias_up;
    const void* bias_down;
    uint32_t M;
    uint32_t hidden_size;
    uint32_t intermediate_size;
  };

  SwiGLUArgs args;
  args.input = input.data<void>();
  args.w_gate = w_gate.data<void>();
  args.w_up = w_up.data<void>();
  args.w_down = w_down.data<void>();
  args.output = output.data<void>();
  args.bias_gate = bias_gate ? bias_gate->data<void>() : nullptr;
  args.bias_up = bias_up ? bias_up->data<void>() : nullptr;
  args.bias_down = bias_down ? bias_down->data<void>() : nullptr;
  args.M = M_;
  args.hidden_size = hidden_size_;
  args.intermediate_size = intermediate_size_;

  compute_encoder.set_bytes(&args, sizeof(SwiGLUArgs), 0);

  // Calculate threadgroup size and grid size
  // Each threadgroup processes one output row
  uint32_t threads_per_threadgroup = 256;

  // Shared memory requirements
  size_t shared_input_size = hidden_size_ * sizeof(uint16_t);  // fp16
  size_t shared_gate_size = intermediate_size_ * sizeof(uint16_t);
  size_t shared_up_size = intermediate_size_ * sizeof(uint16_t);

  compute_encoder.set_threadgroup_memory_length(shared_input_size, 0);
  compute_encoder.set_threadgroup_memory_length(shared_gate_size, 1);
  compute_encoder.set_threadgroup_memory_length(shared_up_size, 2);

  MTL::Size threadgroup_size = MTL::Size::Make(threads_per_threadgroup, 1, 1);
  MTL::Size grid_size = MTL::Size::Make(M_, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_size, threadgroup_size);
}

std::pair<std::vector<mlx::core::array>, std::vector<int>>
SwiGLUMLPPrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {

  // SwiGLU can be vmapped over the batch dimension
  // Input axes: [input_axis, w_gate_axis, w_up_axis, w_down_axis, ...]

  // Weights should not be batched (axis = -1)
  // Input can be batched over first dimension

  std::vector<mlx::core::array> outputs;
  std::vector<int> output_axes;

  if (axes[0] != -1) {
    // Input is batched, output will be batched on same axis
    output_axes.push_back(axes[0]);
  } else {
    output_axes.push_back(-1);
  }

  return {outputs, output_axes};
}

std::vector<mlx::core::array> SwiGLUMLPPrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {

  // Forward-mode autodiff (optional for Phase 1)
  // For now, throw not implemented
  throw std::runtime_error("SwiGLU jvp not implemented yet");
}

std::vector<mlx::core::array> SwiGLUMLPPrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {

  // Reverse-mode autodiff (optional for Phase 1)
  // For now, throw not implemented
  throw std::runtime_error("SwiGLU vjp not implemented yet");
}

bool SwiGLUMLPPrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_ptr = dynamic_cast<const SwiGLUMLPPrimitive*>(&other);
  if (!other_ptr) {
    return false;
  }

  return M_ == other_ptr->M_ &&
         hidden_size_ == other_ptr->hidden_size_ &&
         intermediate_size_ == other_ptr->intermediate_size_ &&
         has_bias_ == other_ptr->has_bias_;
}

void* SwiGLUMLPPrimitive::load_metal_library() {
  // Get Metal device
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);

  // Try to get kernel from default library
  // In a full implementation, this would load from a specific metallib file
  return nullptr;  // Placeholder - Metal library will be handled by MLX device
}

//------------------------------------------------------------------------------
// Public API Functions
//------------------------------------------------------------------------------

mlx::core::array swiglu_mlp(
    const mlx::core::array& input,
    const mlx::core::array& w_gate,
    const mlx::core::array& w_up,
    const mlx::core::array& w_down,
    const mlx::core::array* bias_gate,
    const mlx::core::array* bias_up,
    const mlx::core::array* bias_down,
    mlx::core::StreamOrDevice s) {

  // Validate shapes
  if (input.ndim() != 2) {
    throw std::invalid_argument("Input must be 2D [M, hidden_size]");
  }

  int M = input.shape(0);
  int hidden_size = input.shape(1);

  if (w_gate.ndim() != 2 || w_gate.shape(1) != hidden_size) {
    throw std::invalid_argument("w_gate shape mismatch");
  }

  int intermediate_size = w_gate.shape(0);

  if (w_up.shape(0) != intermediate_size || w_up.shape(1) != hidden_size) {
    throw std::invalid_argument("w_up shape mismatch");
  }

  if (w_down.shape(0) != hidden_size || w_down.shape(1) != intermediate_size) {
    throw std::invalid_argument("w_down shape mismatch");
  }

  // Build inputs vector
  std::vector<mlx::core::array> inputs = {input, w_gate, w_up, w_down};

  bool has_bias = false;
  if (bias_gate) {
    inputs.push_back(*bias_gate);
    has_bias = true;
  }
  if (bias_up) {
    inputs.push_back(*bias_up);
    has_bias = true;
  }
  if (bias_down) {
    inputs.push_back(*bias_down);
    has_bias = true;
  }

  // Create stream
  auto stream = mlx::core::to_stream(s);

  // Create primitive
  auto primitive = std::make_shared<SwiGLUMLPPrimitive>(
      stream, M, hidden_size, intermediate_size, has_bias);

  // Create output array
  mlx::core::array output = mlx::core::zeros_like(input);

  // Evaluate primitive
  std::vector<mlx::core::array> outputs = {output};

  if (stream.device == mlx::core::Device::gpu) {
    primitive->eval_gpu(inputs, outputs);
  } else {
    primitive->eval_cpu(inputs, outputs);
  }

  return outputs[0];
}

mlx::core::array swish(const mlx::core::array& x, mlx::core::StreamOrDevice s) {
  // swish(x) = x * sigmoid(x)
  return mlx::core::multiply(x, mlx::core::sigmoid(x), s);
}

mlx::core::array swiglu(
    const mlx::core::array& gate,
    const mlx::core::array& up,
    mlx::core::StreamOrDevice s) {
  // swiglu(gate, up) = swish(gate) * up
  auto gate_swish = swish(gate, s);
  return mlx::core::multiply(gate_swish, up, s);
}

}  // namespace kernels
}  // namespace mlxr
