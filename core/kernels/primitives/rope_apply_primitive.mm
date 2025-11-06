// Copyright © 2025 MLXR Development
// MLX Primitive-based RoPE (Rotary Positional Embeddings) kernel

#include "rope_apply_primitive.h"

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

// ========================================
// Helper Functions
// ========================================

const char* rope_scaling_mode_name(RoPEScalingMode mode) {
  switch (mode) {
    case RoPEScalingMode::BASE: return "BASE";
    case RoPEScalingMode::NTK: return "NTK";
    case RoPEScalingMode::YARN: return "YARN";
    case RoPEScalingMode::LINEAR: return "LINEAR";
    default: return "Unknown";
  }
}

float compute_ntk_scaled_base(
    float base,
    float scale,
    int head_dim,
    int orig_context) {
  // NTK-aware RoPE scaling
  // base' = base * ((scale * ctx_len / orig_ctx_len) - (scale - 1))^(d/(d-2))
  float exponent = static_cast<float>(head_dim) / (head_dim - 2);
  float scaling_factor = scale - (scale - 1);
  return base * std::pow(scaling_factor, exponent);
}

std::pair<mlx::core::array, mlx::core::array> compute_rope_tables(
    int max_seq_len,
    int head_dim,
    float base,
    RoPEScalingMode scaling_mode,
    float scale_factor,
    int orig_context) {

  if (head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }

  const int num_pairs = head_dim / 2;

  // Apply scaling to base if needed
  float effective_base = base;
  if (scaling_mode == RoPEScalingMode::NTK) {
    effective_base = compute_ntk_scaled_base(base, scale_factor, head_dim, orig_context);
  }

  // Allocate tables
  std::vector<float> cos_table(max_seq_len * num_pairs);
  std::vector<float> sin_table(max_seq_len * num_pairs);

  // Compute θ for each (position, dimension_pair)
  for (int pos = 0; pos < max_seq_len; pos++) {
    float position = static_cast<float>(pos);

    // Apply linear scaling if needed
    if (scaling_mode == RoPEScalingMode::LINEAR) {
      position /= scale_factor;
    }

    for (int pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
      // Compute frequency: base^(-2i/d)
      float freq_exponent = -2.0f * pair_idx / head_dim;
      float freq = std::pow(effective_base, freq_exponent);

      // Compute θ = position * freq
      float theta = position * freq;

      // Store cos and sin
      int idx = pos * num_pairs + pair_idx;
      cos_table[idx] = std::cos(theta);
      sin_table[idx] = std::sin(theta);
    }
  }

  // Convert to MLX arrays (FP16 for efficiency)
  mlx::core::array cos_arr(cos_table.data(), {max_seq_len, num_pairs}, mlx::core::float32);
  mlx::core::array sin_arr(sin_table.data(), {max_seq_len, num_pairs}, mlx::core::float32);

  // Convert to FP16
  cos_arr = mlx::core::astype(cos_arr, mlx::core::float16);
  sin_arr = mlx::core::astype(sin_arr, mlx::core::float16);

  return {cos_arr, sin_arr};
}

// ========================================
// RoPEApplyPrimitive Implementation
// ========================================

RoPEApplyPrimitive::RoPEApplyPrimitive(
    mlx::core::Stream stream,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    RoPEScalingMode scaling_mode,
    float scale_factor,
    int position_offset,
    bool inplace)
    : mlx::core::Primitive(stream),
      batch_size_(batch_size),
      seq_len_(seq_len),
      num_heads_(num_heads),
      head_dim_(head_dim),
      scaling_mode_(scaling_mode),
      scale_factor_(scale_factor),
      position_offset_(position_offset),
      inplace_(inplace),
      library_(nullptr) {

  if (head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }
}

RoPEApplyPrimitive::~RoPEApplyPrimitive() {
  if (library_ != nullptr) {
    auto* lib = static_cast<MTL::Library*>(library_);
    lib->release();
  }
}

void* RoPEApplyPrimitive::load_metal_library() {
  if (library_ != nullptr) {
    return library_;
  }

  // Get Metal device
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  auto mtl_device = d.mtl_device();

  // Load metallib from build directory
  NS::Error* error = nullptr;
  NS::String* lib_path = NS::String::string(
      "/Users/ryanoboyle/MLXR/build/lib/rope_apply.metallib",
      NS::UTF8StringEncoding);

  auto* lib = mtl_device->newLibrary(lib_path, &error);

  if (lib == nullptr) {
    std::ostringstream err;
    err << "Failed to load rope_apply Metal library: ";
    if (error) {
      err << error->localizedDescription()->utf8String();
    }
    throw std::runtime_error(err.str());
  }

  library_ = lib;
  return library_;
}

bool RoPEApplyPrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_rope = dynamic_cast<const RoPEApplyPrimitive*>(&other);
  if (!other_rope) {
    return false;
  }

  return batch_size_ == other_rope->batch_size_ &&
         seq_len_ == other_rope->seq_len_ &&
         num_heads_ == other_rope->num_heads_ &&
         head_dim_ == other_rope->head_dim_ &&
         scaling_mode_ == other_rope->scaling_mode_ &&
         scale_factor_ == other_rope->scale_factor_ &&
         position_offset_ == other_rope->position_offset_ &&
         inplace_ == other_rope->inplace_;
}

void RoPEApplyPrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Validate inputs
  if (inputs.size() != 4) {
    throw std::runtime_error("RoPEApply requires 4 inputs: input, cos_table, sin_table, positions");
  }

  const auto& input = inputs[0];
  const auto& cos_table = inputs[1];
  const auto& sin_table = inputs[2];
  const auto& positions = inputs[3];
  auto& output = outputs[0];

  // Allocate output if not in-place
  if (!inplace_) {
    // Use mlx::zeros to create output with same shape
    output = mlx::core::zeros_like(input);
  } else {
    output = input;  // Reference input for in-place
  }

  // Get data pointers
  const float* input_data = input.data<float>();
  const float* cos_data = cos_table.data<float>();
  const float* sin_data = sin_table.data<float>();
  const int* pos_data = positions.data<int>();
  float* output_data = inplace_ ? const_cast<float*>(input.data<float>()) : output.data<float>();

  const int num_pairs = head_dim_ / 2;

  // Iterate over all heads
  for (int batch = 0; batch < batch_size_; batch++) {
    for (int seq = 0; seq < seq_len_; seq++) {
      // Get position for this token
      int position;
      if (seq_len_ == 1) {
        position = pos_data[batch];
      } else {
        position = pos_data[batch * seq_len_ + seq];
      }
      position += position_offset_;

      for (int head = 0; head < num_heads_; head++) {
        // Calculate offset for this head
        int head_offset = batch * seq_len_ * num_heads_ * head_dim_ +
                         seq * num_heads_ * head_dim_ +
                         head * head_dim_;

        // Apply RoPE to each dimension pair
        for (int pair = 0; pair < num_pairs; pair++) {
          int even_dim = head_offset + pair * 2;
          int odd_dim = head_offset + pair * 2 + 1;

          float x_even = input_data[even_dim];
          float x_odd = input_data[odd_dim];

          // Load cos/sin from tables
          int table_idx = position * num_pairs + pair;
          float cos_val = cos_data[table_idx];
          float sin_val = sin_data[table_idx];

          // Apply rotation
          float y_even = x_even * cos_val - x_odd * sin_val;
          float y_odd = x_odd * cos_val + x_even * sin_val;

          // Apply linear scaling if needed
          if (scaling_mode_ == RoPEScalingMode::LINEAR) {
            y_even *= scale_factor_;
            y_odd *= scale_factor_;
          }

          output_data[even_dim] = y_even;
          output_data[odd_dim] = y_odd;
        }
      }
    }
  }
}

void RoPEApplyPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Validate inputs
  if (inputs.size() != 4) {
    throw std::runtime_error("RoPEApply requires 4 inputs: input, cos_table, sin_table, positions");
  }

  const auto& input = inputs[0];
  const auto& cos_table = inputs[1];
  const auto& sin_table = inputs[2];
  const auto& positions = inputs[3];
  auto& output = outputs[0];

  // Allocate output if not in-place
  if (!inplace_) {
    output = mlx::core::zeros_like(input);
  } else {
    output = input;  // Reference input for in-place
  }

  // Load Metal library
  auto* mtl_lib = static_cast<MTL::Library*>(load_metal_library());

  // Get device and kernel
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  std::string kernel_name = inplace_ ? "rope_apply_inplace" : "rope_apply";
  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Create command buffer and encoder
  auto& compute_encoder = d.get_command_encoder(stream().index);

  // Set kernel
  compute_encoder.set_compute_pipeline_state(kernel);

  // Compute strides (assuming contiguous layout for Phase 1)
  const int input_batch_stride = seq_len_ * num_heads_ * head_dim_;
  const int input_seq_stride = num_heads_ * head_dim_;
  const int input_head_stride = head_dim_;

  const int output_batch_stride = seq_len_ * num_heads_ * head_dim_;
  const int output_seq_stride = num_heads_ * head_dim_;
  const int output_head_stride = head_dim_;

  // Bind buffers and parameters
  struct RoPEArgs {
    const void* input;
    void* output;
    const void* cos_table;
    const void* sin_table;
    const int* positions;
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t scaling_mode;
    float scale_factor;
    uint32_t position_offset;
    uint32_t input_batch_stride;
    uint32_t input_seq_stride;
    uint32_t input_head_stride;
    uint32_t output_batch_stride;
    uint32_t output_seq_stride;
    uint32_t output_head_stride;
  };

  RoPEArgs args;
  args.input = input.data<void>();
  args.output = inplace_ ? const_cast<void*>(input.data<void>()) : output.data<void>();
  args.cos_table = cos_table.data<void>();
  args.sin_table = sin_table.data<void>();
  args.positions = positions.data<int>();
  args.batch_size = batch_size_;
  args.seq_len = seq_len_;
  args.num_heads = num_heads_;
  args.head_dim = head_dim_;
  args.scaling_mode = static_cast<uint32_t>(scaling_mode_);
  args.scale_factor = scale_factor_;
  args.position_offset = position_offset_;
  args.input_batch_stride = input_batch_stride;
  args.input_seq_stride = input_seq_stride;
  args.input_head_stride = input_head_stride;
  args.output_batch_stride = output_batch_stride;
  args.output_seq_stride = output_seq_stride;
  args.output_head_stride = output_head_stride;

  compute_encoder.set_bytes(&args, sizeof(RoPEArgs), 0);

  // Dispatch
  // Grid: batch_size * seq_len * num_heads (one thread per head)
  // Threadgroup size: head_dim / 2 (one thread per dimension pair)
  const int total_heads = batch_size_ * seq_len_ * num_heads_;
  const int threads_per_group = head_dim_ / 2;

  MTL::Size grid_size(total_heads, 1, 1);
  MTL::Size threadgroup_size(threads_per_group, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_size, threadgroup_size);
}

std::pair<std::vector<mlx::core::array>, std::vector<int>>
RoPEApplyPrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("vmap not yet implemented for RoPEApply");
}

std::vector<mlx::core::array> RoPEApplyPrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("jvp not yet implemented for RoPEApply");
}

std::vector<mlx::core::array> RoPEApplyPrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {
  throw std::runtime_error("vjp not yet implemented for RoPEApply");
}

// ========================================
// Public API
// ========================================

mlx::core::array rope_apply(
    const mlx::core::array& input,
    const mlx::core::array& cos_table,
    const mlx::core::array& sin_table,
    const mlx::core::array& positions,
    RoPEScalingMode scaling_mode,
    float scale_factor,
    int position_offset,
    bool inplace,
    mlx::core::StreamOrDevice s) {

  // Validate input shapes
  if (input.ndim() != 4 && input.ndim() != 3) {
    throw std::runtime_error("Input must be 3D [tokens, num_heads, head_dim] or 4D [batch, seq_len, num_heads, head_dim]");
  }

  int batch_size, seq_len, num_heads, head_dim;

  if (input.ndim() == 4) {
    batch_size = input.shape(0);
    seq_len = input.shape(1);
    num_heads = input.shape(2);
    head_dim = input.shape(3);
  } else {  // ndim == 3
    batch_size = input.shape(0);
    seq_len = 1;
    num_heads = input.shape(1);
    head_dim = input.shape(2);
  }

  if (head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }

  // Validate table shapes
  const int num_pairs = head_dim / 2;
  if (cos_table.ndim() != 2 || cos_table.shape(1) != num_pairs) {
    throw std::runtime_error("cos_table must have shape [max_seq_len, head_dim/2]");
  }
  if (sin_table.ndim() != 2 || sin_table.shape(1) != num_pairs) {
    throw std::runtime_error("sin_table must have shape [max_seq_len, head_dim/2]");
  }

  // Validate positions shape
  if (seq_len == 1) {
    if (positions.ndim() != 1 || positions.shape(0) != batch_size) {
      throw std::runtime_error("For flattened input, positions must have shape [tokens]");
    }
  } else {
    if (positions.ndim() != 2 || positions.shape(0) != batch_size || positions.shape(1) != seq_len) {
      throw std::runtime_error("positions must have shape [batch, seq_len]");
    }
  }

  // Create primitive
  auto stream = mlx::core::to_stream(s);
  auto prim = std::make_shared<RoPEApplyPrimitive>(
      stream, batch_size, seq_len, num_heads, head_dim,
      scaling_mode, scale_factor, position_offset, inplace);

  // Prepare inputs
  std::vector<mlx::core::array> inputs_vec = {input, cos_table, sin_table, positions};

  // Create output array
  mlx::core::array output = inplace ? input : mlx::core::zeros_like(input);

  // Evaluate primitive
  std::vector<mlx::core::array> outputs = {output};

  if (stream.device == mlx::core::Device::gpu) {
    prim->eval_gpu(inputs_vec, outputs);
  } else {
    prim->eval_cpu(inputs_vec, outputs);
  }

  return outputs[0];
}

}  // namespace kernels
}  // namespace mlxr
