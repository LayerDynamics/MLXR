/**
 * @file rmsnorm.cpp
 * @brief RMSNorm primitive implementation
 */

#include "rmsnorm.h"

#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlxr {
namespace kernels {

namespace {

// Get kernel name based on dtype
std::string get_kernel_name(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::float16:
      return "rmsnorm_fused_fp16";
    case mlx::core::float32:
      return "rmsnorm_fused";
    default:
      throw std::runtime_error("Unsupported dtype for RMSNorm");
  }
}

}  // namespace

void RMSNormPrimitive::eval_metal(const std::vector<mlx::core::array>& inputs,
                                  mlx::core::array& out) {
  // Get stream and device
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  // Ensure inputs are row-contiguous
  std::vector<mlx::core::array> copies;
  auto input = inputs[0];
  auto weight = inputs[1];

  if (!input.flags().row_contiguous) {
    input = mlx::core::copy(input, s);
    copies.push_back(input);
  }
  if (!weight.flags().row_contiguous) {
    weight = mlx::core::copy(weight, s);
    copies.push_back(weight);
  }

  // Allocate output
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));

  // Calculate dimensions
  auto input_shape = input.shape();
  int hidden_size = input_shape.back();
  size_t batch_seq_len = input.size() / hidden_size;

  // Get Metal kernel
  std::string kernel_name = get_kernel_name(input.dtype());

  // Load kernel from metallib
  std::ostringstream kernel_source;
  kernel_source << "rmsnorm";  // Base name for library lookup

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kernel_name, kernel_source.str());

  // Set pipeline state
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set buffer arguments
  compute_encoder.set_input_array(input, 0);
  compute_encoder.set_input_array(weight, 1);
  compute_encoder.set_output_array(out, 2);

  // Set scalar arguments
  uint32_t batch_seq_len_arg = static_cast<uint32_t>(batch_seq_len);
  uint32_t hidden_size_arg = static_cast<uint32_t>(hidden_size);

  compute_encoder.set_bytes(&batch_seq_len_arg, sizeof(uint32_t), 3);
  compute_encoder.set_bytes(&hidden_size_arg, sizeof(uint32_t), 4);
  compute_encoder.set_bytes(&eps_, sizeof(float), 5);

  // Calculate threadgroup size
  // Each threadgroup processes one sequence
  NS::UInteger max_threads = kernel->maxTotalThreadsPerThreadgroup();

  size_t threadgroup_width = std::min(static_cast<size_t>(max_threads),
                                      static_cast<size_t>(hidden_size));

  // Round down to power of 2
  threadgroup_width =
      1 << static_cast<size_t>(std::floor(std::log2(threadgroup_width)));

  // Set threadgroup memory for reduction
  size_t threadgroup_mem_size = max_threads * sizeof(float);
  compute_encoder.set_threadgroup_memory_length(threadgroup_mem_size, 0);

  // Dispatch
  MTL::Size grid_size = MTL::Size(batch_seq_len, 1, 1);
  MTL::Size threadgroup_size = MTL::Size(threadgroup_width, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_size, threadgroup_size);

  // Register temporary arrays for cleanup
  if (!copies.empty()) {
    d.add_temporaries(std::move(copies), s.index);
  }
}

void RMSNormPrimitive::eval_gpu(const std::vector<mlx::core::array>& inputs,
                                mlx::core::array& out) {
  eval_metal(inputs, out);
}

void RMSNormPrimitive::eval_cpu(const std::vector<mlx::core::array>& inputs,
                                mlx::core::array& out) {
  // Fallback to MLX reference implementation
  auto input = inputs[0];
  auto weight = inputs[1];

  // Compute x^2
  auto x_sq = mlx::core::multiply(input, input);

  // Compute mean over last dimension
  auto mean_sq = mlx::core::mean(x_sq, {-1}, /*keepdims=*/true);

  // Compute rsqrt(mean(x^2) + eps)
  auto rms = mlx::core::rsqrt(mlx::core::add(mean_sq, mlx::core::array(eps_)));

  // Normalize: x * rms * weight
  auto normalized = mlx::core::multiply(input, rms);
  out = mlx::core::multiply(normalized, weight);
}

mlx::core::array rmsnorm(const mlx::core::array& input,
                         const mlx::core::array& weight, float eps,
                         mlx::core::StreamOrDevice s) {
  // Validate inputs
  if (input.ndim() < 1) {
    throw std::invalid_argument("RMSNorm input must be at least 1D");
  }

  if (weight.ndim() != 1) {
    throw std::invalid_argument("RMSNorm weight must be 1D");
  }

  int hidden_size = input.shape(-1);
  if (weight.shape(0) != hidden_size) {
    throw std::invalid_argument(
        "RMSNorm weight size must match input's last dimension");
  }

  // Create output array with same shape and dtype as input
  auto out_shape = input.shape();
  auto out = mlx::core::array(out_shape, input.dtype());

  // Create and evaluate primitive
  std::vector<mlx::core::array> inputs = {input, weight};
  auto prim = std::make_shared<RMSNormPrimitive>(mlx::core::to_stream(s), eps);

  out.set_primitive(prim, inputs);

  return out;
}

}  // namespace kernels
}  // namespace mlxr
