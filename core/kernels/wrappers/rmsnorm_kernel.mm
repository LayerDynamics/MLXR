/**
 * @file rmsnorm_kernel.mm
 * @brief Implementation of RMSNorm kernel wrapper
 */

#include "rmsnorm_kernel.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mlx/mlx.h>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace mlxr {
namespace kernels {

RMSNormKernel::RMSNormKernel(const std::string& library_path) {
  // Load the Metal library
  library_ = load_library(library_path);

  // Get pipelines for different kernel variants
  pipeline_fp32_ = get_pipeline(library_, "rmsnorm_fused");
  pipeline_fp16_ = get_pipeline(library_, "rmsnorm_fused_fp16");
  pipeline_residual_ = get_pipeline(library_, "rmsnorm_fused_residual");
}

RMSNormKernel::~RMSNormKernel() {
  // Resources are managed by Metal's ARC, no manual cleanup needed
}

graph::Tensor RMSNormKernel::forward(
    const graph::Tensor& input,
    const graph::Tensor& weight,
    float eps) {
  // Get input shape
  auto input_shape = input.shape();
  if (input_shape.size() < 2) {
    throw std::invalid_argument("RMSNorm input must be at least 2D");
  }

  // Calculate dimensions
  int hidden_size = input_shape[input_shape.size() - 1];
  int batch_seq_len = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_seq_len *= input_shape[i];
  }

  // Verify weight shape
  auto weight_shape = weight.shape();
  if (weight_shape.size() != 1 || weight_shape[0] != hidden_size) {
    throw std::invalid_argument(
        "RMSNorm weight must be 1D with size matching hidden dimension");
  }

  // Get MLX arrays and evaluate
  auto input_arr = input.array();
  auto weight_arr = weight.array();
  mlx::core::eval(input_arr);
  mlx::core::eval(weight_arr);

  // Create output tensor
  auto output_arr = mlx::core::zeros(input_arr.shape(), input_arr.dtype());
  mlx::core::eval(output_arr);

  // Determine which pipeline to use based on dtype
  void* pipeline =
      (input_arr.dtype() == mlx::core::float16) ? pipeline_fp16_ : pipeline_fp32_;

  // Get raw Metal buffers from MLX arrays
  // Note: MLX arrays are backed by Metal buffers on Apple Silicon
  auto input_buffer = input_arr.data<void>();
  auto weight_buffer = weight_arr.data<void>();
  auto output_buffer = output_arr.data<void>();

  // Get command queue
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue();
  id<MTLComputePipelineState> pipeline_state = (__bridge id<MTLComputePipelineState>)pipeline;

  // Create command buffer
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

  // Set pipeline
  [encoder setComputePipelineState:pipeline_state];

  // Set buffers
  [encoder setBuffer:(__bridge id<MTLBuffer>)input_buffer offset:0 atIndex:0];
  [encoder setBuffer:(__bridge id<MTLBuffer>)weight_buffer offset:0 atIndex:1];
  [encoder setBuffer:(__bridge id<MTLBuffer>)output_buffer offset:0 atIndex:2];

  // Set scalar arguments
  uint batch_seq_len_arg = static_cast<uint>(batch_seq_len);
  uint hidden_size_arg = static_cast<uint>(hidden_size);
  [encoder setBytes:&batch_seq_len_arg length:sizeof(uint) atIndex:3];
  [encoder setBytes:&hidden_size_arg length:sizeof(uint) atIndex:4];
  [encoder setBytes:&eps length:sizeof(float) atIndex:5];

  // Set threadgroup memory (for reduction)
  NSUInteger threadgroup_mem_size = [pipeline_state maxTotalThreadsPerThreadgroup] * sizeof(float);
  [encoder setThreadgroupMemoryLength:threadgroup_mem_size atIndex:0];

  // Calculate thread configuration
  // Each threadgroup processes one sequence
  ThreadgroupSize threadgroup_size = calculate_threadgroup_size(pipeline, hidden_size);
  MTLSize mtl_threadgroup_size = MTLSizeMake(threadgroup_size.width,
                                               threadgroup_size.height,
                                               threadgroup_size.depth);
  MTLSize grid_size = MTLSizeMake(batch_seq_len, 1, 1);

  // Dispatch
  [encoder dispatchThreadgroups:grid_size
            threadsPerThreadgroup:mtl_threadgroup_size];

  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  // Return output as Tensor
  return graph::Tensor(output_arr);
}

graph::Tensor RMSNormKernel::forward_with_residual(
    const graph::Tensor& input,
    const graph::Tensor& weight,
    const graph::Tensor& residual,
    float eps) {
  // Get input shape
  auto input_shape = input.shape();
  if (input_shape.size() < 2) {
    throw std::invalid_argument("RMSNorm input must be at least 2D");
  }

  // Verify shapes match
  auto residual_shape = residual.shape();
  if (input_shape != residual_shape) {
    throw std::invalid_argument("Input and residual shapes must match");
  }

  // Calculate dimensions
  int hidden_size = input_shape[input_shape.size() - 1];
  int batch_seq_len = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_seq_len *= input_shape[i];
  }

  // Get MLX arrays and evaluate
  auto input_arr = input.array();
  auto weight_arr = weight.array();
  auto residual_arr = residual.array();
  mlx::core::eval(input_arr);
  mlx::core::eval(weight_arr);
  mlx::core::eval(residual_arr);

  // Create output tensor
  auto output_arr = mlx::core::zeros(input_arr.shape(), input_arr.dtype());
  mlx::core::eval(output_arr);

  // Get raw Metal buffers
  auto input_buffer = input_arr.data<void>();
  auto weight_buffer = weight_arr.data<void>();
  auto residual_buffer = residual_arr.data<void>();
  auto output_buffer = output_arr.data<void>();

  // Get command queue
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue();
  id<MTLComputePipelineState> pipeline_state = (__bridge id<MTLComputePipelineState>)pipeline_residual_;

  // Create command buffer
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

  // Set pipeline
  [encoder setComputePipelineState:pipeline_state];

  // Set buffers
  [encoder setBuffer:(__bridge id<MTLBuffer>)input_buffer offset:0 atIndex:0];
  [encoder setBuffer:(__bridge id<MTLBuffer>)weight_buffer offset:0 atIndex:1];
  [encoder setBuffer:(__bridge id<MTLBuffer>)residual_buffer offset:0 atIndex:2];
  [encoder setBuffer:(__bridge id<MTLBuffer>)output_buffer offset:0 atIndex:3];

  // Set scalar arguments
  uint batch_seq_len_arg = static_cast<uint>(batch_seq_len);
  uint hidden_size_arg = static_cast<uint>(hidden_size);
  [encoder setBytes:&batch_seq_len_arg length:sizeof(uint) atIndex:4];
  [encoder setBytes:&hidden_size_arg length:sizeof(uint) atIndex:5];
  [encoder setBytes:&eps length:sizeof(float) atIndex:6];

  // Set threadgroup memory
  NSUInteger threadgroup_mem_size = [pipeline_state maxTotalThreadsPerThreadgroup] * sizeof(float);
  [encoder setThreadgroupMemoryLength:threadgroup_mem_size atIndex:0];

  // Calculate thread configuration
  ThreadgroupSize threadgroup_size = calculate_threadgroup_size(pipeline_residual_, hidden_size);
  MTLSize mtl_threadgroup_size = MTLSizeMake(threadgroup_size.width,
                                               threadgroup_size.height,
                                               threadgroup_size.depth);
  MTLSize grid_size = MTLSizeMake(batch_seq_len, 1, 1);

  // Dispatch
  [encoder dispatchThreadgroups:grid_size
            threadsPerThreadgroup:mtl_threadgroup_size];

  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  return graph::Tensor(output_arr);
}

// Singleton instance for convenience function
static RMSNormKernel* global_rmsnorm_kernel = nullptr;

static std::string find_metallib(const std::string& name) {
  // Try multiple paths to locate the metallib file
  std::vector<std::string> search_paths = {
    // From test executable: build/cmake/bin -> ../../lib
    "../../lib/" + name,
    // From project root
    "build/lib/" + name,
    // Relative to build directory
    "../lib/" + name,
    "lib/" + name
  };

  for (const auto& path : search_paths) {
    NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
    if ([[NSFileManager defaultManager] fileExistsAtPath:ns_path]) {
      return path;
    }
  }

  // If not found, return the first path and let it fail with a clear error
  return search_paths[0];
}

graph::Tensor fused_rmsnorm(
    const graph::Tensor& input,
    const graph::Tensor& weight,
    float eps) {
  // Initialize kernel on first use
  if (global_rmsnorm_kernel == nullptr) {
    std::string library_path = find_metallib("rmsnorm.metallib");
    global_rmsnorm_kernel = new RMSNormKernel(library_path);
  }

  return global_rmsnorm_kernel->forward(input, weight, eps);
}

}  // namespace kernels
}  // namespace mlxr
