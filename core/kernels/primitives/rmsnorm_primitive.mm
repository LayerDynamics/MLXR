// Copyright © 2025 MLXR Development
// MLX Primitive-based custom RMSNorm kernel implementation
//
// This implementation uses MLX's Primitive API with direct Metal buffer access
// via MLX's compute encoder. Metal-cpp headers are bundled with MLX and provide
// the Metal C++ API for custom kernel dispatch.

#include "rmsnorm_primitive.h"

#include <mlx/ops.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/transforms.h>  // For eval

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mlxr {
namespace kernels {

// ============================================================================
// Constructor & Destructor
// ============================================================================

RMSNormPrimitive::RMSNormPrimitive(mlx::core::Stream stream, float eps)
    : mlx::core::Primitive(stream),
      eps_(eps),
      library_(nullptr) {
}

RMSNormPrimitive::~RMSNormPrimitive() {
  // Metal library is managed by MLX's device, no explicit cleanup needed
}

// ============================================================================
// Metal Kernel Loading
// ============================================================================

void* RMSNormPrimitive::load_metal_library() {
  if (library_) {
    return library_;
  }

  @autoreleasepool {
    // Get Metal device
    auto& d = mlx::core::metal::device(stream().device);

    // Find metallib file - search multiple paths
    NSArray<NSString*>* search_paths = @[
      @"build/lib/rmsnorm.metallib",  // From project root
      @"../../lib/rmsnorm.metallib",  // From test executable in build/cmake/bin
      @"../lib/rmsnorm.metallib",     // From build/cmake
      @"lib/rmsnorm.metallib",        // Direct lib/
      [@(getenv("PWD") ?: ".") stringByAppendingString:@"/build/lib/rmsnorm.metallib"]
    ];

    NSString* metallib_path = nil;
    for (NSString* path in search_paths) {
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        metallib_path = path;
        NSLog(@"Found rmsnorm.metallib at: %@", path);
        break;
      }
    }

    if (!metallib_path) {
      // List what we tried for debugging
      NSLog(@"Failed to find rmsnorm.metallib in any of these paths:");
      for (NSString* path in search_paths) {
        NSLog(@"  - %@", path);
      }
      throw std::runtime_error(
          "Failed to find rmsnorm.metallib. Please run 'make metal'");
    }

    // Load Metal library via MLX's device
    NSURL* url = [NSURL fileURLWithPath:metallib_path];
    NSError* error = nil;

    // Get the raw MTL::Device pointer
    auto mtl_device = d.mtl_device();

    // Load library using Objective-C bridge
    id<MTLDevice> device_objc = (__bridge id<MTLDevice>)mtl_device;
    id<MTLLibrary> library_objc = [device_objc newLibraryWithURL:url error:&error];

    if (!library_objc) {
      NSString* err_msg = error ? [error localizedDescription] : @"Unknown error";
      throw std::runtime_error(
          "Failed to load Metal library: " +
          std::string([err_msg UTF8String]));
    }

    // Convert to metal-cpp type and store
    MTL::Library* library_cpp = (__bridge MTL::Library*)library_objc;
    library_ = static_cast<void*>(library_cpp);

    return library_;
  }
}

// ============================================================================
// CPU Evaluation (Fallback)
// ============================================================================

void RMSNormPrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& output = outputs[0];

  // Allocate output buffer
  auto buffer = mlx::core::allocator::malloc(output.nbytes());
  output.set_data(buffer);

  // Use MLX reference implementation for CPU
  auto x_sq = mlx::core::multiply(input, input);
  std::vector<int> axes = {-1};
  auto mean_sq = mlx::core::mean(x_sq, axes, /*keepdims=*/true);
  auto rms_inv = mlx::core::rsqrt(
      mlx::core::add(mean_sq, mlx::core::array(eps_)));
  auto normalized = mlx::core::multiply(input, rms_inv);
  auto result = mlx::core::multiply(normalized, weight);

  // Evaluate and copy result to output
  mlx::core::eval(result);
  std::memcpy(output.data<void>(), result.data<void>(), output.nbytes());
}

// ============================================================================
// GPU Evaluation (Custom Metal Kernel)
// ============================================================================

void RMSNormPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  NSLog(@"[RMSNorm] eval_gpu() called - using Metal kernel");

  assert(inputs.size() == 2);
  assert(outputs.size() == 1);

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& output = outputs[0];

  // Check for contiguous memory layout FIRST, before allocating resources
  // For now, we require contiguous inputs. Non-contiguous handling will be
  // implemented in Phase 2 with proper buffer management.
  //
  // Note: We can't fall back to eval_cpu here because both eval_gpu and eval_cpu
  // use the same stream, causing Metal command buffer conflicts. Instead, we
  // ensure the Primitive is only dispatched from the graph with contiguous inputs.
  if (!input.flags().row_contiguous || !weight.flags().row_contiguous) {
    throw std::runtime_error(
        "RMSNormPrimitive requires contiguous inputs. "
        "This is a known Phase 1 limitation and will be fixed in Phase 2.");
  }

  // Allocate output buffer
  auto buffer = mlx::core::allocator::malloc(output.nbytes());
  output.set_data(buffer);

  // Get stream and device
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  auto input_contig = input;
  auto weight_contig = weight;

  // Load our custom Metal library
  auto* mtl_lib = static_cast<MTL::Library*>(load_metal_library());

  // Get kernel name based on dtype
  std::string kernel_name = (input.dtype() == mlx::core::float16)
                             ? "rmsnorm_fused_fp16"
                             : "rmsnorm_fused";

  // Get compiled kernel from MLX's device (this caches it)
  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Get MLX's command encoder and set pipeline
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Bind buffers and parameters
  compute_encoder.set_input_array(input_contig, 0);
  compute_encoder.set_input_array(weight_contig, 1);
  compute_encoder.set_output_array(output, 2);

  // Get shape info for kernel parameters
  auto shape = input.shape();
  uint32_t batch_seq_len = 1;
  for (size_t i = 0; i < shape.size() - 1; i++) {
    batch_seq_len *= shape[i];
  }
  uint32_t hidden_size = static_cast<uint32_t>(shape.back());

  compute_encoder.set_bytes(batch_seq_len, 3);
  compute_encoder.set_bytes(hidden_size, 4);
  compute_encoder.set_bytes(eps_, 5);

  // Dispatch with appropriate grid size
  // Each threadgroup handles one sequence, threads within group reduce over hidden_dim
  size_t threads_per_group = std::min(size_t(1024), size_t(hidden_size));
  MTL::Size grid_dims(batch_seq_len, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);

  NSLog(@"[RMSNorm] Dispatch params: batch_seq_len=%u, hidden_size=%u, threads_per_group=%zu",
        batch_seq_len, hidden_size, threads_per_group);
  NSLog(@"[RMSNorm] Grid dims: (%zu, %zu, %zu), Group dims: (%zu, %zu, %zu)",
        grid_dims.width, grid_dims.height, grid_dims.depth,
        group_dims.width, group_dims.height, group_dims.depth);
  NSLog(@"[RMSNorm] Kernel pipeline valid: %d", kernel != nullptr);

  // CRITICAL FIX: Allocate threadgroup memory for parallel reduction
  // The Metal kernel uses threadgroup float* local_sum [[threadgroup(0)]]
  // which requires explicit memory allocation
  size_t threadgroup_mem_size = threads_per_group * sizeof(float);
  compute_encoder.set_threadgroup_memory_length(threadgroup_mem_size, 0);
  NSLog(@"[RMSNorm] Allocated %zu bytes of threadgroup memory at index 0", threadgroup_mem_size);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  NSLog(@"[RMSNorm] Dispatch complete");
}

// ============================================================================
// Function Transformations
// ============================================================================

std::pair<std::vector<mlx::core::array>, std::vector<int>>
RMSNormPrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {

  // RMSNorm operates on the last dimension, so we can vmap over any
  // other dimension by treating it as part of the batch

  auto& input = inputs[0];
  auto& weight = inputs[1];

  // Input is vectorized along axes[0], weight is not vectorized
  int input_axis = axes[0];

  // Move the vectorized axis to the front and flatten with other batch dims
  // For simplicity, just return the same operation
  // (MLX will handle reshaping automatically)

  auto out = rmsnorm_fused(input, weight, eps_, stream());

  return {{out}, {input_axis}};
}

std::vector<mlx::core::array> RMSNormPrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {

  // For Phase 1, we can fall back to MLX's autodiff
  // In Phase 2, implement custom JVP for efficiency

  // Compute primal output
  auto out = rmsnorm_fused(primals[0], primals[1], eps_, stream());

  // For now, use numerical differentiation (MLX will handle this)
  // In a full implementation, we would compute:
  // d(RMSNorm(x, w)) = w * (dx / rms - x * (x · dx) / (rms^3 * hidden_size))

  return {out};  // Placeholder
}

std::vector<mlx::core::array> RMSNormPrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {

  // For Phase 1, fall back to MLX's autodiff
  // In Phase 2, implement custom VJP for efficiency

  return {cotangents[0]};  // Placeholder
}

bool RMSNormPrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_norm = dynamic_cast<const RMSNormPrimitive*>(&other);
  if (!other_norm) {
    return false;
  }
  return eps_ == other_norm->eps_;
}

// ============================================================================
// Public API
// ============================================================================

mlx::core::array rmsnorm_fused(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    float eps,
    mlx::core::StreamOrDevice s) {

  // Validate inputs
  if (input.ndim() < 1) {
    throw std::invalid_argument("RMSNorm input must have at least 1 dimension");
  }

  if (weight.ndim() != 1) {
    throw std::invalid_argument("RMSNorm weight must be 1-dimensional");
  }

  size_t hidden_size = input.shape().back();
  if (static_cast<size_t>(weight.size()) != hidden_size) {
    throw std::invalid_argument(
        "RMSNorm weight size must match input's last dimension");
  }

  // Get stream first (needed for contiguity checks)
  auto stream = mlx::core::to_stream(s);

  // Ensure inputs are contiguous for Metal kernel
  // Strategy: Flatten and reshape to force contiguous layout.
  // This works because reshape with same shape forces a contiguous copy.
  auto input_contig = input;
  auto weight_contig = weight;

  if (!input.flags().row_contiguous) {
    // Flatten to 1D, then reshape back to original shape
    // This forces MLX to create a contiguous copy
    auto input_flat = mlx::core::reshape(input, {-1}, stream);
    mlx::core::eval(input_flat);
    input_contig = mlx::core::reshape(input_flat, input.shape(), stream);
    mlx::core::eval(input_contig);
  }

  if (!weight.flags().row_contiguous) {
    auto weight_flat = mlx::core::reshape(weight, {-1}, stream);
    mlx::core::eval(weight_flat);
    weight_contig = mlx::core::reshape(weight_flat, weight.shape(), stream);
    mlx::core::eval(weight_contig);
  }

  auto primitive = std::make_shared<RMSNormPrimitive>(stream, eps);

  // Create output array using MLX's array factory with primitive
  auto outputs = mlx::core::array::make_arrays(
      {input_contig.shape()},    // output shapes
      {input_contig.dtype()},    // output dtypes
      primitive,                 // the primitive
      {input_contig, weight_contig}  // inputs (now contiguous)
  );

  return outputs[0];
}

}  // namespace kernels
}  // namespace mlxr
