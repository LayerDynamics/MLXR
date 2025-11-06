# MLX-Metal Buffer Integration Implementation Plan

**Status**: In Progress
**Created**: 2025-11-05
**Goal**: Fix buffer access in custom Metal kernels using MLX Primitive architecture

## Research Summary

### Problem Identified
Our current implementation attempts to use `array.data<void>()` (CPU pointer) as a Metal buffer, causing segfaults:

```cpp
// ❌ WRONG - This is a CPU pointer!
auto input_buffer = input_arr.data<void>();
[encoder setBuffer:(__bridge id<MTLBuffer>)input_buffer offset:0 atIndex:0];
```

### Solution Discovered
MLX uses the **Primitive architecture** with compute encoders that properly manage Metal buffers:

```cpp
// ✅ CORRECT - MLX's compute encoder handles Metal buffers internally
compute_encoder.set_input_array(in, 0);
compute_encoder.set_output_array(out, 2);
```

## Architecture Overview

### MLX Primitive Pattern

1. **Primitive Class**: Inherit from `mlx::core::Primitive`
2. **eval_cpu()**: CPU backend implementation
3. **eval_gpu()**: Metal backend implementation with compute encoder
4. **Stream Management**: Device and stream context for async execution
5. **Buffer Management**: Automatic Metal buffer allocation and lifecycle

### Key Components from MLX Source

From `mlx/backend/metal/conv.cpp`:

```cpp
void Convolution::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {

  // 1. Get stream and Metal device
  auto& s = stream();
  auto& d = metal::device(s.device);

  // 2. Ensure contiguous memory layout
  auto in = inputs[0];
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
  }

  // 3. Allocate output buffer
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  // 4. Get compute encoder and pipeline
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // 5. Bind arrays to Metal buffers (THE KEY PART!)
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_input_array(wt, 1);
  compute_encoder.set_output_array(out, 2);

  // 6. Bind constant parameters
  compute_encoder.set_bytes(params, 3);

  // 7. Dispatch threads
  MTL::Size group_dims = MTL::Size(32, 8, 4);
  MTL::Size grid_dims = MTL::Size(grid_x, grid_y, grid_z);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}
```

## Implementation Plan

### Phase 1: Create Primitive Infrastructure

**File**: `core/kernels/primitives/rmsnorm_primitive.h`

```cpp
#pragma once

#include <mlx/mlx.h>

namespace mlxr {
namespace kernels {

class RMSNormPrimitive : public mlx::core::Primitive {
 public:
  explicit RMSNormPrimitive(
      mlx::core::Stream stream,
      float eps);

  // Required overrides
  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override;

  // For graph optimization
  std::pair<std::vector<mlx::core::array>, std::vector<int>>
  vmap(
      const std::vector<mlx::core::array>& inputs,
      const std::vector<int>& axes) override;

  // For autodiff (optional for Phase 1)
  std::vector<mlx::core::array> jvp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mlx::core::array> vjp(
      const std::vector<mlx::core::array>& primals,
      const std::vector<mlx::core::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mlx::core::array>& outputs) override;

  // Identification
  std::string name() const override { return "rmsnorm_fused"; }

  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;

  // Metal kernel state (loaded once)
  void* library_;        // id<MTLLibrary>
  void* pipeline_fp32_;  // id<MTLComputePipelineState>
  void* pipeline_fp16_;  // id<MTLComputePipelineState>
};

// Public API function
mlx::core::array rmsnorm_fused(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    float eps = 1e-6f,
    mlx::core::StreamOrDevice s = {});

}  // namespace kernels
}  // namespace mlxr
```

### Phase 2: Implement eval_gpu with Proper Buffer Access

**File**: `core/kernels/primitives/rmsnorm_primitive.mm`

```cpp
#include "rmsnorm_primitive.h"
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>

namespace mlxr {
namespace kernels {

void RMSNormPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  assert(inputs.size() == 2);  // input, weight
  assert(outputs.size() == 1);

  auto& input = inputs[0];
  auto& weight = inputs[1];
  auto& output = outputs[0];

  // 1. Get stream and Metal device
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  // 2. Ensure contiguous arrays
  std::vector<mlx::core::array> copies;
  auto input_contig = input;
  auto weight_contig = weight;

  if (!input.flags().row_contiguous) {
    input_contig = mlx::core::metal::contiguous_copy_gpu(input, s);
    copies.push_back(input_contig);
  }

  if (!weight.flags().row_contiguous) {
    weight_contig = mlx::core::metal::contiguous_copy_gpu(weight, s);
    copies.push_back(weight_contig);
  }

  // 3. Allocate output buffer
  output.set_data(
      mlx::core::allocator::malloc_or_wait(output.nbytes()));

  // 4. Get shape info
  auto shape = input.shape();
  uint32_t batch_seq_len = 1;
  for (size_t i = 0; i < shape.size() - 1; i++) {
    batch_seq_len *= shape[i];
  }
  uint32_t hidden_size = shape.back();

  // 5. Select kernel based on dtype
  id<MTLComputePipelineState> pipeline;
  if (input.dtype() == mlx::core::float16) {
    pipeline = (__bridge id<MTLComputePipelineState>)pipeline_fp16_;
  } else {
    pipeline = (__bridge id<MTLComputePipelineState>)pipeline_fp32_;
  }

  // 6. Get compute encoder (THE KEY!)
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(pipeline);

  // 7. Bind Metal buffers via MLX's encoder (NOT direct Metal API!)
  compute_encoder.set_input_array(input_contig, 0);   // input buffer
  compute_encoder.set_input_array(weight_contig, 1);  // weight buffer
  compute_encoder.set_output_array(output, 2);        // output buffer

  // 8. Bind constant parameters
  compute_encoder.set_bytes(batch_seq_len, 3);
  compute_encoder.set_bytes(hidden_size, 4);
  compute_encoder.set_bytes(eps_, 5);

  // 9. Calculate thread dispatch dimensions
  NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
  NSUInteger threads_per_group = std::min((uint32_t)max_threads, hidden_size);

  MTL::Size group_dims = MTL::Size(threads_per_group, 1, 1);
  MTL::Size grid_dims = MTL::Size(batch_seq_len, 1, 1);

  // 10. Dispatch kernel
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // 11. Register temporary buffers for cleanup
  d.add_temporaries(std::move(copies), s.index);
}

// Public API implementation
mlx::core::array rmsnorm_fused(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    float eps,
    mlx::core::StreamOrDevice s) {

  // Create output array with same shape and dtype as input
  auto out = mlx::core::array(input.shape(), input.dtype(), nullptr, {});

  // Create primitive and schedule evaluation
  auto primitive = std::make_shared<RMSNormPrimitive>(
      mlx::core::to_stream(s), eps);

  // Build computation graph
  out.set_primitive(primitive);
  primitive->outputs({out});

  return out;
}

}  // namespace kernels
}  // namespace mlxr
```

### Phase 3: Integration with Existing Code

**Update**: `core/graph/layers.cpp`

```cpp
#ifdef USE_CUSTOM_KERNELS
#include "primitives/rmsnorm_primitive.h"
#endif

Tensor RMSNorm::forward(const Tensor& x) {
#ifdef USE_CUSTOM_KERNELS
  // Use MLX Primitive-based custom kernel
  auto result_arr = mlxr::kernels::rmsnorm_fused(
      x.array(), weight_.array(), eps_);
  return Tensor(result_arr);
#else
  // MLX reference implementation
  auto x_arr = x.array();
  auto x_sq = mlx::core::multiply(x_arr, x_arr);
  std::vector<int> axes_vec = {-1};
  auto mean_sq = mlx::core::mean(x_sq, axes_vec, /*keepdims=*/true);
  auto rms = mlx::core::rsqrt(mlx::core::add(mean_sq, mlx::core::array(eps_)));
  auto normalized = mlx::core::multiply(x_arr, rms);
  auto result = mlx::core::multiply(normalized, weight_.array());
  return Tensor(result);
#endif
}
```

### Phase 4: Build System Updates

**Update**: `core/CMakeLists.txt`

```cmake
# Add Primitive-based custom kernels if enabled
if(USE_CUSTOM_KERNELS)
    list(APPEND CORE_SOURCES
        kernels/primitives/rmsnorm_primitive.mm
        # Future: rope_primitive.mm, matmul_primitive.mm
    )
    message(STATUS "Custom Metal kernels enabled (Primitive-based)")
endif()

# Ensure MLX Metal backend headers are available
if(USE_CUSTOM_KERNELS)
    target_include_directories(mlxr_core PRIVATE
        ${MLX_INCLUDE_DIRS}/../  # To access mlx/backend/metal/
    )
endif()
```

## Key Differences from Current Implementation

| Aspect | Current (Wrong) | New (Correct) |
|--------|----------------|---------------|
| **Architecture** | Direct Metal API wrapper | MLX Primitive integration |
| **Buffer Access** | `array.data<void>()` → Metal cast | `compute_encoder.set_input_array()` |
| **Memory Management** | Manual Metal buffer handling | MLX automatic buffer management |
| **Graph Integration** | External wrapper function | Native Primitive in computation graph |
| **Async Execution** | Manual command buffer | MLX stream-based scheduling |
| **Error Handling** | Segfaults on buffer mismatch | Type-safe MLX encoder API |

## Implementation Steps

1. ✅ **Research Complete**: Understand MLX Primitive architecture
2. ✅ **Plan Created**: Document architecture and approach
3. ⏳ **Create Primitive Header**: Define `RMSNormPrimitive` class
4. ⏳ **Implement eval_gpu**: Use compute encoder for buffer access
5. ⏳ **Implement eval_cpu**: Fallback CPU implementation
6. ⏳ **Update Integration**: Modify `layers.cpp` to use Primitive
7. ⏳ **Build and Test**: Verify no segfaults, all tests pass
8. ⏳ **Benchmark**: Compare performance vs MLX reference

## Expected Outcomes

- ✅ No segfaults - proper Metal buffer access via MLX encoder
- ✅ Full integration with MLX computation graph
- ✅ Automatic memory management and cleanup
- ✅ Support for async execution and stream scheduling
- ✅ Type-safe buffer binding
- 🎯 Performance improvement from fused kernel (target: 1.5-2x speedup)

## Testing Strategy

1. **Unit Tests**: Verify numerical correctness against MLX reference
2. **Memory Tests**: Ensure no leaks or invalid buffer access
3. **Performance Tests**: Benchmark vs MLX built-in RMSNorm
4. **Integration Tests**: Run full model inference with custom kernels

## References

- MLX Custom Extensions: https://ml-explore.github.io/mlx/build/html/dev/extensions.html
- MLX Metal Kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX Conv Example: https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp
- METAL_KERNEL_STATUS.md: Current implementation status

---

**Next Action**: Implement `RMSNormPrimitive` class with proper compute encoder usage
