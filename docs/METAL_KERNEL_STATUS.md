# Custom Metal Kernels - Implementation Status

## Overview

Custom Metal kernel infrastructure has been implemented for MLXR to enable performance optimizations beyond MLX's built-in operations. This document tracks the status of kernel development and integration challenges.

## Completed Work

### 1. Metal Kernel Base Infrastructure ✅
- **Files**: `core/kernels/wrappers/metal_kernel_base.{h,mm}`
- **Status**: Complete and functional
- **Features**:
  - Singleton Metal device and command queue management
  - Metal library loading with caching
  - Pipeline compilation and caching
  - Threadgroup size calculation utilities
  - C++/Objective-C++ interop via PIMPL pattern (opaque pointers)

### 2. Fused RMSNorm Metal Shader ✅
- **Files**: `core/kernels/metal/rmsnorm.metal`
- **Status**: Complete and compiles successfully
- **Features**:
  - Three kernel variants: FP32, FP16, with residual
  - Fuses 5 operations: x², mean, rsqrt, normalize, weight scale
  - Two-pass algorithm with threadgroup reduction
  - FP32 accumulation for numerical stability
- **Performance**: Expected 2-3x speedup over MLX (not yet validated)

### 3. RMSNorm C++ Wrapper ✅
- **Files**: `core/kernels/wrappers/rmsnorm_kernel.{h,mm}`
- **Status**: Complete with path resolution
- **Features**:
  - `RMSNormKernel` class with Metal-MLX bridge
  - `forward()` and `forward_with_residual()` methods
  - Automatic dtype detection (FP32 vs FP16)
  - Multi-path metallib file discovery

### 4. Build System Integration ✅
- **Files**: `core/CMakeLists.txt`, `scripts/build_metal.sh`
- **Status**: Complete and functional
- **Features**:
  - `USE_CUSTOM_KERNELS` CMake option (default: ON)
  - Automatic Metal shader compilation
  - Conditional compilation in `layers.cpp`
  - Proper Objective-C++ compilation for `.mm` files

## Current Challenge: MLX Buffer Integration ⚠️

### Problem
The RMSNorm kernel compiles and loads successfully, but **segfaults** when executing due to incorrect MLX buffer access.

### Root Cause
MLX manages its own Metal buffers internally and doesn't expose them for direct external use:
- `mlx::core::array::data<void>()` returns a **CPU pointer**, not a Metal buffer
- MLX uses lazy evaluation and manages Metal command buffers internally
- Trying to cast `data<void>()` to `id<MTLBuffer>` causes invalid memory access

### Code Location
```cpp
// rmsnorm_kernel.mm:71-73 (INCORRECT)
auto input_buffer = input_arr.data<void>();
// ...
[encoder setBuffer:(__bridge id<MTLBuffer>)input_buffer offset:0 atIndex:0];
```

This assumes `data<void>()` is a Metal buffer, but it's actually:
1. A CPU-accessible pointer, OR
2. Uninitialized if array hasn't been evaluated to CPU

### Test Output
```
Metal device initialized: Apple M4 Max
Loaded Metal library: ../../lib/rmsnorm.metallib
Created pipeline for kernel: rmsnorm_fused
***Exception: SegFault  0.21 sec
```

Kernels load successfully, but crash on buffer access.

## Possible Solutions

### Option 1: Use MLX's Custom Kernel API (Python)
MLX provides `mx.fast.metal_kernel()` for Python custom kernels:
```python
import mlx.core as mx

kernel_source = "..."  # Metal shader code
kernel = mx.fast.metal_kernel(
    name="rmsnorm_fused",
    input_names=["input", "weight"],
    output_shapes=lambda x, w: [x.shape],
    source=kernel_source
)

output = kernel({"input": x, "weight": w})
```

**Pros**:
- Works within MLX's buffer management
- Python integration available immediately

**Cons**:
- Python-only API (no C++ equivalent documented)
- Less control over buffer management
- Requires Python bindings for MLXR

### Option 2: Contribute Kernels to MLX Core
Submit RMSNorm, RoPE, and quantized matmul as contributions to MLX.

**Pros**:
- Integrated into MLX's buffer management
- Benefits entire MLX community
- Maintained by MLX team

**Cons**:
- Long timeline (review, acceptance, release cycle)
- May not align with MLX's roadmap
- Loss of customization control

### Option 3: Buffer Copy Approach (Immediate Solution)
Copy data to/from MLX arrays using our own Metal buffers:

```cpp
// Allocate our own Metal buffers
id<MTLBuffer> input_buffer = [device newBufferWithLength:size options:...];

// Copy from MLX array to Metal buffer
auto input_arr = input.array();
mlx::core::eval(input_arr);
memcpy([input_buffer contents], input_arr.data<float>(), size);

// Execute kernel on our buffers
[encoder setBuffer:input_buffer offset:0 atIndex:0];
// ... dispatch kernel ...

// Copy result back to MLX array
auto output_arr = mlx::core::zeros(shape, dtype);
mlx::core::eval(output_arr);
memcpy(output_arr.data<float>(), [output_buffer contents], size);
```

**Pros**:
- Works immediately with current infrastructure
- Full control over Metal execution
- No MLX internals knowledge required

**Cons**:
- Memory copy overhead (CPU↔GPU)
- Negates some performance gains
- Increased memory usage

### Option 4: MLX Internals Integration (Advanced)
Dig into MLX source code to access internal Metal buffers:

```cpp
// Hypothetical - requires MLX internals
namespace mlx::core::metal {
  id<MTLBuffer> get_buffer(const array& arr);
}

auto input_buffer = mlx::core::metal::get_buffer(input_arr);
```

**Pros**:
- Zero-copy operation
- Maximum performance
- Clean integration

**Cons**:
- Requires deep MLX internals knowledge
- May break with MLX updates
- Potential ABI compatibility issues
- Unofficial/unsupported API usage

## Research Findings (2025-11-05)

After extensive research into MLX's architecture:

1. **MLX Metal Backend is Internal**: The `mlx/backend/metal/*` APIs are not exposed in the public C++ API
2. **Python Custom Kernels**: MLX provides `mx.fast.metal_kernel()` for Python, added in August 2024
3. **C++ Custom Primitives**: Require inheriting from `mlx::core::Primitive` and using internal APIs
4. **Buffer Access**: MLX manages Metal buffers internally; `array.data<void>()` is CPU-side only

### Key Documentation:
- [MLX Custom Extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [MLX Custom Metal Kernels (Python)](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Conv Implementation Reference](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp)

## Updated Recommendation

**Phase 1 (Current - Completion)**:
Use MLX's built-in operations (already highly optimized with Metal) for RMSNorm, RoPE, etc. MLX's operations are fused and optimized by the MLX team.

**Phase 2 (Future Optimization)**:
Choose one of these paths:

1. **Python Custom Kernels** (Recommended for near-term):
   - Use `mx.fast.metal_kernel()` from Python bindings
   - Requires creating Python interface for MLXR
   - Well-documented and supported by MLX

2. **Contribute to MLX Core**:
   - Submit RMSNorm, RoPE as PRs to MLX
   - Benefits entire community
   - Maintained by MLX team

3. **Wait for C++ Custom Kernel API**:
   - MLX team may expose C++ custom kernel API in future
   - Currently Python-only
   - Monitor MLX releases

**Immediate Action**:
Disable custom kernels and rely on MLX's optimized operations. The Metal shader and infrastructure work provides valuable learning and can be revived when MLX provides better C++ support.

## Files Created

All files are complete and ready for integration once buffer access is resolved:

```
core/kernels/
├── metal/
│   ├── rmsnorm.metal          (213 lines, 3 kernel variants)
│   └── hello_world.metal       (test kernel)
└── wrappers/
    ├── metal_kernel_base.h     (94 lines)
    ├── metal_kernel_base.mm    (179 lines)
    ├── rmsnorm_kernel.h        (90 lines)
    └── rmsnorm_kernel.mm       (251 lines)
```

## Build Status

- ✅ Metal shaders compile: `make metal`
- ✅ C++ wrapper compiles: `make build`
- ✅ Library loading works
- ✅ Pipeline creation succeeds
- ⚠️ Kernel execution segfaults (buffer access issue - see research findings)
- ✅ Custom kernels disabled (USE_CUSTOM_KERNELS=OFF)
- ✅ All 59 C++ unit tests pass with MLX built-in operations

## Phase 1 Status: COMPLETE ✅

Phase 1 (Minimal Inference Core) is now complete using MLX's built-in optimized operations:
- **Tensor API**: Full MLX integration with graph::Tensor wrapper
- **Layers**: RMSNorm, Linear, RotaryEmbedding, Attention, MLP, TransformerBlock
- **Model**: TransformerLM with weight loading from safetensors
- **Inference**: Engine with sampling strategies (greedy, top-p, top-k, temperature)
- **Tokenizer**: SentencePiece integration
- **Tests**: 59 unit tests covering tensor, layers, and tokenizer

## Next Steps (Phase 2+)

1. **Research**: Study MLX Python custom kernel implementation
2. **Prototype**: Implement Option 3 (buffer copy) in separate branch
3. **Benchmark**: Measure performance vs MLX reference
4. **Decide**: Choose integration path based on benchmark results

## References

- MLX Custom Kernels: https://ml-explore.github.io/mlx/build/html/usage/custom_metal.html
- MLX Python API: `mlx.core.fast.metal_kernel()`
- Metal Programming Guide: https://developer.apple.com/metal/

---

**Status**: Phase 1 Complete - Custom kernels deferred to Phase 2
**Last Updated**: 2025-11-05
**Author**: Claude (MLXR Development)

## Testing Commands

The Makefile now includes convenient commands for running C++ tests:

- `make test-cpp` - Run C++ unit tests via CTest
- `make test-cpp-verbose` - Run C++ unit tests with verbose GoogleTest output
- `make test-all` - Run both C++ and Python tests
- `make validate` - Quick validation (Phase 0 + C++ tests)
