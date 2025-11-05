#include <metal_stdlib>
using namespace metal;

// Simple vector addition kernel for testing Metal compilation pipeline
// This validates that our build system can compile and link Metal shaders

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}

// Simple matrix multiply kernel (naive implementation for testing)
// Not optimized - just validates compilation
kernel void matrix_multiply_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}

// Simple ReLU activation for testing
kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = max(0.0f, input[index]);
}

// Test half-precision (FP16) support
kernel void vector_add_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}

// Test shared memory / threadgroup memory
kernel void reduction_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    // Simple parallel reduction for testing threadgroup memory
    uint global_index = bid * block_size + tid;
    shared[tid] = input[global_index];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0];
    }
}
