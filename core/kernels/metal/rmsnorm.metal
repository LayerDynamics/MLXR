/**
 * @file rmsnorm.metal
 * @brief Fused RMSNorm (Root Mean Square Layer Normalization) kernel
 *
 * Fuses the following operations into a single kernel:
 * 1. x^2 (square)
 * 2. mean(x^2) (mean reduction)
 * 3. rsqrt(mean + eps) (root mean square)
 * 4. x * rms (normalize)
 * 5. normalized * weight (scale by learned weight)
 *
 * This reduces 5 separate kernel launches to 1, improving performance
 * and reducing memory bandwidth.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Fused RMSNorm kernel
 *
 * Computes: output = (x / rms(x)) * weight
 * where rms(x) = sqrt(mean(x^2) + eps)
 *
 * Uses a two-pass algorithm:
 * Pass 1: Compute sum of squares using threadgroup reduction
 * Pass 2: Normalize and scale by weight
 *
 * @param input Input tensor [batch * seq_len, hidden_size]
 * @param weight Learned weight parameter [hidden_size]
 * @param output Output tensor [batch * seq_len, hidden_size]
 * @param batch_seq_len Number of sequences (batch * seq_len)
 * @param hidden_size Hidden dimension size
 * @param eps Epsilon for numerical stability
 * @param gid Thread position in grid (sequence index)
 * @param lid Thread position in threadgroup
 * @param local_sum Threadgroup shared memory for reduction
 */
kernel void rmsnorm_fused(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_seq_len [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    threadgroup float* local_sum [[threadgroup(0)]]) {

  // Each threadgroup processes one sequence
  if (gid >= batch_seq_len) {
    return;
  }

  // Pointer to this sequence's input
  device const float* x = input + gid * hidden_size;
  device float* y = output + gid * hidden_size;

  // Pass 1: Compute sum of squares using parallel reduction
  float sum_sq = 0.0f;

  // Each thread processes multiple elements if hidden_size > threadgroup_size
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float val = x[i];
    sum_sq += val * val;
  }

  // Store partial sum in threadgroup memory
  local_sum[lid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel reduction in threadgroup memory
  // Use FP32 accumulation for numerical stability
  for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      local_sum[lid] += local_sum[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Thread 0 computes the final RMS value
  float rms;
  if (lid == 0) {
    float mean_sq = local_sum[0] / float(hidden_size);
    rms = rsqrt(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    local_sum[0] = rms;  // Store for all threads to read
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rms = local_sum[0];

  // Pass 2: Normalize and scale by weight
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float normalized = x[i] * rms;
    y[i] = normalized * weight[i];
  }
}

/**
 * @brief Fused RMSNorm kernel with FP16 input/output
 *
 * Same as rmsnorm_fused but uses half precision for input/output
 * while maintaining FP32 accumulation for stability.
 */
kernel void rmsnorm_fused_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch_seq_len [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    threadgroup float* local_sum [[threadgroup(0)]]) {

  if (gid >= batch_seq_len) {
    return;
  }

  device const half* x = input + gid * hidden_size;
  device half* y = output + gid * hidden_size;

  // Pass 1: Compute sum of squares with FP32 accumulation
  float sum_sq = 0.0f;
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float val = float(x[i]);  // Convert to FP32 for accuracy
    sum_sq += val * val;
  }

  local_sum[lid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel reduction
  for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      local_sum[lid] += local_sum[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float rms;
  if (lid == 0) {
    float mean_sq = local_sum[0] / float(hidden_size);
    rms = rsqrt(mean_sq + eps);
    local_sum[0] = rms;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rms = local_sum[0];

  // Pass 2: Normalize and scale (FP16 output)
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float normalized = float(x[i]) * rms;
    y[i] = half(normalized * float(weight[i]));
  }
}

/**
 * @brief Fused RMSNorm with residual add
 *
 * Computes: output = (x / rms(x)) * weight + residual
 *
 * Fuses residual connection into the normalization kernel.
 */
kernel void rmsnorm_fused_residual(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_seq_len [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    threadgroup float* local_sum [[threadgroup(0)]]) {

  if (gid >= batch_seq_len) {
    return;
  }

  device const float* x = input + gid * hidden_size;
  device const float* res = residual + gid * hidden_size;
  device float* y = output + gid * hidden_size;

  // Pass 1: Sum of squares
  float sum_sq = 0.0f;
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float val = x[i];
    sum_sq += val * val;
  }

  local_sum[lid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      local_sum[lid] += local_sum[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float rms;
  if (lid == 0) {
    float mean_sq = local_sum[0] / float(hidden_size);
    rms = rsqrt(mean_sq + eps);
    local_sum[0] = rms;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rms = local_sum[0];

  // Pass 2: Normalize, scale, and add residual
  for (uint i = lid; i < hidden_size; i += threadgroup_size) {
    float normalized = x[i] * rms;
    y[i] = normalized * weight[i] + res[i];
  }
}
