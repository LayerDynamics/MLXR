/**
 * @file rope_apply.metal
 * @brief Rotary Positional Embeddings (RoPE) kernel
 *
 * This kernel applies rotary positional embeddings to query/key tensors.
 * RoPE encodes positional information by rotating pairs of dimensions:
 *
 * For each pair (even, odd):
 *   x_out[even] = x[even] * cos(θ) - x[odd] * sin(θ)
 *   x_out[odd]  = x[odd] * cos(θ) + x[even] * sin(θ)
 *
 * where θ = position * base^(-2i/d) for dimension pair i
 *
 * Features:
 * - Multiple scaling modes: base, NTK-scaled, YaRN-scaled
 * - Precomputed cos/sin tables for efficiency
 * - Support for head_dim: 64, 80, 96, 112, 128, 160, 192, 256
 * - Handles both contiguous and strided tensors
 * - FP16/FP32 precision options
 *
 * RoPE Variants:
 * 1. Base RoPE:
 *    θ = position * base^(-2i/d)
 *    base typically 10000 or 500000 (for long context)
 *
 * 2. NTK-scaled RoPE (for extended context):
 *    base' = base * ((scale * ctx_len / orig_ctx_len) - (scale - 1))^(d/(d-2))
 *
 * 3. YaRN (Yet another RoPE Normalization):
 *    Applies temperature scaling and attention reweighting
 *    Different scaling for low/mid/high frequency components
 *
 * Layout:
 * - Input:     [batch, seq_len, num_heads, head_dim] or [tokens, num_heads, head_dim]
 * - Output:    Same shape as input
 * - cos_table: [max_seq_len, head_dim/2]
 * - sin_table: [max_seq_len, head_dim/2]
 * - positions: [batch, seq_len] or [tokens] (token position indices)
 */

#include <metal_stdlib>
using namespace metal;

/**
 * RoPE scaling modes
 */
enum class RoPEScalingMode : uint {
  BASE = 0,         // Standard RoPE
  NTK = 1,          // NTK-aware interpolation
  YARN = 2,         // YaRN scaling
  LINEAR = 3,       // Linear interpolation
};

/**
 * RoPE kernel arguments
 */
struct RoPEArgs {
  // Input and output
  device const half* input;      // Input tensor
  device half* output;           // Output tensor

  // Precomputed RoPE tables
  device const half* cos_table;  // cos values: [max_seq_len, head_dim/2]
  device const half* sin_table;  // sin values: [max_seq_len, head_dim/2]

  // Position indices
  device const int* positions;   // Token position indices

  // Dimensions
  uint batch_size;               // Number of sequences (or total tokens if flattened)
  uint seq_len;                  // Sequence length (1 if flattened)
  uint num_heads;                // Number of attention heads
  uint head_dim;                 // Dimension per head (must be even)

  // Scaling parameters
  uint scaling_mode;             // RoPEScalingMode
  float scale_factor;            // Linear scaling factor (if LINEAR mode)
  uint position_offset;          // Offset to add to positions (default 0)

  // Tensor strides (for non-contiguous tensors)
  uint input_batch_stride;
  uint input_seq_stride;
  uint input_head_stride;
  uint output_batch_stride;
  uint output_seq_stride;
  uint output_head_stride;
};

/**
 * Apply RoPE to a single head's data
 *
 * Each thread processes one dimension pair (even, odd)
 *
 * Grid layout:
 *   X: batch_size * seq_len * num_heads (one threadgroup per head)
 *   Y: 1
 *   Z: 1
 *
 * Threadgroup layout:
 *   Threads: head_dim / 2 (one thread per dimension pair)
 *   Max 256 threads, so head_dim <= 512
 */
kernel void rope_apply(
    constant RoPEArgs& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]) {

  // Decode linear index to (batch, seq, head)
  const uint total_heads = args.batch_size * args.seq_len * args.num_heads;
  if (gid >= total_heads) {
    return;
  }

  const uint batch_idx = gid / (args.seq_len * args.num_heads);
  const uint seq_idx = (gid / args.num_heads) % args.seq_len;
  const uint head_idx = gid % args.num_heads;

  // Get position for this token
  uint position;
  if (args.seq_len == 1) {
    // Flattened layout: positions[batch_idx] is actual position
    position = args.positions[batch_idx];
  } else {
    // 4D layout: positions[batch_idx * seq_len + seq_idx]
    position = args.positions[batch_idx * args.seq_len + seq_idx];
  }

  // Add position offset
  position += args.position_offset;

  // Calculate base pointers for input and output
  const uint input_offset = batch_idx * args.input_batch_stride +
                           seq_idx * args.input_seq_stride +
                           head_idx * args.input_head_stride;
  const uint output_offset = batch_idx * args.output_batch_stride +
                            seq_idx * args.output_seq_stride +
                            head_idx * args.output_head_stride;

  device const half* input_head = args.input + input_offset;
  device half* output_head = args.output + output_offset;

  // Each thread processes one dimension pair
  const uint num_pairs = args.head_dim / 2;
  for (uint pair_idx = lid; pair_idx < num_pairs; pair_idx += threadgroup_size) {
    const uint even_dim = pair_idx * 2;
    const uint odd_dim = pair_idx * 2 + 1;

    // Load input values
    float x_even = float(input_head[even_dim]);
    float x_odd = float(input_head[odd_dim]);

    // Load cos and sin from precomputed tables
    const uint rope_idx = position * num_pairs + pair_idx;
    float cos_val = float(args.cos_table[rope_idx]);
    float sin_val = float(args.sin_table[rope_idx]);

    // Apply rotation
    // Rotation matrix: [cos -sin]
    //                  [sin  cos]
    float y_even = x_even * cos_val - x_odd * sin_val;
    float y_odd = x_odd * cos_val + x_even * sin_val;

    // Apply scaling if needed
    if (args.scaling_mode == uint(RoPEScalingMode::LINEAR)) {
      y_even *= args.scale_factor;
      y_odd *= args.scale_factor;
    }

    // Store output
    output_head[even_dim] = half(y_even);
    output_head[odd_dim] = half(y_odd);
  }
}

/**
 * In-place RoPE application
 * Modifies input tensor directly instead of writing to separate output
 *
 * Grid and threadgroup layout same as rope_apply
 */
kernel void rope_apply_inplace(
    constant RoPEArgs& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]) {

  const uint total_heads = args.batch_size * args.seq_len * args.num_heads;
  if (gid >= total_heads) {
    return;
  }

  const uint batch_idx = gid / (args.seq_len * args.num_heads);
  const uint seq_idx = (gid / args.num_heads) % args.seq_len;
  const uint head_idx = gid % args.num_heads;

  // Get position
  uint position;
  if (args.seq_len == 1) {
    position = args.positions[batch_idx];
  } else {
    position = args.positions[batch_idx * args.seq_len + seq_idx];
  }
  position += args.position_offset;

  // Calculate base pointer for in-place modification
  const uint input_offset = batch_idx * args.input_batch_stride +
                           seq_idx * args.input_seq_stride +
                           head_idx * args.input_head_stride;

  device half* data_head = const_cast<device half*>(args.input + input_offset);

  // Process dimension pairs
  const uint num_pairs = args.head_dim / 2;
  for (uint pair_idx = lid; pair_idx < num_pairs; pair_idx += threadgroup_size) {
    const uint even_dim = pair_idx * 2;
    const uint odd_dim = pair_idx * 2 + 1;

    // Load values
    float x_even = float(data_head[even_dim]);
    float x_odd = float(data_head[odd_dim]);

    // Load cos/sin
    const uint rope_idx = position * num_pairs + pair_idx;
    float cos_val = float(args.cos_table[rope_idx]);
    float sin_val = float(args.sin_table[rope_idx]);

    // Apply rotation
    float y_even = x_even * cos_val - x_odd * sin_val;
    float y_odd = x_odd * cos_val + x_even * sin_val;

    // Apply scaling
    if (args.scaling_mode == uint(RoPEScalingMode::LINEAR)) {
      y_even *= args.scale_factor;
      y_odd *= args.scale_factor;
    }

    // Store back in-place
    data_head[even_dim] = half(y_even);
    data_head[odd_dim] = half(y_odd);
  }
}

/**
 * FP32 variant of RoPE for higher precision
 * Used when numerical stability is critical
 */
kernel void rope_apply_fp32(
    constant RoPEArgs& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]) {

  const uint total_heads = args.batch_size * args.seq_len * args.num_heads;
  if (gid >= total_heads) {
    return;
  }

  const uint batch_idx = gid / (args.seq_len * args.num_heads);
  const uint seq_idx = (gid / args.num_heads) % args.seq_len;
  const uint head_idx = gid % args.num_heads;

  // Get position
  uint position;
  if (args.seq_len == 1) {
    position = args.positions[batch_idx];
  } else {
    position = args.positions[batch_idx * args.seq_len + seq_idx];
  }
  position += args.position_offset;

  // Input is fp32 instead of fp16
  device const float* input_fp32 = reinterpret_cast<device const float*>(args.input);
  device float* output_fp32 = reinterpret_cast<device float*>(args.output);

  // Calculate offsets (fp32 stride is 2x fp16)
  const uint input_offset = batch_idx * args.input_batch_stride +
                           seq_idx * args.input_seq_stride +
                           head_idx * args.input_head_stride;
  const uint output_offset = batch_idx * args.output_batch_stride +
                            seq_idx * args.output_seq_stride +
                            head_idx * args.output_head_stride;

  device const float* input_head = input_fp32 + input_offset;
  device float* output_head = output_fp32 + output_offset;

  // Process pairs
  const uint num_pairs = args.head_dim / 2;
  for (uint pair_idx = lid; pair_idx < num_pairs; pair_idx += threadgroup_size) {
    const uint even_dim = pair_idx * 2;
    const uint odd_dim = pair_idx * 2 + 1;

    float x_even = input_head[even_dim];
    float x_odd = input_head[odd_dim];

    const uint rope_idx = position * num_pairs + pair_idx;
    float cos_val = float(args.cos_table[rope_idx]);
    float sin_val = float(args.sin_table[rope_idx]);

    float y_even = x_even * cos_val - x_odd * sin_val;
    float y_odd = x_odd * cos_val + x_even * sin_val;

    if (args.scaling_mode == uint(RoPEScalingMode::LINEAR)) {
      y_even *= args.scale_factor;
      y_odd *= args.scale_factor;
    }

    output_head[even_dim] = y_even;
    output_head[odd_dim] = y_odd;
  }
}

/**
 * Batched RoPE arguments for Q and K
 */
struct RoPEQKArgs {
  // Q arguments (embedded RoPEArgs)
  device const half* q_input;
  device half* q_output;
  device const half* q_cos_table;
  device const half* q_sin_table;
  device const int* positions;
  uint batch_size;
  uint seq_len;
  uint num_heads;
  uint head_dim;
  uint scaling_mode;
  float scale_factor;
  uint position_offset;
  uint input_batch_stride;
  uint input_seq_stride;
  uint input_head_stride;
  uint output_batch_stride;
  uint output_seq_stride;
  uint output_head_stride;

  // K arguments
  device const half* k_input;
  device half* k_output;
  device const half* k_cos_table;
  device const half* k_sin_table;
  uint num_kv_heads;  // For GQA
};

/**
 * Batched RoPE for Q and K simultaneously
 * More efficient when both need RoPE at the same time
 *
 * Processes Q and K tensors in a single kernel dispatch
 */
kernel void rope_apply_qk_batched(
    constant RoPEQKArgs& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]) {

  const uint total_heads = args.batch_size * args.seq_len * args.num_heads;
  if (gid >= total_heads) {
    return;
  }

  const uint batch_idx = gid / (args.seq_len * args.num_heads);
  const uint seq_idx = (gid / args.num_heads) % args.seq_len;
  const uint head_idx = gid % args.num_heads;

  // Get position
  uint position;
  if (args.seq_len == 1) {
    position = args.positions[batch_idx];
  } else {
    position = args.positions[batch_idx * args.seq_len + seq_idx];
  }
  position += args.position_offset;

  // === Process Q ===
  const uint q_input_offset = batch_idx * args.input_batch_stride +
                             seq_idx * args.input_seq_stride +
                             head_idx * args.input_head_stride;
  const uint q_output_offset = batch_idx * args.output_batch_stride +
                              seq_idx * args.output_seq_stride +
                              head_idx * args.output_head_stride;

  device const half* q_input_head = args.q_input + q_input_offset;
  device half* q_output_head = args.q_output + q_output_offset;

  const uint num_pairs = args.head_dim / 2;
  for (uint pair_idx = lid; pair_idx < num_pairs; pair_idx += threadgroup_size) {
    const uint even_dim = pair_idx * 2;
    const uint odd_dim = pair_idx * 2 + 1;

    float q_even = float(q_input_head[even_dim]);
    float q_odd = float(q_input_head[odd_dim]);

    const uint rope_idx = position * num_pairs + pair_idx;
    float cos_val = float(args.q_cos_table[rope_idx]);
    float sin_val = float(args.q_sin_table[rope_idx]);

    float q_out_even = q_even * cos_val - q_odd * sin_val;
    float q_out_odd = q_odd * cos_val + q_even * sin_val;

    q_output_head[even_dim] = half(q_out_even);
    q_output_head[odd_dim] = half(q_out_odd);
  }

  // === Process K (with GQA head mapping) ===
  const uint kv_head_idx = head_idx / (args.num_heads / args.num_kv_heads);
  const uint k_input_offset = batch_idx * args.input_batch_stride +
                             seq_idx * args.input_seq_stride +
                             kv_head_idx * args.input_head_stride;
  const uint k_output_offset = batch_idx * args.output_batch_stride +
                              seq_idx * args.output_seq_stride +
                              kv_head_idx * args.output_head_stride;

  device const half* k_input_head = args.k_input + k_input_offset;
  device half* k_output_head = args.k_output + k_output_offset;

  for (uint pair_idx = lid; pair_idx < num_pairs; pair_idx += threadgroup_size) {
    const uint even_dim = pair_idx * 2;
    const uint odd_dim = pair_idx * 2 + 1;

    float k_even = float(k_input_head[even_dim]);
    float k_odd = float(k_input_head[odd_dim]);

    const uint rope_idx = position * num_pairs + pair_idx;
    float k_cos_val = float(args.k_cos_table[rope_idx]);
    float k_sin_val = float(args.k_sin_table[rope_idx]);

    float k_out_even = k_even * k_cos_val - k_odd * k_sin_val;
    float k_out_odd = k_odd * k_cos_val + k_even * k_sin_val;

    k_output_head[even_dim] = half(k_out_even);
    k_output_head[odd_dim] = half(k_out_odd);
  }
}
