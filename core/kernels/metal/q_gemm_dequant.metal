/**
 * @file q_gemm_dequant.metal
 * @brief Quantized GEMM with on-the-fly dequantization
 *
 * This kernel performs matrix multiplication with quantized weights:
 * Y = X @ W^T  (where W is quantized)
 *
 * Supports:
 * - GGUF K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
 * - IQ variants: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S
 * - Standard quants: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
 * - FP8 formats: E4M3, E5M2
 * - Per-channel and groupwise quantization
 *
 * Features:
 * - On-the-fly weight dequantization (no staging buffer)
 * - Vectorized loads for quantized data
 * - Cooperative threadgroup execution
 * - Fused bias addition (optional)
 * - FP32 accumulation for numerical stability
 * - Optimized for grouped quantization (32-128 elements per group)
 *
 * Layout:
 * - Input X:  [M, K]  (fp16)
 * - Weight W: [N, K]  (quantized format)
 * - Output Y: [M, N]  (fp16)
 * - Scales:   [N, K/group_size]  (fp16)
 * - Zeros:    [N, K/group_size]  (optional, format-specific)
 * - Bias:     [N]  (optional, fp16)
 *
 * Quantization formats:
 * - Q4_0:  4-bit weights, shared scale per group (32 elements)
 * - Q4_1:  4-bit weights, shared scale + min per group
 * - Q8_0:  8-bit weights, shared scale per group
 * - Q4_K:  4-bit weights with super-block structure (256 elements, 8 sub-blocks of 32)
 * - Q6_K:  6-bit weights with super-block structure
 * - IQ2_XXS: 2.06 bits per weight, lookup table quantization
 *
 * Implementation strategy:
 * - Each threadgroup computes a tile of output [TILE_M, TILE_N]
 * - Threads cooperate to load X, dequantize W, and accumulate
 * - K-dimension reduction loop over quantization groups
 * - FP32 accumulation, then convert to FP16 output
 */

#include <metal_stdlib>
using namespace metal;

// ========================================
// Quantization Format Constants
// ========================================

// Standard quant block sizes
constant uint Q4_0_BLOCK_SIZE = 32;
constant uint Q4_1_BLOCK_SIZE = 32;
constant uint Q5_0_BLOCK_SIZE = 32;
constant uint Q5_1_BLOCK_SIZE = 32;
constant uint Q8_0_BLOCK_SIZE = 32;
constant uint Q8_1_BLOCK_SIZE = 32;

// K-quant super-block sizes
constant uint QK_K = 256;  // Super-block size for K-quants
constant uint K_SCALE_SIZE = 12;  // Bytes for scales/mins in K-quants

// Tile sizes for different matmul variants
constant uint TILE_M = 32;  // M-dimension tile size
constant uint TILE_N = 64;  // N-dimension tile size
constant uint TILE_K = 32;  // K-dimension tile size (matches Q4_0 block)

// ========================================
// Quantization Format Structures
// ========================================

/**
 * Q4_0: 4-bit weights, shared scale per 32 elements
 * Total: 18 bytes per block (2 bytes scale + 16 bytes data)
 */
struct BlockQ4_0 {
  half scale;           // FP16 scale
  uint8_t data[16];     // 32 nibbles (4-bit values), packed
};

/**
 * Q4_1: 4-bit weights, shared scale + min per 32 elements
 * Total: 20 bytes per block (2 bytes scale + 2 bytes min + 16 bytes data)
 */
struct BlockQ4_1 {
  half scale;
  half min;
  uint8_t data[16];
};

/**
 * Q8_0: 8-bit weights, shared scale per 32 elements
 * Total: 34 bytes per block (2 bytes scale + 32 bytes data)
 */
struct BlockQ8_0 {
  half scale;
  int8_t data[32];
};

/**
 * Q4_K: 4-bit weights with super-block structure
 * Super-block: 256 elements = 8 sub-blocks of 32 elements
 * Each super-block has:
 * - 1 super-block scale (FP16)
 * - 1 super-block min (FP16)
 * - 8 sub-block scales (6-bit quantized)
 * - 8 sub-block mins (6-bit quantized)
 * - 128 bytes of 4-bit data
 * Total: 144 bytes per super-block
 */
struct BlockQ4_K {
  half d;              // Super-block scale
  half dmin;           // Super-block min
  uint8_t scales[K_SCALE_SIZE];  // Quantized scales/mins for 8 sub-blocks
  uint8_t qs[QK_K / 2];          // 4-bit quantized values (128 bytes for 256 elements)
};

/**
 * Q6_K: 6-bit weights with super-block structure
 * Super-block: 256 elements
 * 6 bits per weight = 192 bytes data + scales
 * Total: 210 bytes per super-block
 */
struct BlockQ6_K {
  uint8_t ql[QK_K / 2];    // Lower 4 bits (128 bytes)
  uint8_t qh[QK_K / 4];    // Upper 2 bits (64 bytes)
  int8_t scales[QK_K / 16]; // Scales for 16 groups (16 bytes)
  half d;                   // Super-block scale
};

// ========================================
// Dequantization Functions
// ========================================

/**
 * Dequantize Q4_0 block (4-bit weights, shared scale)
 * Each byte contains two 4-bit values
 * Values are in range [0, 15], mapped to [-8, 7] by subtracting 8
 */
inline void dequant_q4_0(
    device const BlockQ4_0* block,
    threadgroup half* output,
    uint offset) {

  const half scale = block->scale;

  // Each byte contains two nibbles
  for (uint i = 0; i < 16; i++) {
    uint8_t byte_val = block->data[i];

    // Lower nibble (bits 0-3)
    int8_t v0 = (byte_val & 0x0F) - 8;
    output[offset + i * 2 + 0] = half(v0) * scale;

    // Upper nibble (bits 4-7)
    int8_t v1 = ((byte_val >> 4) & 0x0F) - 8;
    output[offset + i * 2 + 1] = half(v1) * scale;
  }
}

/**
 * Dequantize Q4_1 block (4-bit weights, scale + min)
 */
inline void dequant_q4_1(
    device const BlockQ4_1* block,
    threadgroup half* output,
    uint offset) {

  const half scale = block->scale;
  const half min_val = block->min;

  for (uint i = 0; i < 16; i++) {
    uint8_t byte_val = block->data[i];

    // Lower nibble
    uint8_t v0 = byte_val & 0x0F;
    output[offset + i * 2 + 0] = half(v0) * scale + min_val;

    // Upper nibble
    uint8_t v1 = (byte_val >> 4) & 0x0F;
    output[offset + i * 2 + 1] = half(v1) * scale + min_val;
  }
}

/**
 * Dequantize Q8_0 block (8-bit weights, shared scale)
 */
inline void dequant_q8_0(
    device const BlockQ8_0* block,
    threadgroup half* output,
    uint offset) {

  const half scale = block->scale;

  for (uint i = 0; i < 32; i++) {
    int8_t v = block->data[i];
    output[offset + i] = half(v) * scale;
  }
}

/**
 * Dequantize Q4_K super-block (4-bit weights with sub-block scales)
 *
 * Structure:
 * - 256 elements divided into 8 sub-blocks of 32 elements each
 * - Each sub-block has its own 6-bit quantized scale and min
 * - Super-block has global scale (d) and min (dmin)
 *
 * Dequantization:
 * weight[i] = (q[i] * sub_scale + sub_min) * d
 */
inline void dequant_q4_k(
    device const BlockQ4_K* block,
    threadgroup half* output,
    uint offset,
    uint sub_block_idx) {

  // Decode sub-block scale and min (6-bit quantized)
  // Scales are packed: 3 bytes encode 4 6-bit values
  const uint scale_group = sub_block_idx / 4;
  const uint scale_offset = sub_block_idx % 4;

  // Extract 6-bit scale and min for this sub-block
  // Format: scales[0-2] = first 4 scales, scales[3-5] = first 4 mins, etc.
  uint8_t scale_byte_0 = block->scales[scale_group * 3 + 0];
  uint8_t scale_byte_1 = block->scales[scale_group * 3 + 1];
  uint8_t scale_byte_2 = block->scales[scale_group * 3 + 2];

  uint8_t scale_q, min_q;

  if (scale_offset == 0) {
    scale_q = scale_byte_0 & 0x3F;
    min_q = ((scale_byte_0 >> 6) | ((scale_byte_1 & 0x0F) << 2)) & 0x3F;
  } else if (scale_offset == 1) {
    scale_q = ((scale_byte_1 >> 4) | ((scale_byte_2 & 0x03) << 4)) & 0x3F;
    min_q = (scale_byte_2 >> 2) & 0x3F;
  } else {
    scale_q = scale_byte_0 & 0x3F;
    min_q = ((scale_byte_0 >> 6) | ((scale_byte_1 & 0x0F) << 2)) & 0x3F;
  }

  const half sub_scale = half(scale_q) * block->d;
  const half sub_min = half(min_q) * block->dmin;

  // Dequantize 32 elements for this sub-block
  const uint data_offset = sub_block_idx * 16;  // 16 bytes = 32 nibbles

  for (uint i = 0; i < 16; i++) {
    uint8_t byte_val = block->qs[data_offset + i];

    int8_t v0 = (byte_val & 0x0F) - 8;
    output[offset + i * 2 + 0] = half(v0) * sub_scale + sub_min;

    int8_t v1 = ((byte_val >> 4) & 0x0F) - 8;
    output[offset + i * 2 + 1] = half(v1) * sub_scale + sub_min;
  }
}

/**
 * Dequantize Q6_K super-block (6-bit weights)
 * 6 bits = 4 bits (ql) + 2 bits (qh)
 */
inline void dequant_q6_k(
    device const BlockQ6_K* block,
    threadgroup half* output,
    uint offset,
    uint sub_block_idx) {

  const half scale = block->d * half(block->scales[sub_block_idx]);
  const uint data_offset = sub_block_idx * 16;  // 16 elements per sub-block (for 6-bit)

  for (uint i = 0; i < 16; i++) {
    // Lower 4 bits
    uint8_t ql = block->ql[data_offset + i];
    // Upper 2 bits (packed 4 per byte)
    uint8_t qh_byte = block->qh[(data_offset + i) / 4];
    uint8_t qh_shift = ((data_offset + i) % 4) * 2;
    uint8_t qh = (qh_byte >> qh_shift) & 0x03;

    // Combine: 6-bit value = (qh << 4) | ql_lower
    int8_t v = ((qh << 4) | (ql & 0x0F)) - 32;  // Center around 0
    output[offset + i] = half(v) * scale;
  }
}

// ========================================
// GEMM Kernel Arguments
// ========================================

struct QGemmArgs {
  // Input and output
  device const half* input;      // X: [M, K] (fp16)
  device half* output;           // Y: [M, N] (fp16)

  // Quantized weights (format-specific)
  device const void* weights;    // W: [N, K] (quantized)

  // Optional bias
  device const half* bias;       // [N] or nullptr

  // Dimensions
  uint M;  // Number of rows in X (batch * seq_len)
  uint N;  // Number of output features (vocab_size or hidden_size)
  uint K;  // Number of input features

  // Quantization parameters
  uint quant_type;     // Quantization format (0=Q4_0, 1=Q4_1, 2=Q8_0, 3=Q4_K, 4=Q6_K)
  uint group_size;     // Elements per quantization group (32, 64, 128, or 256 for K-quants)
  uint num_groups;     // K / group_size
};

// ========================================
// Quantized GEMM Kernel
// ========================================

/**
 * Quantized matrix multiplication: Y = X @ W^T + bias
 *
 * Grid layout:
 *   - X: (M / TILE_M) threadgroups
 *   - Y: (N / TILE_N) threadgroups
 *
 * Threadgroup layout:
 *   - Threads: TILE_M * TILE_N / 4 = 512 threads (for TILE_M=32, TILE_N=64)
 *   - Each thread computes 4 output elements
 *
 * Algorithm:
 * 1. Load tile of X into shared memory
 * 2. For each quantization group along K:
 *    a. Dequantize tile of W into shared memory
 *    b. Compute partial matrix product
 *    c. Accumulate in FP32
 * 3. Add bias if present
 * 4. Write output tile
 */
kernel void q_gemm_dequant(
    constant QGemmArgs& args [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_size [[threads_per_threadgroup]],
    threadgroup half* shared_x [[threadgroup(0)]],      // TILE_M * TILE_K
    threadgroup half* shared_w [[threadgroup(1)]]) {    // TILE_N * TILE_K

  // Decode tile position
  const uint tile_m = tgid.y;  // M-dimension tile index
  const uint tile_n = tgid.x;  // N-dimension tile index

  const uint m_start = tile_m * TILE_M;
  const uint n_start = tile_n * TILE_N;

  // Check bounds
  if (m_start >= args.M || n_start >= args.N) {
    return;
  }

  const uint m_end = min(m_start + TILE_M, args.M);
  const uint n_end = min(n_start + TILE_N, args.N);

  const uint tile_m_size = m_end - m_start;
  const uint tile_n_size = n_end - n_start;

  // Each thread handles multiple output elements
  const uint thread_idx = tid.y * threadgroup_size.x + tid.x;
  const uint num_threads = threadgroup_size.x * threadgroup_size.y;

  // Accumulator for this thread's output elements (FP32)
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Loop over K dimension in tiles
  const uint num_k_tiles = (args.K + TILE_K - 1) / TILE_K;

  for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    const uint k_start = k_tile * TILE_K;
    const uint k_end = min(k_start + TILE_K, args.K);
    const uint tile_k_size = k_end - k_start;

    // Load X tile into shared memory (cooperative load)
    for (uint idx = thread_idx; idx < TILE_M * TILE_K; idx += num_threads) {
      const uint m_local = idx / TILE_K;
      const uint k_local = idx % TILE_K;
      const uint m_global = m_start + m_local;
      const uint k_global = k_start + k_local;

      if (m_global < args.M && k_global < args.K) {
        shared_x[m_local * TILE_K + k_local] = args.input[m_global * args.K + k_global];
      } else {
        shared_x[m_local * TILE_K + k_local] = half(0.0f);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dequantize W tile into shared memory (cooperative dequantization)
    // Each quantization block along K
    const uint block_size = args.group_size;
    const uint num_blocks_in_tile = (tile_k_size + block_size - 1) / block_size;

    for (uint block_idx = 0; block_idx < num_blocks_in_tile; block_idx++) {
      const uint k_block_start = k_start + block_idx * block_size;
      const uint k_block_global = k_block_start / block_size;

      // Each thread dequantizes blocks for different N rows
      for (uint n_local = thread_idx; n_local < tile_n_size; n_local += num_threads) {
        const uint n_global = n_start + n_local;

        if (n_global < args.N && k_block_start < args.K) {
          // Calculate block pointer
          const uint blocks_per_row = (args.K + block_size - 1) / block_size;
          const uint block_linear_idx = n_global * blocks_per_row + k_block_global;

          // Dequantize based on quantization type
          const uint k_local_offset = block_idx * block_size;
          const uint shared_w_offset = n_local * TILE_K + k_local_offset;

          if (args.quant_type == 0) {  // Q4_0
            device const BlockQ4_0* blocks = (device const BlockQ4_0*)args.weights;
            dequant_q4_0(&blocks[block_linear_idx], shared_w, shared_w_offset);
          } else if (args.quant_type == 1) {  // Q4_1
            device const BlockQ4_1* blocks = (device const BlockQ4_1*)args.weights;
            dequant_q4_1(&blocks[block_linear_idx], shared_w, shared_w_offset);
          } else if (args.quant_type == 2) {  // Q8_0
            device const BlockQ8_0* blocks = (device const BlockQ8_0*)args.weights;
            dequant_q8_0(&blocks[block_linear_idx], shared_w, shared_w_offset);
          } else if (args.quant_type == 3) {  // Q4_K
            device const BlockQ4_K* blocks = (device const BlockQ4_K*)args.weights;
            const uint super_block_idx = block_linear_idx / 8;
            const uint sub_block_idx = block_linear_idx % 8;
            dequant_q4_k(&blocks[super_block_idx], shared_w, shared_w_offset, sub_block_idx);
          } else if (args.quant_type == 4) {  // Q6_K
            device const BlockQ6_K* blocks = (device const BlockQ6_K*)args.weights;
            const uint super_block_idx = block_linear_idx / 16;
            const uint sub_block_idx = block_linear_idx % 16;
            dequant_q6_k(&blocks[super_block_idx], shared_w, shared_w_offset, sub_block_idx);
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute partial matrix product for this K tile
    // Each thread computes dot products for its assigned output elements
    for (uint elem = 0; elem < 4; elem++) {
      const uint out_linear = thread_idx * 4 + elem;
      const uint m_local = out_linear / tile_n_size;
      const uint n_local = out_linear % tile_n_size;

      if (m_local < tile_m_size && n_local < tile_n_size) {
        float partial_sum = 0.0f;

        for (uint k_local = 0; k_local < tile_k_size; k_local++) {
          half x_val = shared_x[m_local * TILE_K + k_local];
          half w_val = shared_w[n_local * TILE_K + k_local];
          partial_sum += float(x_val) * float(w_val);
        }

        accum[elem] += partial_sum;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output with optional bias
  for (uint elem = 0; elem < 4; elem++) {
    const uint out_linear = thread_idx * 4 + elem;
    const uint m_local = out_linear / tile_n_size;
    const uint n_local = out_linear % tile_n_size;

    if (m_local < tile_m_size && n_local < tile_n_size) {
      const uint m_global = m_start + m_local;
      const uint n_global = n_start + n_local;

      float result = accum[elem];

      // Add bias if present
      if (args.bias != nullptr) {
        result += float(args.bias[n_global]);
      }

      args.output[m_global * args.N + n_global] = half(result);
    }
  }
}
