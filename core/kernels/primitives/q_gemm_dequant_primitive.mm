// Copyright © 2025 MLXR Development
// MLX Primitive-based quantized GEMM with dequantization

#include "q_gemm_dequant_primitive.h"

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
// Quantization Block Structures (CPU)
// ========================================

// Must match Metal shader definitions

struct BlockQ4_0 {
  uint16_t scale;  // FP16 as uint16
  uint8_t data[16];
};

struct BlockQ4_1 {
  uint16_t scale;
  uint16_t min;
  uint8_t data[16];
};

struct BlockQ8_0 {
  uint16_t scale;
  int8_t data[32];
};

struct BlockQ4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};

struct BlockQ6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};

// ========================================
// Helper Functions
// ========================================

float QGemmDequantPrimitive::get_bytes_per_weight(QuantType type) {
  switch (type) {
    case QuantType::Q4_0:
      return sizeof(BlockQ4_0) / 32.0f;  // 18 bytes / 32 weights = 0.5625 bytes/weight
    case QuantType::Q4_1:
      return sizeof(BlockQ4_1) / 32.0f;  // 20 bytes / 32 weights = 0.625 bytes/weight
    case QuantType::Q8_0:
      return sizeof(BlockQ8_0) / 32.0f;  // 34 bytes / 32 weights = 1.0625 bytes/weight
    case QuantType::Q4_K:
      return sizeof(BlockQ4_K) / 256.0f; // 144 bytes / 256 weights = 0.5625 bytes/weight
    case QuantType::Q6_K:
      return sizeof(BlockQ6_K) / 256.0f; // 210 bytes / 256 weights = 0.82 bytes/weight
    default:
      throw std::runtime_error("Unsupported quantization type");
  }
}

int QGemmDequantPrimitive::get_default_group_size(QuantType type) {
  switch (type) {
    case QuantType::Q4_0:
    case QuantType::Q4_1:
    case QuantType::Q8_0:
      return 32;
    case QuantType::Q4_K:
    case QuantType::Q6_K:
      return 256;  // Super-block size
    default:
      return 32;
  }
}

const char* quant_type_name(QuantType type) {
  switch (type) {
    case QuantType::Q4_0: return "Q4_0";
    case QuantType::Q4_1: return "Q4_1";
    case QuantType::Q8_0: return "Q8_0";
    case QuantType::Q4_K: return "Q4_K";
    case QuantType::Q6_K: return "Q6_K";
    case QuantType::Q5_0: return "Q5_0";
    case QuantType::Q5_1: return "Q5_1";
    case QuantType::Q8_1: return "Q8_1";
    case QuantType::Q2_K: return "Q2_K";
    case QuantType::Q3_K: return "Q3_K";
    case QuantType::Q5_K: return "Q5_K";
    case QuantType::Q8_K: return "Q8_K";
    case QuantType::IQ2_XXS: return "IQ2_XXS";
    case QuantType::IQ2_XS: return "IQ2_XS";
    case QuantType::IQ3_XXS: return "IQ3_XXS";
    case QuantType::IQ3_S: return "IQ3_S";
    default: return "Unknown";
  }
}

size_t compute_quantized_weight_size(int N, int K, QuantType quant_type) {
  const int group_size = QGemmDequantPrimitive::get_default_group_size(quant_type);
  const int num_groups = (K + group_size - 1) / group_size;
  const size_t groups_total = N * num_groups;

  switch (quant_type) {
    case QuantType::Q4_0:
      return groups_total * sizeof(BlockQ4_0);
    case QuantType::Q4_1:
      return groups_total * sizeof(BlockQ4_1);
    case QuantType::Q8_0:
      return groups_total * sizeof(BlockQ8_0);
    case QuantType::Q4_K:
      return groups_total * sizeof(BlockQ4_K);
    case QuantType::Q6_K:
      return groups_total * sizeof(BlockQ6_K);
    default:
      throw std::runtime_error("Unsupported quantization type");
  }
}

// ========================================
// FP16 Conversion Helpers
// ========================================

inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;

  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      return sign ? -0.0f : 0.0f;
    } else {
      // Denormalized
      float result = mantissa / 1024.0f;
      result *= powf(2.0f, -14.0f);
      return sign ? -result : result;
    }
  } else if (exponent == 31) {
    // Infinity or NaN
    return mantissa ? NAN : (sign ? -INFINITY : INFINITY);
  } else {
    // Normalized
    float result = 1.0f + mantissa / 1024.0f;
    result *= powf(2.0f, (int)exponent - 15);
    return sign ? -result : result;
  }
}

inline uint16_t fp32_to_fp16(float f) {
  uint32_t x = *reinterpret_cast<uint32_t*>(&f);
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t exponent = ((x >> 23) & 0xFF) - 112;
  uint32_t mantissa = x & 0x7FFFFF;

  if (exponent <= 0) {
    // Denormalized or zero
    return sign;
  } else if (exponent >= 31) {
    // Infinity
    return sign | 0x7C00;
  } else {
    // Normalized
    return sign | (exponent << 10) | (mantissa >> 13);
  }
}

// ========================================
// CPU Dequantization Functions
// ========================================

void dequant_q4_0_cpu(const BlockQ4_0* block, float* output, int offset) {
  float scale = fp16_to_fp32(block->scale);

  for (int i = 0; i < 16; i++) {
    uint8_t byte_val = block->data[i];

    int8_t v0 = (byte_val & 0x0F) - 8;
    output[offset + i * 2 + 0] = v0 * scale;

    int8_t v1 = ((byte_val >> 4) & 0x0F) - 8;
    output[offset + i * 2 + 1] = v1 * scale;
  }
}

void dequant_q4_1_cpu(const BlockQ4_1* block, float* output, int offset) {
  float scale = fp16_to_fp32(block->scale);
  float min_val = fp16_to_fp32(block->min);

  for (int i = 0; i < 16; i++) {
    uint8_t byte_val = block->data[i];

    uint8_t v0 = byte_val & 0x0F;
    output[offset + i * 2 + 0] = v0 * scale + min_val;

    uint8_t v1 = (byte_val >> 4) & 0x0F;
    output[offset + i * 2 + 1] = v1 * scale + min_val;
  }
}

void dequant_q8_0_cpu(const BlockQ8_0* block, float* output, int offset) {
  float scale = fp16_to_fp32(block->scale);

  for (int i = 0; i < 32; i++) {
    int8_t v = block->data[i];
    output[offset + i] = v * scale;
  }
}

// ========================================
// QGemmDequantPrimitive Implementation
// ========================================

QGemmDequantPrimitive::QGemmDequantPrimitive(
    mlx::core::Stream stream,
    int M,
    int N,
    int K,
    QuantType quant_type,
    int group_size,
    bool has_bias)
    : mlx::core::Primitive(stream),
      M_(M),
      N_(N),
      K_(K),
      quant_type_(quant_type),
      group_size_(group_size),
      has_bias_(has_bias),
      library_(nullptr) {}

QGemmDequantPrimitive::~QGemmDequantPrimitive() {
  if (library_ != nullptr) {
    auto* lib = static_cast<MTL::Library*>(library_);
    lib->release();
  }
}

void* QGemmDequantPrimitive::load_metal_library() {
  if (library_ != nullptr) {
    return library_;
  }

  // Get Metal device
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  auto mtl_device = d.mtl_device();

  // Load metallib from build directory
  NS::Error* error = nullptr;
  NS::String* lib_path = NS::String::string(
      "/Users/ryanoboyle/MLXR/build/lib/q_gemm_dequant.metallib",
      NS::UTF8StringEncoding);

  auto* lib = mtl_device->newLibrary(lib_path, &error);

  if (lib == nullptr) {
    std::ostringstream err;
    err << "Failed to load q_gemm_dequant Metal library: ";
    if (error) {
      err << error->localizedDescription()->utf8String();
    }
    throw std::runtime_error(err.str());
  }

  library_ = lib;
  return library_;
}

bool QGemmDequantPrimitive::is_equivalent(const mlx::core::Primitive& other) const {
  const auto* other_qgemm = dynamic_cast<const QGemmDequantPrimitive*>(&other);
  if (!other_qgemm) {
    return false;
  }

  return M_ == other_qgemm->M_ &&
         N_ == other_qgemm->N_ &&
         K_ == other_qgemm->K_ &&
         quant_type_ == other_qgemm->quant_type_ &&
         group_size_ == other_qgemm->group_size_ &&
         has_bias_ == other_qgemm->has_bias_;
}

void QGemmDequantPrimitive::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Validate inputs
  if (inputs.size() < 2 || (has_bias_ && inputs.size() < 3)) {
    throw std::runtime_error("QGemmDequant requires input, weights, and optional bias");
  }

  const auto& input = inputs[0];    // [M, K]
  const auto& weights = inputs[1];  // Quantized format
  const mlx::core::array* bias = has_bias_ ? &inputs[2] : nullptr;
  auto& output = outputs[0];        // [M, N]

  // Allocate output
  output = mlx::core::array({M_, N_}, mlx::core::float32);

  // Dequantize weights completely (CPU fallback)
  std::vector<float> w_dequant(N_ * K_);

  const int num_groups = (K_ + group_size_ - 1) / group_size_;

  for (int n = 0; n < N_; n++) {
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
      const int k_start = group_idx * group_size_;
      const int k_end = std::min(k_start + group_size_, K_);
      const int block_linear_idx = n * num_groups + group_idx;

      const int w_offset = n * K_ + k_start;

      // Dequantize based on type
      if (quant_type_ == QuantType::Q4_0) {
        const auto* blocks = static_cast<const BlockQ4_0*>(weights.data<void>());
        dequant_q4_0_cpu(&blocks[block_linear_idx], w_dequant.data(), w_offset);
      } else if (quant_type_ == QuantType::Q4_1) {
        const auto* blocks = static_cast<const BlockQ4_1*>(weights.data<void>());
        dequant_q4_1_cpu(&blocks[block_linear_idx], w_dequant.data(), w_offset);
      } else if (quant_type_ == QuantType::Q8_0) {
        const auto* blocks = static_cast<const BlockQ8_0*>(weights.data<void>());
        dequant_q8_0_cpu(&blocks[block_linear_idx], w_dequant.data(), w_offset);
      } else {
        throw std::runtime_error("Unsupported quantization format for CPU evaluation");
      }
    }
  }

  // Perform matrix multiplication: Y = X @ W^T
  const float* x_data = input.data<float>();
  float* y_data = output.data<float>();

  for (int m = 0; m < M_; m++) {
    for (int n = 0; n < N_; n++) {
      float accum = 0.0f;

      for (int k = 0; k < K_; k++) {
        accum += x_data[m * K_ + k] * w_dequant[n * K_ + k];
      }

      // Add bias if present
      if (bias != nullptr) {
        accum += bias->data<float>()[n];
      }

      y_data[m * N_ + n] = accum;
    }
  }
}

void QGemmDequantPrimitive::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {

  // Validate inputs
  if (inputs.size() < 2 || (has_bias_ && inputs.size() < 3)) {
    throw std::runtime_error("QGemmDequant requires input, weights, and optional bias");
  }

  const auto& input = inputs[0];
  const auto& weights = inputs[1];
  const mlx::core::array* bias = has_bias_ ? &inputs[2] : nullptr;
  auto& output = outputs[0];

  // Allocate output
  output = mlx::core::array({M_, N_}, mlx::core::float16);

  // Load Metal library
  auto* mtl_lib = static_cast<MTL::Library*>(load_metal_library());

  // Get device and kernel
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  auto kernel_name = "q_gemm_dequant";
  auto* kernel = d.get_kernel(kernel_name, mtl_lib);

  // Create command buffer and encoder
  auto& compute_encoder = d.get_command_encoder(stream().index);

  // Set kernel
  compute_encoder.set_compute_pipeline_state(kernel);

  // Bind buffers
  // Buffer 0: Arguments struct
  struct QGemmArgs {
    const void* input;
    void* output;
    const void* weights;
    const void* bias;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t quant_type;
    uint32_t group_size;
    uint32_t num_groups;
  };

  QGemmArgs args;
  args.input = input.data<void>();
  args.output = output.data<void>();
  args.weights = weights.data<void>();
  args.bias = (bias != nullptr) ? bias->data<void>() : nullptr;
  args.M = M_;
  args.N = N_;
  args.K = K_;
  args.quant_type = static_cast<uint32_t>(quant_type_);
  args.group_size = group_size_;
  args.num_groups = (K_ + group_size_ - 1) / group_size_;

  compute_encoder.set_bytes(&args, sizeof(QGemmArgs), 0);

  // Allocate threadgroup memory
  // shared_x: TILE_M * TILE_K * sizeof(half) = 32 * 32 * 2 = 2048 bytes
  // shared_w: TILE_N * TILE_K * sizeof(half) = 64 * 32 * 2 = 4096 bytes
  const uint32_t TILE_M = 32;
  const uint32_t TILE_K = 32;
  const uint32_t TILE_N = 64;

  const size_t shared_x_size = TILE_M * TILE_K * sizeof(uint16_t);
  const size_t shared_w_size = TILE_N * TILE_K * sizeof(uint16_t);

  compute_encoder.set_threadgroup_memory_length(shared_x_size, 0);
  compute_encoder.set_threadgroup_memory_length(shared_w_size, 1);

  // Dispatch
  // Grid: [(N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M]
  // Threadgroup size: [16, 32] = 512 threads
  const uint32_t grid_x = (N_ + TILE_N - 1) / TILE_N;
  const uint32_t grid_y = (M_ + TILE_M - 1) / TILE_M;

  MTL::Size grid_size(grid_x, grid_y, 1);
  MTL::Size threadgroup_size(16, 32, 1);  // 512 threads

  compute_encoder.dispatch_threadgroups(grid_size, threadgroup_size);
}

std::pair<std::vector<mlx::core::array>, std::vector<int>>
QGemmDequantPrimitive::vmap(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("vmap not yet implemented for QGemmDequant");
}

std::vector<mlx::core::array> QGemmDequantPrimitive::jvp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("jvp not yet implemented for QGemmDequant");
}

std::vector<mlx::core::array> QGemmDequantPrimitive::vjp(
    const std::vector<mlx::core::array>& primals,
    const std::vector<mlx::core::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mlx::core::array>& outputs) {
  throw std::runtime_error("vjp not yet implemented for QGemmDequant");
}

// ========================================
// Public API
// ========================================

mlx::core::array q_gemm_dequant(
    const mlx::core::array& input,
    const mlx::core::array& weights,
    int M,
    int N,
    int K,
    QuantType quant_type,
    int group_size,
    const mlx::core::array* bias,
    mlx::core::StreamOrDevice s) {

  // Validate input shapes
  if (input.shape().size() != 2 || input.shape(0) != M || input.shape(1) != K) {
    throw std::runtime_error("Input shape must be [M, K]");
  }

  if (bias != nullptr && (bias->shape().size() != 1 || bias->shape(0) != N)) {
    throw std::runtime_error("Bias shape must be [N]");
  }

  // Create primitive
  auto stream = mlx::core::to_stream(s);
  auto prim = std::make_shared<QGemmDequantPrimitive>(
      stream, M, N, K, quant_type, group_size, bias != nullptr);

  // Prepare inputs
  std::vector<mlx::core::array> inputs_vec = {input, weights};
  if (bias != nullptr) {
    inputs_vec.push_back(*bias);
  }

  // Create output array
  mlx::core::array output({M, N}, mlx::core::float16);

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
