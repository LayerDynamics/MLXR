// Copyright © 2025 MLXR Development
// GGUF file parser implementation

#include "gguf_parser.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace mlxr {
namespace registry {

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

const char* gguf_type_name(GGUFTensorType type) {
  switch (type) {
    case GGUFTensorType::F32:
      return "F32";
    case GGUFTensorType::F16:
      return "F16";
    case GGUFTensorType::Q4_0:
      return "Q4_0";
    case GGUFTensorType::Q4_1:
      return "Q4_1";
    case GGUFTensorType::Q5_0:
      return "Q5_0";
    case GGUFTensorType::Q5_1:
      return "Q5_1";
    case GGUFTensorType::Q8_0:
      return "Q8_0";
    case GGUFTensorType::Q8_1:
      return "Q8_1";
    case GGUFTensorType::Q2_K:
      return "Q2_K";
    case GGUFTensorType::Q3_K:
      return "Q3_K";
    case GGUFTensorType::Q4_K:
      return "Q4_K";
    case GGUFTensorType::Q5_K:
      return "Q5_K";
    case GGUFTensorType::Q6_K:
      return "Q6_K";
    case GGUFTensorType::Q8_K:
      return "Q8_K";
    case GGUFTensorType::IQ2_XXS:
      return "IQ2_XXS";
    case GGUFTensorType::IQ2_XS:
      return "IQ2_XS";
    case GGUFTensorType::IQ3_XXS:
      return "IQ3_XXS";
    case GGUFTensorType::IQ1_S:
      return "IQ1_S";
    case GGUFTensorType::IQ4_NL:
      return "IQ4_NL";
    case GGUFTensorType::IQ3_S:
      return "IQ3_S";
    case GGUFTensorType::IQ2_S:
      return "IQ2_S";
    case GGUFTensorType::IQ4_XS:
      return "IQ4_XS";
    case GGUFTensorType::I8:
      return "I8";
    case GGUFTensorType::I16:
      return "I16";
    case GGUFTensorType::I32:
      return "I32";
    case GGUFTensorType::I64:
      return "I64";
    case GGUFTensorType::F64:
      return "F64";
    case GGUFTensorType::IQ1_M:
      return "IQ1_M";
    default:
      return "Unknown";
  }
}

size_t gguf_type_size(GGUFTensorType type) {
  switch (type) {
    case GGUFTensorType::F32:
      return 4;
    case GGUFTensorType::F16:
      return 2;
    case GGUFTensorType::F64:
      return 8;
    case GGUFTensorType::I8:
      return 1;
    case GGUFTensorType::I16:
      return 2;
    case GGUFTensorType::I32:
      return 4;
    case GGUFTensorType::I64:
      return 8;
    // Quantized types return size per block
    default:
      return 0;  // Handled by gguf_block_size
  }
}

size_t gguf_block_size(GGUFTensorType type) {
  switch (type) {
    case GGUFTensorType::Q4_0:
      return 32;
    case GGUFTensorType::Q4_1:
      return 32;
    case GGUFTensorType::Q5_0:
      return 32;
    case GGUFTensorType::Q5_1:
      return 32;
    case GGUFTensorType::Q8_0:
      return 32;
    case GGUFTensorType::Q8_1:
      return 32;
    case GGUFTensorType::Q2_K:
      return 256;
    case GGUFTensorType::Q3_K:
      return 256;
    case GGUFTensorType::Q4_K:
      return 256;
    case GGUFTensorType::Q5_K:
      return 256;
    case GGUFTensorType::Q6_K:
      return 256;
    case GGUFTensorType::Q8_K:
      return 256;
    default:
      return 1;  // Non-quantized
  }
}

float gguf_bytes_per_weight(GGUFTensorType type) {
  switch (type) {
    case GGUFTensorType::F32:
      return 4.0f;
    case GGUFTensorType::F16:
      return 2.0f;
    case GGUFTensorType::F64:
      return 8.0f;
    case GGUFTensorType::Q4_0:
      return 0.5f + 2.0f / 32.0f;  // 4 bits + scale
    case GGUFTensorType::Q4_1:
      return 0.5f + 4.0f / 32.0f;  // 4 bits + scale + min
    case GGUFTensorType::Q5_0:
      return 0.625f + 2.0f / 32.0f;  // 5 bits + scale
    case GGUFTensorType::Q5_1:
      return 0.625f + 4.0f / 32.0f;  // 5 bits + scale + min
    case GGUFTensorType::Q8_0:
      return 1.0f + 2.0f / 32.0f;  // 8 bits + scale
    case GGUFTensorType::Q8_1:
      return 1.0f + 4.0f / 32.0f;  // 8 bits + scale + min
    case GGUFTensorType::Q2_K:
      return 0.25f + 12.0f / 256.0f;  // ~2.3 bits/weight
    case GGUFTensorType::Q3_K:
      return 0.375f + 12.0f / 256.0f;  // ~3.4 bits/weight
    case GGUFTensorType::Q4_K:
      return 0.5f + 12.0f / 256.0f;  // ~4.5 bits/weight
    case GGUFTensorType::Q5_K:
      return 0.625f + 12.0f / 256.0f;  // ~5.5 bits/weight
    case GGUFTensorType::Q6_K:
      return 0.75f + 12.0f / 256.0f;  // ~6.5 bits/weight
    case GGUFTensorType::Q8_K:
      return 1.0f + 12.0f / 256.0f;  // ~8.0 bits/weight
    case GGUFTensorType::I8:
      return 1.0f;
    case GGUFTensorType::I16:
      return 2.0f;
    case GGUFTensorType::I32:
      return 4.0f;
    case GGUFTensorType::I64:
      return 8.0f;
    default:
      return 4.0f;  // Default to F32
  }
}

uint64_t calculate_tensor_size(const GGUFTensorInfo& tensor) {
  // Calculate total number of elements
  uint64_t n_elements = 1;
  for (auto dim : tensor.dimensions) {
    n_elements *= dim;
  }

  // For quantized types, compute based on block structure
  size_t block_sz = gguf_block_size(tensor.type);
  if (block_sz > 1) {
    uint64_t n_blocks = (n_elements + block_sz - 1) / block_sz;

    // Bytes per block for each quantization type
    size_t bytes_per_block = 0;
    switch (tensor.type) {
      case GGUFTensorType::Q4_0:
        bytes_per_block = 18;
        break;  // 2 + 16
      case GGUFTensorType::Q4_1:
        bytes_per_block = 20;
        break;  // 2 + 2 + 16
      case GGUFTensorType::Q5_0:
        bytes_per_block = 22;
        break;  // 2 + 4 + 16
      case GGUFTensorType::Q5_1:
        bytes_per_block = 24;
        break;  // 2 + 2 + 4 + 16
      case GGUFTensorType::Q8_0:
        bytes_per_block = 34;
        break;  // 2 + 32
      case GGUFTensorType::Q8_1:
        bytes_per_block = 36;
        break;  // 2 + 2 + 32
      case GGUFTensorType::Q2_K:
        bytes_per_block = 80;
        break;  // Super-block for K-quants
      case GGUFTensorType::Q3_K:
        bytes_per_block = 108;
        break;
      case GGUFTensorType::Q4_K:
        bytes_per_block = 144;
        break;
      case GGUFTensorType::Q5_K:
        bytes_per_block = 176;
        break;
      case GGUFTensorType::Q6_K:
        bytes_per_block = 208;
        break;
      case GGUFTensorType::Q8_K:
        bytes_per_block = 292;
        break;
      default:
        bytes_per_block = block_sz * 4;
        break;  // Fallback
    }

    return n_blocks * bytes_per_block;
  }

  // For non-quantized types
  size_t elem_size = gguf_type_size(tensor.type);
  return n_elements * elem_size;
}

std::string gguf_type_to_mlx_dtype(GGUFTensorType type) {
  switch (type) {
    case GGUFTensorType::F32:
      return "float32";
    case GGUFTensorType::F16:
      return "float16";
    case GGUFTensorType::F64:
      return "float64";
    case GGUFTensorType::I8:
      return "int8";
    case GGUFTensorType::I16:
      return "int16";
    case GGUFTensorType::I32:
      return "int32";
    case GGUFTensorType::I64:
      return "int64";
    // Quantized types need dequantization
    default:
      return "float16";  // Will be dequantized to FP16
  }
}

//------------------------------------------------------------------------------
// GGUFFile Implementation
//------------------------------------------------------------------------------

GGUFFile::GGUFFile() : data_offset_(0) {
  header_.magic = 0;
  header_.version = 0;
  header_.tensor_count = 0;
  header_.metadata_kv_count = 0;
}

GGUFFile::~GGUFFile() = default;

bool GGUFFile::parse(const std::string& file_path) {
  std::ifstream stream(file_path, std::ios::binary);
  if (!stream.is_open()) {
    error_ = "Failed to open file: " + file_path;
    return false;
  }

  return parse_stream(stream);
}

bool GGUFFile::parse_stream(std::ifstream& stream) {
  // Reset state
  metadata_.clear();
  tensors_.clear();
  data_offset_ = 0;
  error_.clear();

  // Read header
  if (!read_header(stream)) {
    return false;
  }

  // Read metadata key-value pairs
  if (!read_metadata(stream)) {
    return false;
  }

  // Read tensor infos
  if (!read_tensor_infos(stream)) {
    return false;
  }

  // Calculate data offset (aligned to 32 bytes)
  data_offset_ = stream.tellg();
  uint64_t alignment = 32;
  data_offset_ = ((data_offset_ + alignment - 1) / alignment) * alignment;

  return true;
}

bool GGUFFile::read_header(std::ifstream& stream) {
  stream.read(reinterpret_cast<char*>(&header_.magic), sizeof(header_.magic));
  if (header_.magic != GGUF_MAGIC) {
    error_ = "Invalid GGUF magic number";
    return false;
  }

  stream.read(reinterpret_cast<char*>(&header_.version),
              sizeof(header_.version));
  if (header_.version < GGUF_VERSION_V1 || header_.version > GGUF_VERSION_V3) {
    error_ = "Unsupported GGUF version: " + std::to_string(header_.version);
    return false;
  }

  stream.read(reinterpret_cast<char*>(&header_.tensor_count),
              sizeof(header_.tensor_count));
  stream.read(reinterpret_cast<char*>(&header_.metadata_kv_count),
              sizeof(header_.metadata_kv_count));

  return stream.good();
}

bool GGUFFile::read_string(std::ifstream& stream, std::string& out) {
  uint64_t length;
  stream.read(reinterpret_cast<char*>(&length), sizeof(length));

  if (!stream.good() || length > 10485760) {  // Sanity check: 10MB max
    error_ = "Invalid string length";
    return false;
  }

  out.resize(length);
  stream.read(&out[0], length);

  return stream.good();
}

bool GGUFFile::read_metadata_value(std::ifstream& stream, GGUFMetadataType type,
                                   GGUFMetadataValue& out) {
  out.type = type;

  switch (type) {
    case GGUFMetadataType::UINT8:
      stream.read(reinterpret_cast<char*>(&out.u8), sizeof(out.u8));
      break;
    case GGUFMetadataType::INT8:
      stream.read(reinterpret_cast<char*>(&out.i8), sizeof(out.i8));
      break;
    case GGUFMetadataType::UINT16:
      stream.read(reinterpret_cast<char*>(&out.u16), sizeof(out.u16));
      break;
    case GGUFMetadataType::INT16:
      stream.read(reinterpret_cast<char*>(&out.i16), sizeof(out.i16));
      break;
    case GGUFMetadataType::UINT32:
      stream.read(reinterpret_cast<char*>(&out.u32), sizeof(out.u32));
      break;
    case GGUFMetadataType::INT32:
      stream.read(reinterpret_cast<char*>(&out.i32), sizeof(out.i32));
      break;
    case GGUFMetadataType::UINT64:
      stream.read(reinterpret_cast<char*>(&out.u64), sizeof(out.u64));
      break;
    case GGUFMetadataType::INT64:
      stream.read(reinterpret_cast<char*>(&out.i64), sizeof(out.i64));
      break;
    case GGUFMetadataType::FLOAT32:
      stream.read(reinterpret_cast<char*>(&out.f32), sizeof(out.f32));
      break;
    case GGUFMetadataType::FLOAT64:
      stream.read(reinterpret_cast<char*>(&out.f64), sizeof(out.f64));
      break;
    case GGUFMetadataType::BOOL:
      stream.read(reinterpret_cast<char*>(&out.b), sizeof(out.b));
      break;
    case GGUFMetadataType::STRING:
      if (!read_string(stream, out.str)) {
        return false;
      }
      break;
    case GGUFMetadataType::ARRAY: {
      out.arr = std::make_shared<GGUFArray>();
      uint32_t array_type;
      stream.read(reinterpret_cast<char*>(&array_type), sizeof(array_type));
      out.arr->type = static_cast<GGUFMetadataType>(array_type);
      stream.read(reinterpret_cast<char*>(&out.arr->length),
                  sizeof(out.arr->length));

      out.arr->values.resize(out.arr->length);
      for (uint64_t i = 0; i < out.arr->length; i++) {
        out.arr->values[i] = std::make_shared<GGUFMetadataValue>();
        if (!read_metadata_value(stream, out.arr->type, *out.arr->values[i])) {
          return false;
        }
      }
      break;
    }
    default:
      error_ = "Unknown metadata type: " +
               std::to_string(static_cast<uint32_t>(type));
      return false;
  }

  return stream.good();
}

bool GGUFFile::read_metadata(std::ifstream& stream) {
  for (uint64_t i = 0; i < header_.metadata_kv_count; i++) {
    std::string key;
    if (!read_string(stream, key)) {
      error_ = "Failed to read metadata key";
      return false;
    }

    uint32_t value_type;
    stream.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));

    GGUFMetadataValue value;
    if (!read_metadata_value(stream, static_cast<GGUFMetadataType>(value_type),
                             value)) {
      error_ = "Failed to read metadata value for key: " + key;
      return false;
    }

    metadata_[key] = value;
  }

  return true;
}

bool GGUFFile::read_tensor_infos(std::ifstream& stream) {
  tensors_.reserve(header_.tensor_count);

  for (uint64_t i = 0; i < header_.tensor_count; i++) {
    GGUFTensorInfo tensor;

    // Read tensor name
    if (!read_string(stream, tensor.name)) {
      error_ = "Failed to read tensor name";
      return false;
    }

    // Read number of dimensions
    stream.read(reinterpret_cast<char*>(&tensor.n_dimensions),
                sizeof(tensor.n_dimensions));

    // Read dimensions
    tensor.dimensions.resize(tensor.n_dimensions);
    for (uint32_t d = 0; d < tensor.n_dimensions; d++) {
      stream.read(reinterpret_cast<char*>(&tensor.dimensions[d]),
                  sizeof(uint64_t));
    }

    // Read tensor type
    uint32_t type_val;
    stream.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
    tensor.type = static_cast<GGUFTensorType>(type_val);

    // Read tensor offset
    stream.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));

    // Calculate tensor size
    tensor.size = calculate_tensor_size(tensor);

    tensors_.push_back(tensor);

    if (!stream.good()) {
      error_ = "Failed to read tensor info";
      return false;
    }
  }

  return true;
}

bool GGUFFile::has_metadata(const std::string& key) const {
  return metadata_.find(key) != metadata_.end();
}

const GGUFMetadataValue* GGUFFile::get_metadata(const std::string& key) const {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::string GGUFFile::get_arch() const {
  auto* val = get_metadata("general.architecture");
  return val ? val->as_string() : "";
}

uint32_t GGUFFile::get_context_length() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".context_length");
  return val ? val->as_uint32() : 2048;
}

uint32_t GGUFFile::get_embedding_length() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".embedding_length");
  return val ? val->as_uint32() : 0;
}

uint32_t GGUFFile::get_block_count() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".block_count");
  return val ? val->as_uint32() : 0;
}

uint32_t GGUFFile::get_feed_forward_length() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".feed_forward_length");
  return val ? val->as_uint32() : 0;
}

uint32_t GGUFFile::get_attention_head_count() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".attention.head_count");
  return val ? val->as_uint32() : 0;
}

uint32_t GGUFFile::get_attention_head_count_kv() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".attention.head_count_kv");
  return val ? val->as_uint32() : get_attention_head_count();  // Default to MHA
}

float GGUFFile::get_rope_freq_base() const {
  std::string arch = get_arch();
  auto* val = get_metadata(arch + ".rope.freq_base");
  return val ? val->as_float32() : 10000.0f;
}

std::string GGUFFile::get_tokenizer_model() const {
  auto* val = get_metadata("tokenizer.ggml.model");
  return val ? val->as_string() : "";
}

const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
  for (const auto& tensor : tensors_) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

std::vector<std::string> GGUFFile::get_tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& tensor : tensors_) {
    names.push_back(tensor.name);
  }
  return names;
}

void GGUFFile::print_info() const {
  std::cout << "\n=== GGUF File Info ===\n";
  std::cout << "Version: " << header_.version << "\n";
  std::cout << "Tensor count: " << header_.tensor_count << "\n";
  std::cout << "Metadata KV count: " << header_.metadata_kv_count << "\n";
  std::cout << "Data offset: " << data_offset_ << " bytes\n";
  std::cout << "\nArchitecture: " << get_arch() << "\n";
  std::cout << "Context length: " << get_context_length() << "\n";
  std::cout << "Embedding length: " << get_embedding_length() << "\n";
  std::cout << "Block count: " << get_block_count() << "\n";
  std::cout << "Feed-forward length: " << get_feed_forward_length() << "\n";
  std::cout << "Attention heads: " << get_attention_head_count() << "\n";
  std::cout << "KV heads: " << get_attention_head_count_kv() << "\n";
  std::cout << "RoPE freq base: " << get_rope_freq_base() << "\n";
  std::cout << "Tokenizer: " << get_tokenizer_model() << "\n";
}

void GGUFFile::print_metadata() const {
  std::cout << "\n=== Metadata (" << metadata_.size() << " entries) ===\n";
  for (const auto& [key, value] : metadata_) {
    std::cout << "  " << key << ": ";
    switch (value.type) {
      case GGUFMetadataType::UINT8:
        std::cout << static_cast<int>(value.u8);
        break;
      case GGUFMetadataType::INT8:
        std::cout << static_cast<int>(value.i8);
        break;
      case GGUFMetadataType::UINT16:
        std::cout << value.u16;
        break;
      case GGUFMetadataType::INT16:
        std::cout << value.i16;
        break;
      case GGUFMetadataType::UINT32:
        std::cout << value.u32;
        break;
      case GGUFMetadataType::INT32:
        std::cout << value.i32;
        break;
      case GGUFMetadataType::UINT64:
        std::cout << value.u64;
        break;
      case GGUFMetadataType::INT64:
        std::cout << value.i64;
        break;
      case GGUFMetadataType::FLOAT32:
        std::cout << value.f32;
        break;
      case GGUFMetadataType::FLOAT64:
        std::cout << value.f64;
        break;
      case GGUFMetadataType::BOOL:
        std::cout << (value.b ? "true" : "false");
        break;
      case GGUFMetadataType::STRING:
        std::cout << "\"" << value.str << "\"";
        break;
      case GGUFMetadataType::ARRAY:
        std::cout << "[array of " << value.arr->length << " elements]";
        break;
    }
    std::cout << "\n";
  }
}

void GGUFFile::print_tensors() const {
  std::cout << "\n=== Tensors (" << tensors_.size() << " tensors) ===\n";
  std::cout << std::left << std::setw(50) << "Name" << std::setw(10) << "Type"
            << std::setw(30) << "Shape" << std::setw(15) << "Size (bytes)"
            << "Offset\n";
  std::cout << std::string(115, '-') << "\n";

  for (const auto& tensor : tensors_) {
    std::cout << std::left << std::setw(50) << tensor.name << std::setw(10)
              << gguf_type_name(tensor.type);

    // Print shape
    std::string shape = "[";
    for (size_t i = 0; i < tensor.dimensions.size(); i++) {
      if (i > 0) shape += ", ";
      shape += std::to_string(tensor.dimensions[i]);
    }
    shape += "]";
    std::cout << std::setw(30) << shape << std::setw(15) << tensor.size
              << tensor.offset << "\n";
  }
}

}  // namespace registry
}  // namespace mlxr
