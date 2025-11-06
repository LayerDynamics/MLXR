// Copyright © 2025 MLXR Development
// GGUF (GGML Universal Format) file parser

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlxr {
namespace registry {

// GGUF format constants (from llama.cpp specification)
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_V1 = 1;
constexpr uint32_t GGUF_VERSION_V2 = 2;
constexpr uint32_t GGUF_VERSION_V3 = 3;  // Current version

// Metadata value types
enum class GGUFMetadataType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

// Tensor data types (GGML types)
enum class GGUFTensorType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
};

// Forward declarations
class GGUFMetadataValue;

// GGUF array value (for metadata arrays)
struct GGUFArray {
  GGUFMetadataType type;
  uint64_t length;
  std::vector<std::shared_ptr<GGUFMetadataValue>> values;
};

// Metadata value (variant-like structure)
class GGUFMetadataValue {
 public:
  GGUFMetadataType type;

  // Storage for different types
  union {
    uint8_t u8;
    int8_t i8;
    uint16_t u16;
    int16_t i16;
    uint32_t u32;
    int32_t i32;
    uint64_t u64;
    int64_t i64;
    float f32;
    double f64;
    bool b;
  };

  std::string str;                 // For STRING type
  std::shared_ptr<GGUFArray> arr;  // For ARRAY type

  GGUFMetadataValue() : type(GGUFMetadataType::UINT8), u64(0) {}

  // Helper getters
  uint8_t as_uint8() const { return u8; }
  int8_t as_int8() const { return i8; }
  uint16_t as_uint16() const { return u16; }
  int16_t as_int16() const { return i16; }
  uint32_t as_uint32() const { return u32; }
  int32_t as_int32() const { return i32; }
  uint64_t as_uint64() const { return u64; }
  int64_t as_int64() const { return i64; }
  float as_float32() const { return f32; }
  double as_float64() const { return f64; }
  bool as_bool() const { return b; }
  const std::string& as_string() const { return str; }
  const GGUFArray& as_array() const { return *arr; }
};

// Tensor information
struct GGUFTensorInfo {
  std::string name;
  uint32_t n_dimensions;
  std::vector<uint64_t> dimensions;  // Shape
  GGUFTensorType type;
  uint64_t offset;  // Offset in file (from data section start)
  uint64_t size;    // Size in bytes
};

// GGUF file header
struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

// Complete GGUF file structure
class GGUFFile {
 public:
  GGUFFile();
  ~GGUFFile();

  /**
   * Parse GGUF file from path
   * @param file_path Path to .gguf file
   * @return true if successful, false otherwise
   */
  bool parse(const std::string& file_path);

  /**
   * Parse GGUF file from already opened stream
   * @param stream Input stream positioned at start of GGUF file
   * @return true if successful, false otherwise
   */
  bool parse_stream(std::ifstream& stream);

  // Accessors
  const GGUFHeader& header() const { return header_; }
  const std::unordered_map<std::string, GGUFMetadataValue>& metadata() const {
    return metadata_;
  }
  const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
  uint64_t data_offset() const { return data_offset_; }

  // Metadata helpers
  bool has_metadata(const std::string& key) const;
  const GGUFMetadataValue* get_metadata(const std::string& key) const;

  // Common metadata getters (with defaults)
  std::string get_arch() const;
  uint32_t get_context_length() const;
  uint32_t get_embedding_length() const;
  uint32_t get_block_count() const;
  uint32_t get_feed_forward_length() const;
  uint32_t get_attention_head_count() const;
  uint32_t get_attention_head_count_kv() const;
  float get_rope_freq_base() const;
  std::string get_tokenizer_model() const;

  // Tensor helpers
  const GGUFTensorInfo* find_tensor(const std::string& name) const;
  std::vector<std::string> get_tensor_names() const;

  // Debug
  void print_info() const;
  void print_metadata() const;
  void print_tensors() const;

  // Error handling
  const std::string& error() const { return error_; }
  bool has_error() const { return !error_.empty(); }

 private:
  // Parsing helpers
  bool read_header(std::ifstream& stream);
  bool read_metadata(std::ifstream& stream);
  bool read_tensor_infos(std::ifstream& stream);
  bool read_string(std::ifstream& stream, std::string& out);
  bool read_metadata_value(std::ifstream& stream, GGUFMetadataType type,
                           GGUFMetadataValue& out);

  // Data members
  GGUFHeader header_;
  std::unordered_map<std::string, GGUFMetadataValue> metadata_;
  std::vector<GGUFTensorInfo> tensors_;
  uint64_t data_offset_;  // Byte offset where tensor data starts
  std::string error_;
};

// Utility functions
const char* gguf_type_name(GGUFTensorType type);
size_t gguf_type_size(GGUFTensorType type);   // Size per element
size_t gguf_block_size(GGUFTensorType type);  // Block size for quantized types
float gguf_bytes_per_weight(GGUFTensorType type);  // Average bytes per weight

// Calculate tensor size in bytes
uint64_t calculate_tensor_size(const GGUFTensorInfo& tensor);

// Convert GGUF type to MLX dtype string
std::string gguf_type_to_mlx_dtype(GGUFTensorType type);

}  // namespace registry
}  // namespace mlxr
