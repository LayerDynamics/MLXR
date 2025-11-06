// Copyright © 2025 MLXR Development
// Unit tests for GGUF parser

#include "gguf_parser.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <vector>

using namespace mlxr::registry;

// Helper function to create a minimal valid GGUF file in memory
std::vector<uint8_t> create_minimal_gguf() {
  std::vector<uint8_t> data;

  // Helper to append bytes
  auto append = [&data](const void* ptr, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(ptr);
    data.insert(data.end(), bytes, bytes + size);
  };

  // Helper to append string
  auto append_string = [&append](const std::string& str) {
    uint64_t len = str.size();
    append(&len, sizeof(len));
    append(str.data(), str.size());
  };

  // Header
  uint32_t magic = GGUF_MAGIC;
  uint32_t version = GGUF_VERSION_V3;
  uint64_t tensor_count = 1;
  uint64_t metadata_kv_count = 3;

  append(&magic, sizeof(magic));
  append(&version, sizeof(version));
  append(&tensor_count, sizeof(tensor_count));
  append(&metadata_kv_count, sizeof(metadata_kv_count));

  // Metadata: general.architecture = "llama"
  append_string("general.architecture");
  uint32_t type = static_cast<uint32_t>(GGUFMetadataType::STRING);
  append(&type, sizeof(type));
  append_string("llama");

  // Metadata: llama.context_length = 2048
  append_string("llama.context_length");
  type = static_cast<uint32_t>(GGUFMetadataType::UINT32);
  append(&type, sizeof(type));
  uint32_t ctx_len = 2048;
  append(&ctx_len, sizeof(ctx_len));

  // Metadata: llama.embedding_length = 4096
  append_string("llama.embedding_length");
  type = static_cast<uint32_t>(GGUFMetadataType::UINT32);
  append(&type, sizeof(type));
  uint32_t embed_len = 4096;
  append(&embed_len, sizeof(embed_len));

  // Tensor info: "token_embd.weight"
  append_string("token_embd.weight");
  uint32_t n_dims = 2;
  append(&n_dims, sizeof(n_dims));
  uint64_t dim0 = 32000;  // vocab_size
  uint64_t dim1 = 4096;   // embedding_dim
  append(&dim0, sizeof(dim0));
  append(&dim1, sizeof(dim1));
  uint32_t tensor_type = static_cast<uint32_t>(GGUFTensorType::F16);
  append(&tensor_type, sizeof(tensor_type));
  uint64_t offset = 0;
  append(&offset, sizeof(offset));

  return data;
}

TEST(GGUFParserTest, ParseMinimalFile) {
  // Create minimal GGUF file
  auto data = create_minimal_gguf();

  // Write to temporary file
  std::string temp_path = "/tmp/test_minimal.gguf";
  std::ofstream out(temp_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()), data.size());
  out.close();

  // Parse file
  GGUFFile gguf;
  ASSERT_TRUE(gguf.parse(temp_path));
  EXPECT_FALSE(gguf.has_error());

  // Verify header
  EXPECT_EQ(gguf.header().magic, GGUF_MAGIC);
  EXPECT_EQ(gguf.header().version, GGUF_VERSION_V3);
  EXPECT_EQ(gguf.header().tensor_count, 1);
  EXPECT_EQ(gguf.header().metadata_kv_count, 3);

  // Verify metadata
  EXPECT_TRUE(gguf.has_metadata("general.architecture"));
  EXPECT_TRUE(gguf.has_metadata("llama.context_length"));
  EXPECT_TRUE(gguf.has_metadata("llama.embedding_length"));

  EXPECT_EQ(gguf.get_arch(), "llama");
  EXPECT_EQ(gguf.get_context_length(), 2048);
  EXPECT_EQ(gguf.get_embedding_length(), 4096);

  // Verify tensors
  EXPECT_EQ(gguf.tensors().size(), 1);
  const auto* tensor = gguf.find_tensor("token_embd.weight");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->n_dimensions, 2);
  EXPECT_EQ(tensor->dimensions[0], 32000);
  EXPECT_EQ(tensor->dimensions[1], 4096);
  EXPECT_EQ(tensor->type, GGUFTensorType::F16);

  // Clean up
  std::remove(temp_path.c_str());
}

TEST(GGUFParserTest, InvalidMagic) {
  // Create file with wrong magic
  std::vector<uint8_t> data(16, 0);
  uint32_t bad_magic = 0x12345678;
  std::memcpy(data.data(), &bad_magic, sizeof(bad_magic));

  std::string temp_path = "/tmp/test_bad_magic.gguf";
  std::ofstream out(temp_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()), data.size());
  out.close();

  GGUFFile gguf;
  EXPECT_FALSE(gguf.parse(temp_path));
  EXPECT_TRUE(gguf.has_error());

  std::remove(temp_path.c_str());
}

TEST(GGUFParserTest, TensorSizeCalculation) {
  GGUFTensorInfo tensor;
  tensor.n_dimensions = 2;
  tensor.dimensions = {4096, 11008};
  tensor.type = GGUFTensorType::F16;

  uint64_t size = calculate_tensor_size(tensor);
  EXPECT_EQ(size, 4096 * 11008 * 2);  // FP16 = 2 bytes

  // Test quantized tensor
  tensor.type = GGUFTensorType::Q4_0;
  size = calculate_tensor_size(tensor);
  uint64_t n_elements = 4096 * 11008;
  uint64_t n_blocks = (n_elements + 31) / 32;  // Q4_0 block size = 32
  EXPECT_EQ(size, n_blocks * 18);              // Q4_0: 18 bytes per block
}

TEST(GGUFParserTest, TypeNameConversion) {
  EXPECT_STREQ(gguf_type_name(GGUFTensorType::F32), "F32");
  EXPECT_STREQ(gguf_type_name(GGUFTensorType::F16), "F16");
  EXPECT_STREQ(gguf_type_name(GGUFTensorType::Q4_0), "Q4_0");
  EXPECT_STREQ(gguf_type_name(GGUFTensorType::Q4_K), "Q4_K");
}

TEST(GGUFParserTest, BytesPerWeight) {
  EXPECT_FLOAT_EQ(gguf_bytes_per_weight(GGUFTensorType::F32), 4.0f);
  EXPECT_FLOAT_EQ(gguf_bytes_per_weight(GGUFTensorType::F16), 2.0f);
  EXPECT_LT(gguf_bytes_per_weight(GGUFTensorType::Q4_0), 1.0f);
  EXPECT_LT(gguf_bytes_per_weight(GGUFTensorType::Q4_K), 1.0f);
}

TEST(GGUFParserTest, MLXDtypeConversion) {
  EXPECT_EQ(gguf_type_to_mlx_dtype(GGUFTensorType::F32), "float32");
  EXPECT_EQ(gguf_type_to_mlx_dtype(GGUFTensorType::F16), "float16");
  EXPECT_EQ(gguf_type_to_mlx_dtype(GGUFTensorType::I32), "int32");
  // Quantized types -> float16 (will be dequantized)
  EXPECT_EQ(gguf_type_to_mlx_dtype(GGUFTensorType::Q4_0), "float16");
}
