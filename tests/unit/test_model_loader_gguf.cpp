// Copyright © 2025 MLXR Development
// Unit tests for ModelLoader GGUF functionality

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include "runtime/mmap_loader.h"
#include "registry/gguf_parser.h"
#include "registry/model_registry.h"
#include "server/model_loader.h"

namespace fs = std::filesystem;

namespace mlxr {
namespace server {
namespace test {

class ModelLoaderGGUFTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temp directory
    test_dir_ = fs::temp_directory_path() / "mlxr_test_gguf";
    fs::create_directories(test_dir_);

    // Create in-memory registry for testing
    registry_ = std::make_shared<registry::ModelRegistry>(":memory:");
    ASSERT_TRUE(registry_->initialize());

    // Create model loader
    loader_ = std::make_unique<ModelLoader>(registry_);
  }

  void TearDown() override {
    if (fs::exists(test_dir_)) {
      fs::remove_all(test_dir_);
    }
  }

  fs::path test_dir_;
  std::shared_ptr<registry::ModelRegistry> registry_;
  std::unique_ptr<ModelLoader> loader_;
};

// Test GGUF parsing integration
// Note: This is a unit test that doesn't require actual GGUF files
// For full integration testing, see test_model_loader_full.cpp
TEST_F(ModelLoaderGGUFTest, GGUFParserBasic) {
  // Test that GGUF parser is accessible and has the right API
  registry::GGUFFile gguf;

  // Parsing a non-existent file should fail
  EXPECT_FALSE(gguf.parse("/nonexistent/file.gguf"));
  EXPECT_FALSE(gguf.error().empty());
}

// Test GGUF type conversions
TEST_F(ModelLoaderGGUFTest, TypeConversions) {
  // Test GGUF type to MLX dtype conversion
  using registry::GGUFTensorType;

  // FP32
  auto dtype_f32 = registry::gguf_type_to_mlx_dtype(GGUFTensorType::F32);
  EXPECT_EQ(dtype_f32, "float32");

  // FP16
  auto dtype_f16 = registry::gguf_type_to_mlx_dtype(GGUFTensorType::F16);
  EXPECT_EQ(dtype_f16, "float16");

  // Q4_0
  auto dtype_q40 = registry::gguf_type_to_mlx_dtype(GGUFTensorType::Q4_0);
  EXPECT_FALSE(dtype_q40.empty());

  // Test type names
  auto name_f32 = registry::gguf_type_name(GGUFTensorType::F32);
  EXPECT_EQ(name_f32, "F32");

  auto name_f16 = registry::gguf_type_name(GGUFTensorType::F16);
  EXPECT_EQ(name_f16, "F16");
}

// Test GGUF block sizes
TEST_F(ModelLoaderGGUFTest, BlockSizes) {
  using registry::GGUFTensorType;

  // Non-quantized types have block size 1
  EXPECT_EQ(registry::gguf_block_size(GGUFTensorType::F32), 1);
  EXPECT_EQ(registry::gguf_block_size(GGUFTensorType::F16), 1);

  // Quantized types have specific block sizes
  auto q4_0_block = registry::gguf_block_size(GGUFTensorType::Q4_0);
  EXPECT_GT(q4_0_block, 1);

  auto q8_0_block = registry::gguf_block_size(GGUFTensorType::Q8_0);
  EXPECT_GT(q8_0_block, 1);
}

// Test model registry integration
TEST_F(ModelLoaderGGUFTest, RegistryIntegration) {
  // Register a test model
  registry::ModelInfo info;
  info.model_id = "test-model";
  info.name = "Test Model";
  info.architecture = registry::ModelArchitecture::LLAMA;
  info.format = registry::ModelFormat::GGUF;
  info.file_path = (test_dir_ / "test.gguf").string();
  info.tokenizer_path = (test_dir_ / "tokenizer.model").string();
  info.quant_type = registry::QuantizationType::Q4_0;
  info.param_count = 1'100'000'000;
  info.hidden_size = 2048;
  info.num_layers = 22;
  info.num_heads = 32;
  info.num_kv_heads = 4;
  info.intermediate_size = 5632;
  info.vocab_size = 32000;
  info.context_length = 2048;
  info.rope_freq_base = 10000.0f;

  auto model_id = registry_->register_model(info);
  EXPECT_GT(model_id, 0);

  // Query back
  auto retrieved = registry_->get_model_by_identifier("test-model");
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved->name, "Test Model");
  EXPECT_EQ(retrieved->format, registry::ModelFormat::GGUF);
}

// Test LoadModelConfig defaults
TEST_F(ModelLoaderGGUFTest, LoadConfigDefaults) {
  LoadModelConfig config;

  // Check defaults
  EXPECT_EQ(config.kv_block_size, 32);
  EXPECT_EQ(config.kv_num_blocks, 8192);
  EXPECT_EQ(config.max_new_tokens, 2048);
  EXPECT_TRUE(config.use_cached_attention);
  EXPECT_TRUE(config.prefetch_weights);
  EXPECT_FALSE(config.lock_weights);
}

// Test error handling - model not found
TEST_F(ModelLoaderGGUFTest, ModelNotFound) {
  LoadModelConfig config;

  auto result = loader_->load_model("nonexistent-model", config);
  EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(loader_->last_error().empty());
  EXPECT_NE(loader_->last_error().find("not found"), std::string::npos);
}

}  // namespace test
}  // namespace server
}  // namespace mlxr

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
