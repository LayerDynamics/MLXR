// Copyright © 2025 MLXR Development
// Integration tests for ModelLoader end-to-end functionality

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include "../../daemon/registry/model_registry.h"
#include "../../daemon/server/model_loader.h"

namespace fs = std::filesystem;

namespace mlxr {
namespace server {
namespace test {

class ModelLoaderIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temp directory
    test_dir_ = fs::temp_directory_path() / "mlxr_test_integration";
    fs::create_directories(test_dir_);

    // Create in-memory registry
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

// Test LoadModelConfig structure
TEST_F(ModelLoaderIntegrationTest, ConfigStructure) {
  LoadModelConfig config;

  // Test defaults
  EXPECT_EQ(config.kv_block_size, 32);
  EXPECT_EQ(config.kv_num_blocks, 256);
  EXPECT_EQ(config.max_new_tokens, 128);
  EXPECT_TRUE(config.use_cached_attention);
  EXPECT_FALSE(config.prefetch_weights);
  EXPECT_FALSE(config.lock_weights);

  // Test custom values
  config.kv_block_size = 16;
  config.kv_num_blocks = 512;
  config.max_new_tokens = 256;
  config.prefetch_weights = true;

  EXPECT_EQ(config.kv_block_size, 16);
  EXPECT_EQ(config.kv_num_blocks, 512);
  EXPECT_EQ(config.max_new_tokens, 256);
  EXPECT_TRUE(config.prefetch_weights);
}

// Test LoadedModel structure
TEST_F(ModelLoaderIntegrationTest, LoadedModelStructure) {
  LoadedModel loaded;

  // Initially all should be null/empty
  EXPECT_EQ(loaded.model, nullptr);
  EXPECT_EQ(loaded.pager, nullptr);
  EXPECT_EQ(loaded.tokenizer, nullptr);
  EXPECT_EQ(loaded.engine, nullptr);
  EXPECT_EQ(loaded.loader, nullptr);
}

// Test registry integration
TEST_F(ModelLoaderIntegrationTest, RegistryIntegration) {
  // Register a model
  registry::ModelInfo info;
  info.model_id = "test-llama-1b";
  info.name = "Test Llama 1B";
  info.family = "llama";
  info.format = registry::ModelFormat::GGUF;
  info.file_path = (test_dir_ / "model.gguf").string();
  info.tokenizer_path = (test_dir_ / "tokenizer.model").string();
  info.dtype = "Q4_0";
  info.quantization = "Q4_0";
  info.num_params = 1'100'000'000;
  info.hidden_size = 2048;
  info.num_layers = 22;
  info.num_heads = 32;
  info.num_kv_heads = 4;
  info.intermediate_size = 5632;
  info.vocab_size = 32000;
  info.context_length = 2048;
  info.rope_freq_base = 10000.0f;

  auto model_id = registry_->register_model(info);
  ASSERT_GT(model_id, 0);

  // Query by identifier
  auto retrieved = registry_->get_model_by_identifier("test-llama-1b");
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved->name, "Test Llama 1B");
  EXPECT_EQ(retrieved->num_layers, 22);
  EXPECT_EQ(retrieved->num_kv_heads, 4);
  EXPECT_EQ(retrieved->hidden_size, 2048);

  // Query by ID
  auto retrieved_by_id = registry_->get_model(model_id);
  ASSERT_TRUE(retrieved_by_id.has_value());
  EXPECT_EQ(retrieved_by_id->name, "Test Llama 1B");
}

// Test error handling - missing files
TEST_F(ModelLoaderIntegrationTest, ErrorHandlingMissingFiles) {
  // Register model with non-existent files
  registry::ModelInfo info;
  info.model_id = "missing-model";
  info.name = "Missing Model";
  info.family = "llama";
  info.format = registry::ModelFormat::GGUF;
  info.file_path = "/nonexistent/model.gguf";
  info.tokenizer_path = "/nonexistent/tokenizer.model";
  info.dtype = "Q4_0";
  info.num_params = 1'000'000'000;
  info.hidden_size = 2048;
  info.num_layers = 22;
  info.num_heads = 32;
  info.num_kv_heads = 4;
  info.intermediate_size = 5632;
  info.vocab_size = 32000;
  info.context_length = 2048;
  info.rope_freq_base = 10000.0f;

  registry_->register_model(info);

  // Try to load - should fail
  LoadModelConfig config;
  auto result = loader_->load_model("missing-model", config);

  EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(loader_->last_error().empty());
}

// Test error handling - model not in registry
TEST_F(ModelLoaderIntegrationTest, ErrorHandlingNotInRegistry) {
  LoadModelConfig config;

  auto result = loader_->load_model("nonexistent", config);

  EXPECT_FALSE(result.has_value());
  EXPECT_FALSE(loader_->last_error().empty());
  EXPECT_NE(loader_->last_error().find("not found"), std::string::npos);
}

// Test multiple models in registry
TEST_F(ModelLoaderIntegrationTest, MultipleModels) {
  // Register multiple models
  for (int i = 0; i < 3; i++) {
    registry::ModelInfo info;
    info.model_id = "model-" + std::to_string(i);
    info.name = "Model " + std::to_string(i);
    info.family = "llama";
    info.format = registry::ModelFormat::GGUF;
    info.file_path = (test_dir_ / ("model" + std::to_string(i) + ".gguf")).string();
    info.tokenizer_path = (test_dir_ / ("tokenizer" + std::to_string(i) + ".model")).string();
    info.dtype = "Q4_0";
    info.num_params = 1'000'000'000;
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
  }

  // List all models
  auto models = registry_->list_models();
  EXPECT_EQ(models.size(), 3);

  // Verify each one
  for (int i = 0; i < 3; i++) {
    auto model = registry_->get_model_by_identifier("model-" + std::to_string(i));
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(model->name, "Model " + std::to_string(i));
  }
}

// Test model info validation
TEST_F(ModelLoaderIntegrationTest, ModelInfoValidation) {
  registry::ModelInfo info;
  info.model_id = "validation-test";
  info.name = "Validation Test";
  info.family = "llama";
  info.format = registry::ModelFormat::GGUF;
  info.file_path = (test_dir_ / "model.gguf").string();
  info.tokenizer_path = (test_dir_ / "tokenizer.model").string();

  // Set valid architecture parameters
  info.hidden_size = 2048;
  info.num_layers = 22;
  info.num_heads = 32;
  info.num_kv_heads = 4;  // GQA: 4 KV heads, 32 query heads
  info.intermediate_size = 5632;
  info.vocab_size = 32000;
  info.context_length = 2048;

  // Verify KV heads <= num_heads
  EXPECT_LE(info.num_kv_heads, info.num_heads);

  // Verify head_dim calculation
  int head_dim = info.hidden_size / info.num_heads;
  EXPECT_EQ(head_dim, 64);  // 2048 / 32 = 64

  // Verify hidden_size is divisible by num_heads
  EXPECT_EQ(info.hidden_size % info.num_heads, 0);
}

// Test GQA vs MHA configurations
TEST_F(ModelLoaderIntegrationTest, GQAVsMHA) {
  // GQA model (Grouped Query Attention)
  registry::ModelInfo gqa_info;
  gqa_info.model_id = "gqa-model";
  gqa_info.name = "GQA Model";
  gqa_info.family = "llama";
  gqa_info.format = registry::ModelFormat::GGUF;
  gqa_info.file_path = (test_dir_ / "gqa.gguf").string();
  gqa_info.tokenizer_path = (test_dir_ / "tokenizer.model").string();
  gqa_info.hidden_size = 2048;
  gqa_info.num_layers = 22;
  gqa_info.num_heads = 32;
  gqa_info.num_kv_heads = 4;  // GQA: 4 KV heads < 32 query heads

  // MHA model (Multi-Head Attention)
  registry::ModelInfo mha_info = gqa_info;
  mha_info.model_id = "mha-model";
  mha_info.name = "MHA Model";
  mha_info.num_kv_heads = 32;  // MHA: KV heads == query heads

  // Register both
  auto gqa_id = registry_->register_model(gqa_info);
  auto mha_id = registry_->register_model(mha_info);

  ASSERT_GT(gqa_id, 0);
  ASSERT_GT(mha_id, 0);

  // Verify
  auto gqa = registry_->get_model(gqa_id);
  auto mha = registry_->get_model(mha_id);

  ASSERT_TRUE(gqa.has_value());
  ASSERT_TRUE(mha.has_value());

  EXPECT_EQ(gqa->num_kv_heads, 4);  // GQA
  EXPECT_EQ(mha->num_kv_heads, 32);  // MHA
}

}  // namespace test
}  // namespace server
}  // namespace mlxr

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
