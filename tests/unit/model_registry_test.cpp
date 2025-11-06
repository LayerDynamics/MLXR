// Copyright © 2025 MLXR Development
// Model registry unit tests

#include "model_registry.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace mlxr::registry;

namespace {

class ModelRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use in-memory database for tests
    test_db_path_ = ":memory:";
    registry_ = std::make_unique<ModelRegistry>(test_db_path_);
    ASSERT_TRUE(registry_->initialize());
  }

  void TearDown() override { registry_.reset(); }

  std::string test_db_path_;
  std::unique_ptr<ModelRegistry> registry_;
};

// Test basic model registration
TEST_F(ModelRegistryTest, RegisterModel) {
  ModelInfo info;
  info.name = "Test Llama 7B";
  info.model_id = "test-llama-7b";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/test_model.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 7000000000;
  info.context_length = 2048;
  info.hidden_size = 4096;
  info.num_layers = 32;
  info.num_heads = 32;
  info.num_kv_heads = 32;
  info.vocab_size = 32000;
  info.quant_type = QuantizationType::Q4_K;
  info.tokenizer_type = "llama";

  int64_t model_id = registry_->register_model(info);
  EXPECT_GT(model_id, 0);

  // Retrieve and verify
  auto retrieved = registry_->get_model(model_id);
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved->name, "Test Llama 7B");
  EXPECT_EQ(retrieved->model_id, "test-llama-7b");
  EXPECT_EQ(retrieved->architecture, ModelArchitecture::LLAMA);
  EXPECT_EQ(retrieved->context_length, 2048);
  EXPECT_EQ(retrieved->hidden_size, 4096);
}

// Test duplicate model ID rejection
TEST_F(ModelRegistryTest, DuplicateModelId) {
  ModelInfo info;
  info.name = "Model 1";
  info.model_id = "duplicate-test";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/model1.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 1000000;

  int64_t id1 = registry_->register_model(info);
  EXPECT_GT(id1, 0);

  // Try to register with same model_id
  info.name = "Model 2";
  info.file_path = "/tmp/model2.gguf";
  int64_t id2 = registry_->register_model(info);
  EXPECT_EQ(id2, -1);  // Should fail
}

// Test listing models
TEST_F(ModelRegistryTest, ListModels) {
  // Register multiple models
  for (int i = 0; i < 5; i++) {
    ModelInfo info;
    info.name = "Model " + std::to_string(i);
    info.model_id = "model-" + std::to_string(i);
    info.architecture = ModelArchitecture::LLAMA;
    info.file_path = "/tmp/model" + std::to_string(i) + ".gguf";
    info.format = ModelFormat::GGUF;
    info.file_size = 1000000 * (i + 1);

    int64_t id = registry_->register_model(info);
    EXPECT_GT(id, 0);
  }

  auto models = registry_->list_models();
  EXPECT_EQ(models.size(), 5);
}

// Test filtering by architecture
TEST_F(ModelRegistryTest, FilterByArchitecture) {
  // Register Llama model
  ModelInfo llama;
  llama.name = "Llama 7B";
  llama.model_id = "llama-7b";
  llama.architecture = ModelArchitecture::LLAMA;
  llama.file_path = "/tmp/llama.gguf";
  llama.format = ModelFormat::GGUF;
  llama.file_size = 7000000000;
  registry_->register_model(llama);

  // Register Mistral model
  ModelInfo mistral;
  mistral.name = "Mistral 7B";
  mistral.model_id = "mistral-7b";
  mistral.architecture = ModelArchitecture::MISTRAL;
  mistral.file_path = "/tmp/mistral.gguf";
  mistral.format = ModelFormat::GGUF;
  mistral.file_size = 7000000000;
  registry_->register_model(mistral);

  // Filter by Llama architecture
  QueryOptions options;
  options.architecture = ModelArchitecture::LLAMA;
  auto llama_models = registry_->list_models(options);

  EXPECT_EQ(llama_models.size(), 1);
  EXPECT_EQ(llama_models[0].model_id, "llama-7b");

  // Filter by Mistral architecture
  options.architecture = ModelArchitecture::MISTRAL;
  auto mistral_models = registry_->list_models(options);

  EXPECT_EQ(mistral_models.size(), 1);
  EXPECT_EQ(mistral_models[0].model_id, "mistral-7b");
}

// Test model tags
TEST_F(ModelRegistryTest, ModelTags) {
  ModelInfo info;
  info.name = "Tagged Model";
  info.model_id = "tagged-model";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/tagged.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 1000000;

  int64_t model_id = registry_->register_model(info);
  ASSERT_GT(model_id, 0);

  // Add tags
  std::unordered_map<std::string, std::string> tags = {
      {"task", "chat"}, {"language", "english"}, {"size", "7b"}};
  EXPECT_TRUE(registry_->add_tags(model_id, tags));

  // Get tags
  auto retrieved_tags = registry_->get_tags(model_id);
  EXPECT_EQ(retrieved_tags.size(), 3);
  EXPECT_EQ(retrieved_tags["task"], "chat");
  EXPECT_EQ(retrieved_tags["language"], "english");
  EXPECT_EQ(retrieved_tags["size"], "7b");

  // Filter by tag (using required_tags vector)
  QueryOptions options;
  options.required_tags.push_back("task:chat");
  auto chat_models = registry_->list_models(options);
  EXPECT_EQ(chat_models.size(), 1);
  EXPECT_EQ(chat_models[0].model_id, "tagged-model");
}

// Test adapter registration
TEST_F(ModelRegistryTest, RegisterAdapter) {
  // First register base model
  ModelInfo base;
  base.name = "Base Model";
  base.model_id = "base-model";
  base.architecture = ModelArchitecture::LLAMA;
  base.file_path = "/tmp/base.gguf";
  base.format = ModelFormat::GGUF;
  base.file_size = 7000000000;

  int64_t base_id = registry_->register_model(base);
  ASSERT_GT(base_id, 0);

  // Register adapter
  AdapterInfo adapter;
  adapter.base_model_id = base_id;
  adapter.name = "Test LoRA";
  adapter.adapter_id = "test-lora";
  adapter.file_path = "/tmp/lora.safetensors";
  adapter.adapter_type = "lora";
  adapter.rank = 8;
  adapter.scale = 1.0;

  int64_t adapter_id = registry_->register_adapter(adapter);
  EXPECT_GT(adapter_id, 0);

  // Get adapters for base model
  auto adapters = registry_->get_adapters(base_id);
  EXPECT_EQ(adapters.size(), 1);
  EXPECT_EQ(adapters[0].name, "Test LoRA");
  EXPECT_EQ(adapters[0].rank, 8);
}

// Test model deletion
TEST_F(ModelRegistryTest, DeleteModel) {
  ModelInfo info;
  info.name = "Model to Delete";
  info.model_id = "delete-me";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/delete.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 1000000;

  int64_t model_id = registry_->register_model(info);
  ASSERT_GT(model_id, 0);

  // Verify it exists
  auto retrieved = registry_->get_model(model_id);
  ASSERT_TRUE(retrieved.has_value());

  // Remove it
  EXPECT_TRUE(registry_->remove_model(model_id, false));

  // Verify it's gone
  retrieved = registry_->get_model(model_id);
  EXPECT_FALSE(retrieved.has_value());
}

// Test updating model
TEST_F(ModelRegistryTest, UpdateModel) {
  ModelInfo info;
  info.name = "Original Name";
  info.model_id = "update-test";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/original.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 1000000;
  info.is_loaded = false;

  int64_t model_id = registry_->register_model(info);
  ASSERT_GT(model_id, 0);

  // Update fields
  info.id = model_id;
  info.name = "Updated Name";

  EXPECT_TRUE(registry_->update_model(info));

  // Update loaded state separately
  registry_->set_model_loaded(model_id, true);

  // Verify update
  auto updated = registry_->get_model(model_id);
  ASSERT_TRUE(updated.has_value());
  EXPECT_EQ(updated->name, "Updated Name");
  EXPECT_TRUE(updated->is_loaded);
}

// Test search by name
TEST_F(ModelRegistryTest, SearchByName) {
  // Register models with different names
  ModelInfo info1;
  info1.name = "Llama 2 7B Chat";
  info1.model_id = "llama2-7b-chat";
  info1.architecture = ModelArchitecture::LLAMA;
  info1.file_path = "/tmp/llama2.gguf";
  info1.format = ModelFormat::GGUF;
  info1.file_size = 7000000000;
  registry_->register_model(info1);

  ModelInfo info2;
  info2.name = "Mistral 7B Instruct";
  info2.model_id = "mistral-7b-instruct";
  info2.architecture = ModelArchitecture::MISTRAL;
  info2.file_path = "/tmp/mistral.gguf";
  info2.format = ModelFormat::GGUF;
  info2.file_size = 7000000000;
  registry_->register_model(info2);

  // Search for "llama"
  QueryOptions options;
  options.search_term = "llama";
  auto results = registry_->list_models(options);

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].model_id, "llama2-7b-chat");
}

// Test touch (update last used timestamp)
TEST_F(ModelRegistryTest, TouchModel) {
  ModelInfo info;
  info.name = "Touch Test";
  info.model_id = "touch-test";
  info.architecture = ModelArchitecture::LLAMA;
  info.file_path = "/tmp/touch.gguf";
  info.format = ModelFormat::GGUF;
  info.file_size = 1000000;

  int64_t model_id = registry_->register_model(info);
  ASSERT_GT(model_id, 0);

  // Get initial timestamp
  auto initial = registry_->get_model(model_id);
  ASSERT_TRUE(initial.has_value());
  int64_t initial_timestamp = initial->last_used_timestamp;

  // Wait a bit (timestamp has second precision)
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Touch the model
  registry_->touch_model(model_id);

  // Verify timestamp updated
  auto touched = registry_->get_model(model_id);
  ASSERT_TRUE(touched.has_value());
  EXPECT_GT(touched->last_used_timestamp, initial_timestamp);
}

}  // namespace
