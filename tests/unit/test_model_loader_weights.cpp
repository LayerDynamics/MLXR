// Copyright © 2025 MLXR Development
// Unit tests for ModelLoader weight loading functionality

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "runtime/mmap_loader.h"
#include "registry/gguf_parser.h"
#include "server/model_loader.h"

namespace fs = std::filesystem;

namespace mlxr {
namespace server {
namespace test {

class ModelLoaderWeightsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temp directory for test files
    test_dir_ = fs::temp_directory_path() / "mlxr_test_weights";
    fs::create_directories(test_dir_);
  }

  void TearDown() override {
    // Clean up test directory
    if (fs::exists(test_dir_)) {
      fs::remove_all(test_dir_);
    }
  }

  // Create a simple test file with known content
  std::string create_test_file(const std::string& name, size_t size_mb) {
    fs::path file_path = test_dir_ / name;
    std::ofstream file(file_path, std::ios::binary);

    // Write size_mb megabytes of data
    std::vector<char> data(1024 * 1024, 'A');
    for (size_t i = 0; i < size_mb; i++) {
      file.write(data.data(), data.size());
    }

    file.close();
    return file_path.string();
  }

  fs::path test_dir_;
};

// Test basic weight loader initialization
TEST_F(ModelLoaderWeightsTest, LoadWeightsBasic) {
  // Create test file
  auto file_path = create_test_file("test_weights.bin", 10);

  // Create weight loader directly
  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  // Check file size
  EXPECT_EQ(loader.file_size(), 10 * 1024 * 1024);
  EXPECT_EQ(loader.file_path(), file_path);
  EXPECT_FALSE(loader.is_mapped());
}

// Test mapping entire file
TEST_F(ModelLoaderWeightsTest, MapAll) {
  auto file_path = create_test_file("test_weights.bin", 5);

  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  // Map entire file
  auto region = loader.map_all(false);
  ASSERT_TRUE(region.is_valid);
  EXPECT_EQ(region.size, 5 * 1024 * 1024);
  EXPECT_NE(region.data, nullptr);
  EXPECT_TRUE(loader.is_mapped());
}

// Test mapping with prefetch
TEST_F(ModelLoaderWeightsTest, MapWithPrefetch) {
  auto file_path = create_test_file("test_weights.bin", 5);

  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  // Map with prefetch
  auto region = loader.map_all(true);
  ASSERT_TRUE(region.is_valid);
  EXPECT_TRUE(loader.is_mapped());
}

// Test tensor registration
TEST_F(ModelLoaderWeightsTest, RegisterTensors) {
  auto file_path = create_test_file("test_weights.bin", 10);

  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  // Register a tensor
  WeightTensor tensor;
  tensor.name = "model.layers.0.weight";
  tensor.shape = {768, 768};
  tensor.file_offset = 0;
  tensor.data_size = 768 * 768 * 2;  // FP16
  tensor.dtype = "float16";

  loader.register_tensor(tensor);

  // Verify tensor was registered
  auto tensor_info = loader.get_tensor_info("model.layers.0.weight");
  ASSERT_TRUE(tensor_info.has_value());
  EXPECT_EQ(tensor_info->name, "model.layers.0.weight");
  EXPECT_EQ(tensor_info->shape.size(), 2);
  EXPECT_EQ(tensor_info->dtype, "float16");

  // List tensors
  auto names = loader.list_tensors();
  EXPECT_EQ(names.size(), 1);
  EXPECT_EQ(names[0], "model.layers.0.weight");
}

// Test mapping individual tensor
TEST_F(ModelLoaderWeightsTest, MapTensor) {
  auto file_path = create_test_file("test_weights.bin", 10);

  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  // Register tensor
  WeightTensor tensor;
  tensor.name = "test.weight";
  tensor.shape = {512, 512};
  tensor.file_offset = 1024;  // Offset 1KB
  tensor.data_size = 512 * 512 * 2;
  tensor.dtype = "float16";

  loader.register_tensor(tensor);

  // Map the tensor
  auto region = loader.map_tensor("test.weight", false);
  ASSERT_TRUE(region.is_valid);
  EXPECT_EQ(region.size, 512 * 512 * 2);
  EXPECT_NE(region.data, nullptr);
}

// Test stats
TEST_F(ModelLoaderWeightsTest, GetStats) {
  auto file_path = create_test_file("test_weights.bin", 10);

  MMapWeightLoader loader(file_path, true);
  ASSERT_TRUE(loader.initialize());

  auto stats = loader.get_stats();
  EXPECT_EQ(stats.total_file_size, 10 * 1024 * 1024);
  EXPECT_EQ(stats.total_mapped_bytes, 0);
  EXPECT_EQ(stats.num_registered_tensors, 0);

  // Map entire file
  loader.map_all(false);

  stats = loader.get_stats();
  EXPECT_GT(stats.total_mapped_bytes, 0);
}

// Test error handling - non-existent file
TEST_F(ModelLoaderWeightsTest, NonExistentFile) {
  MMapWeightLoader loader("/nonexistent/file.bin", true);
  EXPECT_FALSE(loader.initialize());
}

// Test error handling - empty file
TEST_F(ModelLoaderWeightsTest, EmptyFile) {
  fs::path file_path = test_dir_ / "empty.bin";
  std::ofstream file(file_path);
  file.close();

  MMapWeightLoader loader(file_path.string(), true);
  EXPECT_FALSE(loader.initialize());
}

}  // namespace test
}  // namespace server
}  // namespace mlxr
