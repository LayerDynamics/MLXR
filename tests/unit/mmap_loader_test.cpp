// Copyright © 2025 MLXR Development
// Memory-mapped weight loader unit tests

#include "runtime/mmap_loader.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <fstream>

using namespace mlxr;

namespace {

class MMapLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary test file
    test_file_path_ = "/tmp/mlxr_mmap_test.bin";
    create_test_file();
  }

  void TearDown() override {
    // Clean up test file
    unlink(test_file_path_.c_str());
  }

  void create_test_file() {
    std::ofstream file(test_file_path_, std::ios::binary);

    // Write 16KB of test data
    test_data_.resize(16 * 1024);
    for (size_t i = 0; i < test_data_.size(); i++) {
      test_data_[i] = static_cast<char>(i % 256);
    }

    file.write(test_data_.data(), test_data_.size());
    file.close();
  }

  std::string test_file_path_;
  std::vector<char> test_data_;
};

// Test basic initialization
TEST_F(MMapLoaderTest, Initialization) {
  MMapWeightLoader loader(test_file_path_);
  EXPECT_TRUE(loader.initialize());
  EXPECT_EQ(loader.file_size(), test_data_.size());
  EXPECT_EQ(loader.file_path(), test_file_path_);
}

// Test initialization with non-existent file
TEST_F(MMapLoaderTest, InitializationNonExistentFile) {
  MMapWeightLoader loader("/tmp/nonexistent_file.bin");
  EXPECT_FALSE(loader.initialize());
}

// Test tensor registration
TEST_F(MMapLoaderTest, TensorRegistration) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  WeightTensor tensor;
  tensor.name = "test.weight";
  tensor.shape = {128, 256};
  tensor.file_offset = 0;
  tensor.data_size = 1024;
  tensor.dtype = "fp32";

  loader.register_tensor(tensor);

  auto retrieved = loader.get_tensor_info("test.weight");
  ASSERT_TRUE(retrieved.has_value());
  EXPECT_EQ(retrieved->name, "test.weight");
  EXPECT_EQ(retrieved->file_offset, 0);
  EXPECT_EQ(retrieved->data_size, 1024);
}

// Test listing tensors
TEST_F(MMapLoaderTest, ListTensors) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  // Register multiple tensors
  for (int i = 0; i < 5; i++) {
    WeightTensor tensor;
    tensor.name = "tensor_" + std::to_string(i);
    tensor.file_offset = i * 1024;
    tensor.data_size = 1024;
    loader.register_tensor(tensor);
  }

  auto tensor_names = loader.list_tensors();
  EXPECT_EQ(tensor_names.size(), 5);
}

// Test mapping entire file
TEST_F(MMapLoaderTest, MapAll) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_all(false);
  ASSERT_TRUE(region.is_valid);
  EXPECT_EQ(region.size, test_data_.size());

  // Verify data
  char* data = region.as<char>();
  for (size_t i = 0; i < std::min(size_t(1024), test_data_.size()); i++) {
    EXPECT_EQ(data[i], test_data_[i]);
  }

  EXPECT_TRUE(loader.is_mapped());
}

// Test mapping specific region
TEST_F(MMapLoaderTest, MapRegion) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  size_t offset = 1024;
  size_t size = 2048;

  auto region = loader.map_region(offset, size, false);
  ASSERT_TRUE(region.is_valid);
  EXPECT_EQ(region.size, size);
  EXPECT_EQ(region.file_offset, offset);

  // Verify data
  char* data = region.as<char>();
  for (size_t i = 0; i < size; i++) {
    EXPECT_EQ(data[i], test_data_[offset + i]);
  }
}

// Test mapping by tensor name
TEST_F(MMapLoaderTest, MapTensor) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  WeightTensor tensor;
  tensor.name = "test.weight";
  tensor.file_offset = 512;
  tensor.data_size = 1024;
  loader.register_tensor(tensor);

  auto region = loader.map_tensor("test.weight", false);
  ASSERT_TRUE(region.is_valid);
  EXPECT_EQ(region.size, 1024);

  // Verify data
  char* data = region.as<char>();
  for (size_t i = 0; i < 1024; i++) {
    EXPECT_EQ(data[i], test_data_[512 + i]);
  }
}

// Test mapping non-existent tensor
TEST_F(MMapLoaderTest, MapNonExistentTensor) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_tensor("nonexistent", false);
  EXPECT_FALSE(region.is_valid);
}

// Test statistics
TEST_F(MMapLoaderTest, Statistics) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto stats_before = loader.get_stats();
  EXPECT_EQ(stats_before.total_file_size, test_data_.size());
  EXPECT_EQ(stats_before.total_mapped_bytes, 0);
  EXPECT_EQ(stats_before.num_active_mappings, 0);

  // Map a region
  auto region = loader.map_region(0, 1024, false);
  ASSERT_TRUE(region.is_valid);

  auto stats_after = loader.get_stats();
  EXPECT_GT(stats_after.total_mapped_bytes, 0);
  EXPECT_GT(stats_after.num_active_mappings, 0);
}

// Test unmapping
TEST_F(MMapLoaderTest, Unmap) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_region(0, 1024, false);
  ASSERT_TRUE(region.is_valid);

  auto stats_before = loader.get_stats();
  size_t mapped_before = stats_before.total_mapped_bytes;

  loader.unmap_region(region);

  auto stats_after = loader.get_stats();
  EXPECT_LT(stats_after.total_mapped_bytes, mapped_before);
}

// Test memory advice
TEST_F(MMapLoaderTest, MemoryAdvice) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_region(0, 4096, false);
  ASSERT_TRUE(region.is_valid);

  // Test various advice patterns
  EXPECT_TRUE(
      loader.advise(region, MMapWeightLoader::AdvicePattern::SEQUENTIAL));
  EXPECT_TRUE(loader.advise(region, MMapWeightLoader::AdvicePattern::RANDOM));
  EXPECT_TRUE(loader.advise(region, MMapWeightLoader::AdvicePattern::NORMAL));
}

// Test region at() method
TEST_F(MMapLoaderTest, RegionAt) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_region(0, 1024, false);
  ASSERT_TRUE(region.is_valid);

  // Test accessing data at offset
  void* ptr = region.at(512);
  ASSERT_NE(ptr, nullptr);

  char* data = static_cast<char*>(ptr);
  EXPECT_EQ(data[0], test_data_[512]);

  // Test out of bounds
  void* bad_ptr = region.at(2048);
  EXPECT_EQ(bad_ptr, nullptr);
}

// Test typed pointer access
TEST_F(MMapLoaderTest, TypedPointer) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto region = loader.map_region(0, 1024, false);
  ASSERT_TRUE(region.is_valid);

  // Access as char*
  char* char_ptr = region.as<char>();
  EXPECT_EQ(char_ptr[0], test_data_[0]);

  // Access as int32_t*
  int32_t* int_ptr = region.as<int32_t>();
  EXPECT_NE(int_ptr, nullptr);
}

// Test move constructor
TEST_F(MMapLoaderTest, MoveConstructor) {
  MMapWeightLoader loader1(test_file_path_);
  ASSERT_TRUE(loader1.initialize());

  auto region = loader1.map_region(0, 1024, false);
  ASSERT_TRUE(region.is_valid);

  // Move construct
  MMapWeightLoader loader2(std::move(loader1));

  EXPECT_EQ(loader2.file_size(), test_data_.size());
  EXPECT_GT(loader2.get_stats().total_mapped_bytes, 0);
}

// Test move assignment
TEST_F(MMapLoaderTest, MoveAssignment) {
  MMapWeightLoader loader1(test_file_path_);
  ASSERT_TRUE(loader1.initialize());

  MMapWeightLoader loader2("/tmp/dummy.bin", true);

  // Move assign
  loader2 = std::move(loader1);

  EXPECT_EQ(loader2.file_size(), test_data_.size());
}

// Test page size
TEST_F(MMapLoaderTest, PageSize) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  auto stats = loader.get_stats();
  EXPECT_GT(stats.page_size, 0);
  // Page size should be power of 2
  EXPECT_EQ(stats.page_size & (stats.page_size - 1), 0);
}

// Test multiple mappings
TEST_F(MMapLoaderTest, MultipleMappings) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  // Map multiple non-overlapping regions
  auto region1 = loader.map_region(0, 1024, false);
  auto region2 = loader.map_region(1024, 1024, false);
  auto region3 = loader.map_region(2048, 1024, false);

  ASSERT_TRUE(region1.is_valid);
  ASSERT_TRUE(region2.is_valid);
  ASSERT_TRUE(region3.is_valid);

  // Verify each region has correct data
  char* data1 = region1.as<char>();
  char* data2 = region2.as<char>();
  char* data3 = region3.as<char>();

  EXPECT_EQ(data1[0], test_data_[0]);
  EXPECT_EQ(data2[0], test_data_[1024]);
  EXPECT_EQ(data3[0], test_data_[2048]);

  auto stats = loader.get_stats();
  EXPECT_GE(stats.num_active_mappings, 3);
}

// Test prefetch hint
TEST_F(MMapLoaderTest, PrefetchHint) {
  MMapWeightLoader loader(test_file_path_);
  ASSERT_TRUE(loader.initialize());

  // Map with prefetch - should not fail
  auto region = loader.map_region(0, 4096, true);
  ASSERT_TRUE(region.is_valid);

  // Data should still be accessible
  char* data = region.as<char>();
  EXPECT_EQ(data[0], test_data_[0]);
}

}  // namespace
