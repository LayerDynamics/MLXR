// Copyright © 2025 MLXR Development
// Unit tests for ModelLoader pager functionality

#include <gtest/gtest.h>

#include <memory>

#include "runtime/kv/arena.h"
#include "runtime/kv/pager.h"
#include "registry/model_registry.h"
#include "server/model_loader.h"

namespace mlxr {
namespace server {
namespace test {

class ModelLoaderPagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create in-memory registry
    registry_ = std::make_shared<registry::ModelRegistry>(":memory:");
    ASSERT_TRUE(registry_->initialize());

    // Create model loader
    loader_ = std::make_unique<ModelLoader>(registry_);
  }

  std::shared_ptr<registry::ModelRegistry> registry_;
  std::unique_ptr<ModelLoader> loader_;
};

// Test pager creation with default config
TEST_F(ModelLoaderPagerTest, CreatePagerDefault) {
  LoadModelConfig config;
  config.kv_num_layers = 22;
  config.kv_num_heads = 4;
  config.kv_head_dim = 64;
  config.kv_block_size = 32;
  config.kv_num_blocks = 256;

  // We can't directly call create_pager() since it's private,
  // but we can test the arena and pager independently
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = config.kv_num_layers;
  arena_config.num_kv_heads = config.kv_num_heads;
  arena_config.head_dim = config.kv_head_dim;
  arena_config.block_size_tokens = config.kv_block_size;
  arena_config.num_blocks = config.kv_num_blocks;

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  ASSERT_NE(arena, nullptr);

  // Check arena stats
  auto stats = arena->get_stats();
  EXPECT_EQ(stats.total_blocks, config.kv_num_blocks);
  EXPECT_EQ(stats.free_blocks, config.kv_num_blocks);
  EXPECT_EQ(stats.allocated_blocks, 0);

  // Create pager
  auto pager = std::make_shared<runtime::kv::Pager>(arena);
  ASSERT_NE(pager, nullptr);

  // Check pager stats
  auto pager_stats = pager->get_stats();
  EXPECT_EQ(pager_stats.num_sequences, 0);
  EXPECT_EQ(pager_stats.num_active_sequences, 0);
}

// Test arena block allocation
TEST_F(ModelLoaderPagerTest, ArenaBlockAllocation) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 10;  // Small for testing

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);

  // Allocate a block
  auto block_id = arena->allocate_block();
  ASSERT_GE(block_id, 0);

  auto stats = arena->get_stats();
  EXPECT_EQ(stats.allocated_blocks, 1);
  EXPECT_EQ(stats.free_blocks, 9);

  // Free the block
  arena->free_block(block_id);

  stats = arena->get_stats();
  EXPECT_EQ(stats.allocated_blocks, 0);
  EXPECT_EQ(stats.free_blocks, 10);
}

// Test sequence creation and block allocation
TEST_F(ModelLoaderPagerTest, SequenceCreation) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 100;

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  // Create sequence
  int seq_id = 0;
  ASSERT_TRUE(pager->create_sequence(seq_id));

  auto stats = pager->get_stats();
  EXPECT_EQ(stats.num_sequences, 1);
  EXPECT_EQ(stats.num_active_sequences, 1);

  // Get sequence
  auto* seq = pager->get_sequence(seq_id);
  ASSERT_NE(seq, nullptr);
  EXPECT_EQ(seq->id(), seq_id);
  EXPECT_EQ(seq->num_tokens(), 0);
  EXPECT_EQ(seq->block_size(), 32);
}

// Test block allocation for sequence
TEST_F(ModelLoaderPagerTest, SequenceBlockAllocation) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 100;

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  // Create sequence
  int seq_id = 0;
  pager->create_sequence(seq_id);

  // Allocate blocks for 64 tokens (should need 2 blocks)
  ASSERT_TRUE(pager->allocate_blocks_for_sequence(seq_id, 64));

  auto* seq = pager->get_sequence(seq_id);
  ASSERT_NE(seq, nullptr);

  auto page_table = seq->page_table();
  EXPECT_EQ(page_table.size(), 2);  // 64 tokens / 32 tokens per block = 2 blocks

  // Check arena stats
  auto arena_stats = arena->get_stats();
  EXPECT_EQ(arena_stats.allocated_blocks, 2);
}

// Test multiple sequences
TEST_F(ModelLoaderPagerTest, MultipleSequences) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 100;

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  // Create multiple sequences
  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(pager->create_sequence(i));
    ASSERT_TRUE(pager->allocate_blocks_for_sequence(i, 32 * (i + 1)));
  }

  auto stats = pager->get_stats();
  EXPECT_EQ(stats.num_sequences, 5);
  EXPECT_EQ(stats.num_active_sequences, 5);

  // Total blocks: 1 + 2 + 3 + 4 + 5 = 15 blocks
  auto arena_stats = arena->get_stats();
  EXPECT_EQ(arena_stats.allocated_blocks, 15);
}

// Test sequence deletion
TEST_F(ModelLoaderPagerTest, SequenceDeletion) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 100;

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  // Create and allocate
  pager->create_sequence(0);
  pager->allocate_blocks_for_sequence(0, 64);

  auto arena_stats_before = arena->get_stats();
  EXPECT_EQ(arena_stats_before.allocated_blocks, 2);

  // Delete sequence
  pager->delete_sequence(0);

  auto stats = pager->get_stats();
  EXPECT_EQ(stats.num_sequences, 0);

  // Blocks should be freed
  auto arena_stats_after = arena->get_stats();
  EXPECT_EQ(arena_stats_after.allocated_blocks, 0);
  EXPECT_EQ(arena_stats_after.free_blocks, 100);
}

// Test capacity limits
TEST_F(ModelLoaderPagerTest, CapacityLimits) {
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = 22;
  arena_config.num_kv_heads = 4;
  arena_config.head_dim = 64;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 3;  // Only 3 blocks

  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  // Create sequence
  pager->create_sequence(0);

  // Try to allocate 4 blocks (should fail, only 3 available)
  EXPECT_FALSE(pager->allocate_blocks_for_sequence(0, 128));  // 128 tokens = 4 blocks

  // Allocate 3 blocks (should succeed)
  EXPECT_TRUE(pager->allocate_blocks_for_sequence(0, 96));  // 96 tokens = 3 blocks
}

}  // namespace test
}  // namespace server
}  // namespace mlxr

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
