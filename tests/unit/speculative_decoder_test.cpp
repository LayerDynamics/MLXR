// Copyright © 2025 MLXR Development
// Speculative decoder unit tests

#include "runtime/spec/speculative_decoder.h"

#include <gtest/gtest.h>

using namespace mlxr::spec;

namespace {

// Note: Full integration tests with real models are in integration tests
// These unit tests focus on the speculative decoding logic and statistics

class SpeculativeDecoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Note: Models are nullptr for these unit tests
    // We're testing the config, stats, and control flow logic
    target_model_ = nullptr;
    draft_model_ = nullptr;

    // Default config
    config_.num_draft_tokens = 4;
    config_.min_acceptance_rate = 0.5f;
    config_.acceptance_window = 10;
    config_.adaptive_length = true;
    config_.enabled = true;

    // Note: Cannot create actual decoder without models
    // Tests will focus on data structures and config
  }

  void TearDown() override {
    // Nothing to clean up
  }

  std::shared_ptr<mlxr::LlamaModel> target_model_;
  std::shared_ptr<mlxr::LlamaModel> draft_model_;
  SpeculativeConfig config_;
};

// Test configuration structure
TEST_F(SpeculativeDecoderTest, ConfigStructure) {
  EXPECT_EQ(config_.num_draft_tokens, 4);
  EXPECT_FLOAT_EQ(config_.min_acceptance_rate, 0.5f);
  EXPECT_EQ(config_.acceptance_window, 10);
  EXPECT_TRUE(config_.adaptive_length);
  EXPECT_TRUE(config_.enabled);
}

// Test configuration defaults
TEST_F(SpeculativeDecoderTest, ConfigDefaults) {
  SpeculativeConfig default_config;

  EXPECT_EQ(default_config.num_draft_tokens, 4);
  EXPECT_FLOAT_EQ(default_config.min_acceptance_rate, 0.5f);
  EXPECT_EQ(default_config.acceptance_window, 100);
  EXPECT_TRUE(default_config.adaptive_length);
  EXPECT_FLOAT_EQ(default_config.draft_temperature, 1.0f);
  EXPECT_FLOAT_EQ(default_config.target_temperature, 1.0f);
  EXPECT_TRUE(default_config.enabled);
}

// Test result methods
TEST_F(SpeculativeDecoderTest, ResultMethods) {
  SpeculationResult result;
  result.draft_tokens = {1, 2, 3, 4};
  result.accepted_tokens = {1, 2, 3};
  result.num_accepted = 3;
  result.bonus_token = 5;

  EXPECT_FLOAT_EQ(result.acceptance_rate(), 0.75f);
  EXPECT_EQ(result.total_tokens(), 4);  // 3 accepted + 1 bonus
  EXPECT_FLOAT_EQ(result.speedup(), 4.0f);
}

// Test result methods without bonus
TEST_F(SpeculativeDecoderTest, ResultMethodsNoBonus) {
  SpeculationResult result;
  result.draft_tokens = {1, 2, 3, 4};
  result.accepted_tokens = {1, 2};
  result.num_accepted = 2;
  result.bonus_token = std::nullopt;

  EXPECT_FLOAT_EQ(result.acceptance_rate(), 0.5f);
  EXPECT_EQ(result.total_tokens(), 2);  // 2 accepted, no bonus
  EXPECT_FLOAT_EQ(result.speedup(), 2.0f);
}

// Test stats methods
TEST_F(SpeculativeDecoderTest, StatsMethods) {
  SpeculativeStats stats;
  stats.total_attempts = 10;
  stats.total_proposed = 40;
  stats.total_accepted = 30;
  stats.total_bonus = 5;

  EXPECT_FLOAT_EQ(stats.overall_acceptance_rate(), 0.75f);
  EXPECT_FLOAT_EQ(stats.average_speedup(), 3.5f);  // (30 + 5) / 10
  EXPECT_FLOAT_EQ(stats.tokens_per_attempt(), 3.5f);
}

}  // namespace
