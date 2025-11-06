// Copyright © 2025 MLXR Development
// Speculative decoding with draft model proposer and verification

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace mlxr {

// Forward declarations
class LlamaModel;
class Tokenizer;

namespace spec {

// Configuration for speculative decoding
struct SpeculativeConfig {
  // Number of tokens to propose per step
  int num_draft_tokens = 4;

  // Minimum acceptance rate before disabling speculation
  float min_acceptance_rate = 0.5f;

  // Window size for acceptance rate calculation
  int acceptance_window = 100;

  // Enable adaptive speculation length
  bool adaptive_length = true;

  // Temperature for draft model sampling
  float draft_temperature = 1.0f;

  // Temperature for target model verification
  float target_temperature = 1.0f;

  // Enable/disable speculation
  bool enabled = true;
};

// Result of a speculation attempt
struct SpeculationResult {
  // Draft tokens that were proposed
  std::vector<int> draft_tokens;

  // Tokens accepted by target model
  std::vector<int> accepted_tokens;

  // Number of tokens accepted
  int num_accepted;

  // Final token chosen (bonus token from verification)
  std::optional<int> bonus_token;

  // Time spent in draft model (ms)
  double draft_time_ms;

  // Time spent in target model (ms)
  double target_time_ms;

  // Acceptance rate for this attempt
  float acceptance_rate() const {
    if (draft_tokens.empty()) return 0.0f;
    return static_cast<float>(num_accepted) / draft_tokens.size();
  }

  // Total tokens generated (accepted + bonus)
  int total_tokens() const {
    return num_accepted + (bonus_token.has_value() ? 1 : 0);
  }

  // Speedup factor (tokens per draft+verify cycle)
  float speedup() const { return static_cast<float>(total_tokens()); }
};

// Statistics for speculative decoding
struct SpeculativeStats {
  // Total speculation attempts
  uint64_t total_attempts = 0;

  // Total tokens proposed
  uint64_t total_proposed = 0;

  // Total tokens accepted
  uint64_t total_accepted = 0;

  // Total bonus tokens
  uint64_t total_bonus = 0;

  // Overall acceptance rate
  float overall_acceptance_rate() const {
    if (total_proposed == 0) return 0.0f;
    return static_cast<float>(total_accepted) / total_proposed;
  }

  // Average speedup
  float average_speedup() const {
    if (total_attempts == 0) return 1.0f;
    return static_cast<float>(total_accepted + total_bonus) / total_attempts;
  }

  // Tokens per second improvement
  float tokens_per_attempt() const {
    if (total_attempts == 0) return 0.0f;
    return static_cast<float>(total_accepted + total_bonus) / total_attempts;
  }
};

/**
 * Speculative Decoder
 *
 * Uses a smaller draft model to propose k tokens quickly, then verifies
 * them with the target model in parallel. Accepts tokens that match and
 * can generate a bonus token when all are accepted.
 *
 * Algorithm:
 * 1. Draft model proposes k tokens autoregressively
 * 2. Target model processes all k tokens + context in single forward pass
 * 3. Compare draft tokens with target model's predictions
 * 4. Accept tokens until first mismatch
 * 5. If all k accepted, sample bonus token from target model
 * 6. Update KV cache with accepted tokens only
 */
class SpeculativeDecoder {
 public:
  /**
   * Create speculative decoder
   * @param target_model Main model for generation
   * @param draft_model Smaller/faster draft model
   * @param config Speculation configuration
   */
  SpeculativeDecoder(std::shared_ptr<LlamaModel> target_model,
                     std::shared_ptr<LlamaModel> draft_model,
                     const SpeculativeConfig& config);

  ~SpeculativeDecoder();

  /**
   * Perform one speculation step
   * @param context_tokens Current context (prompt + generated tokens)
   * @param max_new_tokens Maximum tokens to generate
   * @return Speculation result with accepted tokens
   */
  SpeculationResult speculate(const std::vector<int>& context_tokens,
                              int max_new_tokens);

  /**
   * Generate tokens with speculative decoding
   * @param prompt_tokens Input prompt
   * @param max_tokens Maximum tokens to generate
   * @param callback Called for each accepted token
   * @return All generated tokens
   */
  std::vector<int> generate(const std::vector<int>& prompt_tokens,
                            int max_tokens,
                            std::function<void(int)> callback = nullptr);

  /**
   * Reset KV cache and internal state
   */
  void reset();

  /**
   * Get current statistics
   */
  SpeculativeStats get_stats() const;

  /**
   * Get current acceptance rate (rolling window)
   */
  float get_current_acceptance_rate() const;

  /**
   * Update configuration
   */
  void update_config(const SpeculativeConfig& config);

  /**
   * Enable/disable speculation
   */
  void set_enabled(bool enabled);

  /**
   * Check if speculation is currently enabled
   */
  bool is_enabled() const { return config_.enabled; }

 private:
  // Models
  std::shared_ptr<LlamaModel> target_model_;
  std::shared_ptr<LlamaModel> draft_model_;

  // Configuration
  SpeculativeConfig config_;

  // Statistics
  SpeculativeStats stats_;

  // Rolling window for acceptance rate
  std::vector<float> acceptance_history_;
  size_t history_index_;

  // Adaptive speculation length tracking
  int current_draft_length_;

  /**
   * Propose tokens using draft model
   * @param context Current context
   * @param num_tokens Number of tokens to propose
   * @return Draft tokens
   */
  std::vector<int> propose_tokens(const std::vector<int>& context,
                                  int num_tokens);

  /**
   * Verify draft tokens with target model
   * @param context Current context
   * @param draft_tokens Proposed tokens
   * @return Number of accepted tokens and optional bonus token
   */
  std::pair<int, std::optional<int>> verify_tokens(
      const std::vector<int>& context, const std::vector<int>& draft_tokens);

  /**
   * Sample token from logits with temperature
   * @param logits Output logits from model
   * @param temperature Sampling temperature
   * @return Sampled token ID
   */
  int sample_token(const std::vector<float>& logits, float temperature);

  /**
   * Update acceptance rate tracking
   */
  void update_acceptance_tracking(float rate);

  /**
   * Adjust speculation length based on acceptance rate
   */
  void adjust_speculation_length();

  /**
   * Check if speculation should be disabled due to low acceptance
   */
  bool should_disable_speculation() const;
};

}  // namespace spec
}  // namespace mlxr
