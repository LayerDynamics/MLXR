// Copyright © 2025 MLXR Development
// Speculative decoder implementation

#include "speculative_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

#include "graph/model.h"

namespace mlxr {
namespace spec {

SpeculativeDecoder::SpeculativeDecoder(std::shared_ptr<LlamaModel> target_model,
                                       std::shared_ptr<LlamaModel> draft_model,
                                       const SpeculativeConfig& config)
    : target_model_(target_model),
      draft_model_(draft_model),
      config_(config),
      history_index_(0),
      current_draft_length_(config.num_draft_tokens) {
  // Initialize acceptance history with optimistic values
  acceptance_history_.resize(config.acceptance_window, 1.0f);
}

SpeculativeDecoder::~SpeculativeDecoder() = default;

SpeculationResult SpeculativeDecoder::speculate(
    const std::vector<int>& context_tokens, int max_new_tokens) {
  SpeculationResult result;

  if (!config_.enabled) {
    // Speculation disabled, fall back to regular decoding
    return result;
  }

  // Determine how many tokens to propose
  int num_to_propose = std::min(current_draft_length_, max_new_tokens);
  if (num_to_propose <= 0) {
    return result;
  }

  auto start_draft = std::chrono::steady_clock::now();

  // Step 1: Draft model proposes tokens
  result.draft_tokens = propose_tokens(context_tokens, num_to_propose);

  auto end_draft = std::chrono::steady_clock::now();
  result.draft_time_ms =
      std::chrono::duration<double, std::milli>(end_draft - start_draft)
          .count();

  if (result.draft_tokens.empty()) {
    return result;
  }

  auto start_verify = std::chrono::steady_clock::now();

  // Step 2: Target model verifies draft tokens
  auto [num_accepted, bonus_token] =
      verify_tokens(context_tokens, result.draft_tokens);

  auto end_verify = std::chrono::steady_clock::now();
  result.target_time_ms =
      std::chrono::duration<double, std::milli>(end_verify - start_verify)
          .count();

  result.num_accepted = num_accepted;
  result.bonus_token = bonus_token;

  // Extract accepted tokens
  result.accepted_tokens.assign(result.draft_tokens.begin(),
                                result.draft_tokens.begin() + num_accepted);

  // Update statistics
  stats_.total_attempts++;
  stats_.total_proposed += result.draft_tokens.size();
  stats_.total_accepted += num_accepted;
  if (bonus_token.has_value()) {
    stats_.total_bonus++;
  }

  // Update acceptance tracking
  update_acceptance_tracking(result.acceptance_rate());

  // Adapt speculation length if enabled
  if (config_.adaptive_length) {
    adjust_speculation_length();
  }

  return result;
}

std::vector<int> SpeculativeDecoder::generate(
    const std::vector<int>& prompt_tokens, int max_tokens,
    std::function<void(int)> callback) {
  std::vector<int> generated_tokens;
  std::vector<int> context = prompt_tokens;

  while (static_cast<int>(generated_tokens.size()) < max_tokens) {
    int remaining = max_tokens - static_cast<int>(generated_tokens.size());

    if (config_.enabled && !should_disable_speculation()) {
      // Use speculative decoding
      SpeculationResult result = speculate(context, remaining);

      if (result.total_tokens() > 0) {
        // Add accepted tokens
        for (int token : result.accepted_tokens) {
          generated_tokens.push_back(token);
          context.push_back(token);
          if (callback) {
            callback(token);
          }
        }

        // Add bonus token if present
        if (result.bonus_token.has_value()) {
          int bonus = result.bonus_token.value();
          generated_tokens.push_back(bonus);
          context.push_back(bonus);
          if (callback) {
            callback(bonus);
          }
        }
      } else {
        // Speculation failed, fall back to regular decoding
        // This is a placeholder - in real implementation would call target
        // model
        break;
      }
    } else {
      // Regular decoding (no speculation)
      // Placeholder - would generate one token at a time
      break;
    }
  }

  return generated_tokens;
}

void SpeculativeDecoder::reset() {
  stats_ = SpeculativeStats();
  acceptance_history_.clear();
  acceptance_history_.resize(config_.acceptance_window, 1.0f);
  history_index_ = 0;
  current_draft_length_ = config_.num_draft_tokens;
}

SpeculativeStats SpeculativeDecoder::get_stats() const { return stats_; }

float SpeculativeDecoder::get_current_acceptance_rate() const {
  if (acceptance_history_.empty()) {
    return 1.0f;
  }

  float sum = std::accumulate(acceptance_history_.begin(),
                              acceptance_history_.end(), 0.0f);

  return sum / acceptance_history_.size();
}

void SpeculativeDecoder::update_config(const SpeculativeConfig& config) {
  config_ = config;

  // Resize acceptance history if window size changed
  if (static_cast<int>(acceptance_history_.size()) !=
      config.acceptance_window) {
    acceptance_history_.resize(config.acceptance_window, 1.0f);
    history_index_ = 0;
  }

  current_draft_length_ = config.num_draft_tokens;
}

void SpeculativeDecoder::set_enabled(bool enabled) {
  config_.enabled = enabled;
}

// Private methods

std::vector<int> SpeculativeDecoder::propose_tokens(
    const std::vector<int>& context, int num_tokens) {
  std::vector<int> draft_tokens;
  draft_tokens.reserve(num_tokens);

  std::vector<int> current_context = context;

  // Autoregressively generate draft tokens
  for (int i = 0; i < num_tokens; i++) {
    // Forward pass through draft model
    // Placeholder: In real implementation, would call draft_model_->forward()
    // For now, just generate dummy tokens

    // Get logits from draft model (placeholder)
    std::vector<float> logits(32000, 0.0f);

    // Sample token
    int token = sample_token(logits, config_.draft_temperature);

    draft_tokens.push_back(token);
    current_context.push_back(token);
  }

  return draft_tokens;
}

std::pair<int, std::optional<int>> SpeculativeDecoder::verify_tokens(
    const std::vector<int>& context, const std::vector<int>& draft_tokens) {
  if (draft_tokens.empty()) {
    return {0, std::nullopt};
  }

  // Build verification context: original context + all draft tokens
  std::vector<int> verify_context = context;
  verify_context.insert(verify_context.end(), draft_tokens.begin(),
                        draft_tokens.end());

  // Forward pass through target model with all draft tokens
  // Placeholder: In real implementation, would call target_model_->forward()
  // This would return logits for each position

  // For now, simulate verification with random acceptance
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  int num_accepted = 0;
  for (size_t i = 0; i < draft_tokens.size(); i++) {
    // Get target model's prediction at this position
    // Placeholder: would compare draft_tokens[i] with target prediction

    // Simulate acceptance with decreasing probability
    float accept_prob = 0.7f * std::pow(0.9f, static_cast<float>(i));
    if (dis(gen) < accept_prob) {
      num_accepted++;
    } else {
      // Mismatch found, stop accepting
      break;
    }
  }

  // If all tokens accepted, try to generate bonus token
  std::optional<int> bonus_token;
  if (num_accepted == static_cast<int>(draft_tokens.size())) {
    // Sample bonus token from target model's logits at the end
    std::vector<float> logits(32000, 0.0f);
    int bonus = sample_token(logits, config_.target_temperature);
    bonus_token = bonus;
  }

  return {num_accepted, bonus_token};
}

int SpeculativeDecoder::sample_token(const std::vector<float>& logits,
                                     float temperature) {
  if (logits.empty()) {
    return 0;
  }

  // Apply temperature
  std::vector<float> scaled_logits = logits;
  if (temperature != 1.0f && temperature > 0.0f) {
    for (float& logit : scaled_logits) {
      logit /= temperature;
    }
  }

  // Compute softmax
  float max_logit =
      *std::max_element(scaled_logits.begin(), scaled_logits.end());
  std::vector<float> probs(scaled_logits.size());
  float sum = 0.0f;

  for (size_t i = 0; i < scaled_logits.size(); i++) {
    probs[i] = std::exp(scaled_logits[i] - max_logit);
    sum += probs[i];
  }

  for (float& prob : probs) {
    prob /= sum;
  }

  // Sample from distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  float r = dis(gen);
  float cumsum = 0.0f;

  for (size_t i = 0; i < probs.size(); i++) {
    cumsum += probs[i];
    if (r < cumsum) {
      return static_cast<int>(i);
    }
  }

  // Fallback
  return static_cast<int>(probs.size() - 1);
}

void SpeculativeDecoder::update_acceptance_tracking(float rate) {
  acceptance_history_[history_index_] = rate;
  history_index_ = (history_index_ + 1) % acceptance_history_.size();
}

void SpeculativeDecoder::adjust_speculation_length() {
  float current_rate = get_current_acceptance_rate();

  // Increase draft length if acceptance rate is high
  if (current_rate > 0.8f && current_draft_length_ < config_.num_draft_tokens) {
    current_draft_length_++;
  }
  // Decrease draft length if acceptance rate is low
  else if (current_rate < 0.5f && current_draft_length_ > 1) {
    current_draft_length_--;
  }
}

bool SpeculativeDecoder::should_disable_speculation() const {
  float current_rate = get_current_acceptance_rate();
  return current_rate < config_.min_acceptance_rate;
}

}  // namespace spec
}  // namespace mlxr
