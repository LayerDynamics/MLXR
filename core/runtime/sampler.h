/**
 * @file sampler.h
 * @brief Sampling strategies for text generation
 *
 * Implements various decoding strategies:
 * - Greedy sampling (argmax)
 * - Temperature sampling
 * - Top-k sampling
 * - Top-p (nucleus) sampling
 * - Combined strategies
 */

#pragma once

#include <random>
#include <vector>

#include "../graph/tensor.h"

namespace mlxr {
namespace runtime {

/**
 * @brief Configuration for sampling strategies
 */
struct SamplerConfig {
  // Temperature for sampling (1.0 = no change, <1.0 = more conservative, >1.0 =
  // more random)
  float temperature = 1.0f;

  // Top-k: sample from top k tokens (0 = disabled)
  int top_k = 0;

  // Top-p (nucleus): sample from tokens with cumulative probability <= top_p
  // (0.0 = disabled)
  float top_p = 0.0f;

  // Random seed for reproducibility
  unsigned int seed = 0;

  // Minimum probability for a token to be considered (prevents numerical
  // issues)
  float min_p = 0.0f;

  // Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition)
  float repetition_penalty = 1.0f;
};

/**
 * @brief Sampler for text generation
 *
 * Handles various sampling strategies for selecting next tokens during
 * generation. Supports greedy, temperature, top-k, and top-p sampling.
 */
class Sampler {
 public:
  /**
   * @brief Construct sampler with given configuration
   * @param config Sampling configuration
   */
  explicit Sampler(const SamplerConfig& config = SamplerConfig());

  /**
   * @brief Sample next token from logits
   * @param logits Raw model output logits [vocab_size]
   * @param prev_tokens Previously generated tokens for repetition penalty
   * @return Sampled token ID
   */
  int sample(const graph::Tensor& logits,
             const std::vector<int>& prev_tokens = {});

  /**
   * @brief Greedy sampling (argmax)
   * @param logits Raw model output logits [vocab_size]
   * @return Token ID with highest probability
   */
  static int sample_greedy(const graph::Tensor& logits);

  /**
   * @brief Sample with temperature
   * @param logits Raw model output logits [vocab_size]
   * @param temperature Temperature parameter
   * @param rng Random number generator
   * @return Sampled token ID
   */
  static int sample_temperature(const graph::Tensor& logits, float temperature,
                                std::mt19937& rng);

  /**
   * @brief Top-k sampling
   * @param logits Raw model output logits [vocab_size]
   * @param k Number of top tokens to sample from
   * @param temperature Temperature parameter
   * @param rng Random number generator
   * @return Sampled token ID
   */
  static int sample_top_k(const graph::Tensor& logits, int k, float temperature,
                          std::mt19937& rng);

  /**
   * @brief Top-p (nucleus) sampling
   * @param logits Raw model output logits [vocab_size]
   * @param p Cumulative probability threshold
   * @param temperature Temperature parameter
   * @param rng Random number generator
   * @return Sampled token ID
   */
  static int sample_top_p(const graph::Tensor& logits, float p,
                          float temperature, std::mt19937& rng);

  /**
   * @brief Apply repetition penalty to logits
   * @param logits Raw model output logits [vocab_size]
   * @param prev_tokens Previously generated tokens
   * @param penalty Repetition penalty factor
   * @return Modified logits with penalty applied
   */
  static graph::Tensor apply_repetition_penalty(
      const graph::Tensor& logits, const std::vector<int>& prev_tokens,
      float penalty);

  /**
   * @brief Convert logits to probabilities using softmax
   * @param logits Raw logits
   * @param temperature Temperature to apply before softmax
   * @return Probabilities
   */
  static graph::Tensor logits_to_probs(const graph::Tensor& logits,
                                       float temperature = 1.0f);

  /**
   * @brief Sample from categorical distribution
   * @param probs Probability distribution [vocab_size]
   * @param rng Random number generator
   * @return Sampled index
   */
  static int sample_categorical(const std::vector<float>& probs,
                                std::mt19937& rng);

 private:
  SamplerConfig config_;
  std::mt19937 rng_;
};

}  // namespace runtime
}  // namespace mlxr
