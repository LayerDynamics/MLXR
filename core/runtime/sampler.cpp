/**
 * @file sampler.cpp
 * @brief Implementation of sampling strategies for text generation
 */

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "mlx/mlx.h"

namespace mlxr {
namespace runtime {

Sampler::Sampler(const SamplerConfig& config) : config_(config) {
  // Initialize random number generator
  if (config_.seed == 0) {
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(config_.seed);
  }
}

int Sampler::sample(const graph::Tensor& logits,
                    const std::vector<int>& prev_tokens) {
  // Apply repetition penalty if needed
  graph::Tensor modified_logits = logits;
  if (config_.repetition_penalty != 1.0f && !prev_tokens.empty()) {
    modified_logits = apply_repetition_penalty(logits, prev_tokens,
                                               config_.repetition_penalty);
  }

  // Select sampling strategy based on config
  if (config_.temperature == 0.0f ||
      (config_.top_k == 0 && config_.top_p == 0.0f)) {
    // Greedy sampling
    return sample_greedy(modified_logits);
  } else if (config_.top_k > 0 && config_.top_p > 0.0f) {
    // Combined top-k and top-p
    // First apply top-k, then top-p
    auto temp_logits = modified_logits;
    return sample_top_p(temp_logits, config_.top_p, config_.temperature, rng_);
  } else if (config_.top_k > 0) {
    // Top-k sampling
    return sample_top_k(modified_logits, config_.top_k, config_.temperature,
                        rng_);
  } else if (config_.top_p > 0.0f) {
    // Top-p sampling
    return sample_top_p(modified_logits, config_.top_p, config_.temperature,
                        rng_);
  } else {
    // Pure temperature sampling
    return sample_temperature(modified_logits, config_.temperature, rng_);
  }
}

int Sampler::sample_greedy(const graph::Tensor& logits) {
  // Find argmax
  auto logits_arr = logits.array();

  // Evaluate the array to ensure data is available
  mlx::core::eval(logits_arr);

  auto shape = logits.shape();
  if (shape.size() != 1) {
    throw std::invalid_argument("sample_greedy expects 1D logits tensor");
  }

  (void)shape[0];  // vocab_size - suppress unused warning

  // Use MLX argmax
  auto argmax_arr =
      mlx::core::argmax(logits_arr, /*axis=*/-1, /*keepdims=*/false);
  mlx::core::eval(argmax_arr);

  // Extract the scalar value
  int token_id = argmax_arr.item<int>();

  return token_id;
}

graph::Tensor Sampler::logits_to_probs(const graph::Tensor& logits,
                                       float temperature) {
  auto logits_arr = logits.array();

  // Apply temperature
  if (temperature != 1.0f && temperature > 0.0f) {
    logits_arr = mlx::core::divide(logits_arr, mlx::core::array(temperature));
  }

  // Apply softmax
  auto probs_arr = mlx::core::softmax(logits_arr, /*axis=*/-1);

  return graph::Tensor(probs_arr);
}

int Sampler::sample_categorical(const std::vector<float>& probs,
                                std::mt19937& rng) {
  // Sample from categorical distribution
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float rand_val = dist(rng);

  float cumsum = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    cumsum += probs[i];
    if (rand_val < cumsum) {
      return static_cast<int>(i);
    }
  }

  // Fallback to last token (shouldn't happen with proper probabilities)
  return static_cast<int>(probs.size() - 1);
}

int Sampler::sample_temperature(const graph::Tensor& logits, float temperature,
                                std::mt19937& rng) {
  // Convert to probabilities
  auto probs = logits_to_probs(logits, temperature);
  auto probs_arr = probs.array();
  mlx::core::eval(probs_arr);

  // Convert to vector for sampling
  auto shape = probs.shape();
  int vocab_size = shape[0];

  std::vector<float> probs_vec(vocab_size);
  // Copy data from MLX array to vector
  const float* probs_data = probs_arr.data<float>();
  std::copy(probs_data, probs_data + vocab_size, probs_vec.begin());

  return sample_categorical(probs_vec, rng);
}

int Sampler::sample_top_k(const graph::Tensor& logits, int k, float temperature,
                          std::mt19937& rng) {
  auto logits_arr = logits.array();
  mlx::core::eval(logits_arr);

  auto shape = logits.shape();
  int vocab_size = shape[0];

  // Get top-k indices
  // MLX doesn't have a direct top-k, so we'll use argsort
  auto sorted_indices = mlx::core::argsort(logits_arr, /*axis=*/-1);
  mlx::core::eval(sorted_indices);

  // Take last k elements (highest values) since argsort is ascending
  int actual_k = std::min(k, vocab_size);
  auto top_k_indices = mlx::core::slice(sorted_indices, {vocab_size - actual_k},
                                        {vocab_size}, {1});
  mlx::core::eval(top_k_indices);

  // Get logits for top-k tokens
  auto top_k_logits = mlx::core::take(logits_arr, top_k_indices, 0);
  mlx::core::eval(top_k_logits);

  // Convert to probabilities with temperature
  graph::Tensor top_k_logits_tensor(top_k_logits);
  auto probs = logits_to_probs(top_k_logits_tensor, temperature);
  auto probs_arr = probs.array();
  mlx::core::eval(probs_arr);

  // Convert to vector
  std::vector<float> probs_vec(actual_k);
  const float* probs_data = probs_arr.data<float>();
  std::copy(probs_data, probs_data + actual_k, probs_vec.begin());

  // Sample from top-k
  int sampled_idx = sample_categorical(probs_vec, rng);

  // Map back to original vocabulary index
  const int* indices_data = top_k_indices.data<int>();
  return indices_data[sampled_idx];
}

int Sampler::sample_top_p(const graph::Tensor& logits, float p,
                          float temperature, std::mt19937& rng) {
  auto logits_arr = logits.array();
  mlx::core::eval(logits_arr);

  auto shape = logits.shape();
  int vocab_size = shape[0];

  // Convert to probabilities
  auto probs = logits_to_probs(logits, temperature);
  auto probs_arr = probs.array();
  mlx::core::eval(probs_arr);

  // Sort probabilities in descending order
  auto sorted_indices = mlx::core::argsort(probs_arr, /*axis=*/-1);
  mlx::core::eval(sorted_indices);

  // Get sorted probabilities (in ascending order, need to reverse)
  auto sorted_probs = mlx::core::take(probs_arr, sorted_indices, 0);
  mlx::core::eval(sorted_probs);

  // Copy to vectors for easier manipulation
  std::vector<float> probs_vec(vocab_size);
  std::vector<int> indices_vec(vocab_size);
  const float* probs_data = sorted_probs.data<float>();
  const int* indices_data = sorted_indices.data<int>();
  std::copy(probs_data, probs_data + vocab_size, probs_vec.begin());
  std::copy(indices_data, indices_data + vocab_size, indices_vec.begin());

  // Reverse to get descending order
  std::reverse(probs_vec.begin(), probs_vec.end());
  std::reverse(indices_vec.begin(), indices_vec.end());

  // Find nucleus (tokens with cumulative probability <= p)
  float cumsum = 0.0f;
  int nucleus_size = 0;
  for (int i = 0; i < vocab_size; ++i) {
    cumsum += probs_vec[i];
    nucleus_size++;
    if (cumsum >= p) {
      break;
    }
  }

  // Renormalize probabilities in nucleus
  std::vector<float> nucleus_probs(nucleus_size);
  float nucleus_sum = 0.0f;
  for (int i = 0; i < nucleus_size; ++i) {
    nucleus_probs[i] = probs_vec[i];
    nucleus_sum += probs_vec[i];
  }

  // Normalize
  for (int i = 0; i < nucleus_size; ++i) {
    nucleus_probs[i] /= nucleus_sum;
  }

  // Sample from nucleus
  int sampled_idx = sample_categorical(nucleus_probs, rng);

  return indices_vec[sampled_idx];
}

graph::Tensor Sampler::apply_repetition_penalty(
    const graph::Tensor& logits, const std::vector<int>& prev_tokens,
    float penalty) {
  if (penalty == 1.0f || prev_tokens.empty()) {
    return logits;
  }

  auto logits_arr = logits.array();
  mlx::core::eval(logits_arr);

  auto shape = logits.shape();
  int vocab_size = shape[0];

  // Copy logits to vector for modification
  std::vector<float> logits_vec(vocab_size);
  const float* logits_data = logits_arr.data<float>();
  std::copy(logits_data, logits_data + vocab_size, logits_vec.begin());

  // Apply penalty to previously seen tokens
  for (int token : prev_tokens) {
    if (token >= 0 && token < vocab_size) {
      // If logit is positive, divide by penalty; if negative, multiply by
      // penalty
      if (logits_vec[token] > 0.0f) {
        logits_vec[token] /= penalty;
      } else {
        logits_vec[token] *= penalty;
      }
    }
  }

  // Convert back to MLX array
  auto modified_arr =
      mlx::core::array(logits_vec.begin(), {vocab_size}, mlx::core::float32);

  return graph::Tensor(modified_arr);
}

}  // namespace runtime
}  // namespace mlxr
