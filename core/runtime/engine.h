/**
 * @file engine.h
 * @brief Inference engine for text generation
 *
 * Provides a high-level interface for running inference with LLM models.
 * Handles tokenization, model forward pass, sampling, and generation loop.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../graph/model.h"
#include "kv/pager.h"
#include "sampler.h"
#include "tokenizer/tokenizer.h"

namespace mlxr {
namespace runtime {

/**
 * @brief KV cache for incremental inference
 *
 * Stores key-value tensors for each transformer layer to enable
 * efficient autoregressive generation without recomputing past tokens.
 */
struct InferenceCache {
  // Model-level KV cache (contains per-layer K,V tensors)
  // Used by simple LlamaModel
  graph::KVCache kv_cache;

  // Sequence ID for paged cache
  // Used by CachedLlamaModel with Pager
  int seq_id = 0;

  // Number of tokens currently cached
  int cached_tokens = 0;

  // Whether cache has been initialized
  bool initialized = false;

  // Clear the cache
  void clear() {
    kv_cache.clear();
    cached_tokens = 0;
    initialized = false;
  }
};

/**
 * @brief Configuration for generation
 */
struct GenerationConfig {
  // Maximum number of tokens to generate
  int max_new_tokens = 128;

  // Maximum total sequence length (prompt + generated)
  int max_seq_len = 2048;

  // Sampling configuration
  SamplerConfig sampler_config;

  // Stop tokens (generation stops when any of these is generated)
  std::vector<int> stop_tokens;

  // Whether to echo the prompt in the output
  bool echo_prompt = false;

  // Whether to print generation progress
  bool verbose = false;

  // Whether to use CachedLlamaModel with paged KV cache and Metal kernels
  // If true, Engine will use CachedLlamaModel (zero-copy optimization)
  // If false, Engine will use simple LlamaModel (basic concatenation)
  bool use_cached_attention = true;  // Enable by default for best performance

  // KV cache block size (for paged cache, only used if
  // use_cached_attention=true)
  int kv_block_size = 32;  // 32 tokens per block

  // Number of KV cache blocks (for paged cache, only used if
  // use_cached_attention=true)
  int kv_num_blocks = 256;  // Total capacity: 32 * 256 = 8192 tokens
};

/**
 * @brief Inference engine for text generation
 *
 * Main interface for running inference with LLM models.
 * Combines model, tokenizer, and sampler into a unified generation pipeline.
 */
class Engine {
 public:
  /**
   * @brief Construct engine with simple LlamaModel
   * @param model LLM model for generation
   * @param tokenizer Tokenizer for encoding/decoding text
   * @param config Generation configuration
   */
  Engine(std::shared_ptr<graph::LlamaModel> model,
         std::shared_ptr<Tokenizer> tokenizer,
         const GenerationConfig& config = GenerationConfig());

  /**
   * @brief Construct engine with CachedLlamaModel (zero-copy optimization)
   * @param model Cached LLM model with paged KV cache
   * @param pager KV cache pager
   * @param tokenizer Tokenizer for encoding/decoding text
   * @param config Generation configuration
   */
  Engine(std::shared_ptr<graph::CachedLlamaModel> model,
         std::shared_ptr<kv::Pager> pager, std::shared_ptr<Tokenizer> tokenizer,
         const GenerationConfig& config = GenerationConfig());

  /**
   * @brief Generate text from a prompt
   * @param prompt Input text prompt
   * @param config Optional generation config (overrides default)
   * @return Generated text
   */
  std::string generate(const std::string& prompt,
                       const GenerationConfig* config = nullptr);

  /**
   * @brief Generate tokens from input token IDs
   * @param input_ids Input token IDs
   * @param config Optional generation config (overrides default)
   * @return Generated token IDs (includes input if echo_prompt is true)
   */
  std::vector<int> generate_tokens(const std::vector<int>& input_ids,
                                   const GenerationConfig* config = nullptr);

  /**
   * @brief Prefill phase: Process prompt tokens and populate KV cache
   * @param input_ids Input token IDs (prompt tokens)
   * @param cache KV cache to populate (will be initialized if not already)
   * @return Logits for next token [vocab_size]
   */
  graph::Tensor forward_prefill(const std::vector<int>& input_ids,
                                InferenceCache* cache);

  /**
   * @brief Decode phase: Generate next token using existing KV cache
   * @param token_id Last generated token ID
   * @param cache Existing KV cache from prefill/previous decode steps
   * @return Logits for next token [vocab_size]
   */
  graph::Tensor forward_decode(int token_id, InferenceCache* cache);

  /**
   * @brief Encode text to token IDs
   * @param text Input text
   * @return Token IDs
   */
  std::vector<int> encode(const std::string& text);

  /**
   * @brief Decode token IDs to text
   * @param token_ids Token IDs
   * @return Decoded text
   */
  std::string decode(const std::vector<int>& token_ids);

  /**
   * @brief Check if using cached attention
   */
  bool is_using_cached_attention() const { return use_cached_; }

  /**
   * @brief Get tokenizer reference
   */
  Tokenizer& tokenizer() { return *tokenizer_; }

  /**
   * @brief Get generation config
   */
  const GenerationConfig& config() const { return config_; }

  /**
   * @brief Set generation config
   */
  void set_config(const GenerationConfig& config) { config_ = config; }

 private:
  /**
   * @brief Run single forward pass through model
   * @param input_ids Input token IDs [batch=1, seq_len]
   * @return Logits for next token [vocab_size]
   */
  graph::Tensor forward(const std::vector<int>& input_ids);

  /**
   * @brief Check if token is a stop token
   * @param token_id Token ID to check
   * @param stop_tokens List of stop token IDs
   * @return True if token is in stop list
   */
  static bool is_stop_token(int token_id, const std::vector<int>& stop_tokens);

  /**
   * @brief Allocate sequence in pager for inference cache
   * @param cache Inference cache to initialize with sequence ID
   * @param num_tokens Number of tokens to allocate blocks for
   */
  void allocate_cache_sequence(InferenceCache* cache, int num_tokens);

  /**
   * @brief Release sequence from pager
   * @param cache Inference cache containing sequence ID to release
   */
  void release_cache_sequence(InferenceCache* cache);

  // Simple model (basic concatenation)
  std::shared_ptr<graph::LlamaModel> simple_model_;

  // Cached model (paged KV cache + Metal kernels)
  std::shared_ptr<graph::CachedLlamaModel> cached_model_;

  // Pager for cached model (nullptr if using simple model)
  std::shared_ptr<kv::Pager> pager_;

  // Which model is active
  bool use_cached_;

  std::shared_ptr<Tokenizer> tokenizer_;
  GenerationConfig config_;

  // Next sequence ID to allocate
  int next_seq_id_ = 0;
};

/**
 * @brief Load engine from model directory
 * @param model_dir Directory containing model weights and config
 * @param tokenizer_path Path to tokenizer model
 * @param config Generation configuration
 * @return Initialized engine, or nullptr on failure
 */
std::unique_ptr<Engine> load_engine(
    const std::string& model_dir, const std::string& tokenizer_path,
    const GenerationConfig& config = GenerationConfig());

}  // namespace runtime
}  // namespace mlxr
