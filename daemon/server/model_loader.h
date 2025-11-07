// Copyright © 2025 MLXR Development
// Model loading utility for REST/gRPC servers

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "../../core/graph/model.h"
#include "../../core/runtime/engine.h"
#include "../../core/runtime/kv/pager.h"
#include "../../core/runtime/mmap_loader.h"
#include "../../core/runtime/tokenizer/tokenizer.h"
#include "../registry/model_registry.h"

namespace mlxr {
namespace server {

/**
 * @brief Result of model loading operation
 */
struct LoadedModel {
  // Model components
  std::shared_ptr<graph::CachedLlamaModel> model;
  std::shared_ptr<runtime::kv::Pager> pager;
  std::shared_ptr<runtime::Tokenizer> tokenizer;
  std::shared_ptr<runtime::Engine> engine;

  // Model metadata
  registry::ModelInfo info;

  // Weight loader (keep alive for mmap)
  std::shared_ptr<MMapWeightLoader> loader;

  // Generation config used
  runtime::GenerationConfig config;
};

/**
 * @brief Configuration for model loading
 */
struct LoadModelConfig {
  // Whether to use cached attention (Metal kernels)
  bool use_cached_attention = true;

  // KV cache configuration
  int kv_block_size = 32;    // tokens per block
  int kv_num_blocks = 8192;  // total blocks (32 * 8192 = 262K tokens)
  int kv_num_layers = 32;    // number of model layers
  int kv_num_heads = 32;     // number of KV heads
  int kv_head_dim = 128;     // head dimension

  // Generation defaults
  int max_seq_len = 4096;
  int max_new_tokens = 2048;

  // Whether to prefetch weights into memory
  bool prefetch_weights = true;

  // Whether to lock weights in memory (prevent swapping)
  bool lock_weights = false;
};

/**
 * @brief Utility for loading models from registry
 *
 * **IMPORTANT: Current implementation is a compilation stub.**
 *
 * This class provides the interface for model loading but the actual
 * implementation (load_model) is not yet complete. It will:
 * - Successfully query the model registry
 * - Log found model information
 * - Return nullopt with "not yet fully implemented" error
 *
 * Full implementation requires proper API integration for:
 * - Tokenizer loading (SentencePieceTokenizer constructor)
 * - Arena/Pager creation with correct signatures
 * - Weight loading from GGUF/safetensors
 * - CachedLlamaModel instantiation with correct ModelConfig
 *
 * See docs/SESSION_2025_11_07_MODEL_LOADING.md for implementation plan.
 *
 * Handles the complete model loading pipeline:
 * 1. Query model metadata from registry
 * 2. Load weights via mmap
 * 3. Create model (LlamaModel or CachedLlamaModel)
 * 4. Initialize tokenizer
 * 5. Create pager (if using cached attention)
 * 6. Create engine
 */
class ModelLoader {
 public:
  /**
   * @brief Construct model loader
   * @param registry Model registry for metadata queries
   */
  explicit ModelLoader(std::shared_ptr<registry::ModelRegistry> registry);

  /**
   * @brief Load model by name
   * @param model_name Model name or ID
   * @param config Loading configuration
   * @return Loaded model components, or nullopt on failure
   */
  std::optional<LoadedModel> load_model(const std::string& model_name,
                                        const LoadModelConfig& config =
                                            LoadModelConfig());

  /**
   * @brief Load model by registry ID
   * @param model_id Database ID from registry
   * @param config Loading configuration
   * @return Loaded model components, or nullopt on failure
   */
  std::optional<LoadedModel> load_model_by_id(int64_t model_id,
                                               const LoadModelConfig& config =
                                                   LoadModelConfig());

  /**
   * @brief Load tokenizer for a model
   * @param info Model metadata
   * @return Tokenizer, or nullptr on failure
   */
  std::shared_ptr<runtime::Tokenizer> load_tokenizer(
      const registry::ModelInfo& info);

  /**
   * @brief Create pager for KV cache
   * @param config KV cache configuration
   * @return Pager instance
   */
  std::shared_ptr<runtime::kv::Pager> create_pager(
      const LoadModelConfig& config);

  /**
   * @brief Get last error message
   */
  const std::string& last_error() const { return last_error_; }

 private:
  /**
   * @brief Load model weights via mmap
   * @param file_path Path to model file
   * @param prefetch Whether to prefetch pages
   * @param lock Whether to lock in memory
   * @return Weight loader, or nullptr on failure
   */
  std::shared_ptr<MMapWeightLoader> load_weights(const std::string& file_path,
                                                  bool prefetch, bool lock);

  /**
   * @brief Create CachedLlamaModel from weights
   * @param loader Weight loader
   * @param info Model metadata
   * @param pager KV cache pager
   * @return Model instance, or nullptr on failure
   */
  std::shared_ptr<graph::CachedLlamaModel> create_cached_model(
      std::shared_ptr<MMapWeightLoader> loader,
      const registry::ModelInfo& info,
      std::shared_ptr<runtime::kv::Pager> pager);

  /**
   * @brief Create LlamaModel from weights (fallback, no Metal kernels)
   * @param loader Weight loader
   * @param info Model metadata
   * @return Model instance, or nullptr on failure
   */
  std::shared_ptr<graph::LlamaModel> create_simple_model(
      std::shared_ptr<MMapWeightLoader> loader,
      const registry::ModelInfo& info);

  /**
   * @brief Load GGUF tensor metadata and register with weight loader
   * @param loader Weight loader to register tensors with
   * @param file_path Path to GGUF file
   * @return true if successful, false otherwise
   */
  bool load_gguf_tensors(std::shared_ptr<MMapWeightLoader> loader,
                         const std::string& file_path);

  std::shared_ptr<registry::ModelRegistry> registry_;
  std::string last_error_;
};

}  // namespace server
}  // namespace mlxr
