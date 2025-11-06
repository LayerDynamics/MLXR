/**
 * @file model.h
 * @brief Complete Llama model implementation with weight loading
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "attention_cached.h"
#include "layers.h"
#include "tensor.h"

namespace mlxr {
namespace graph {

/**
 * @brief KV cache for incremental inference
 *
 * Stores cached key and value tensors for each transformer layer.
 * Enables O(1) decode instead of O(n²) by reusing previous computations.
 */
struct KVCache {
  /**
   * @brief Per-layer cache entries
   *
   * Each entry is a pair of (key_cache, value_cache) tensors.
   * Shape: [batch, num_kv_heads, cached_seq_len, head_dim]
   */
  std::vector<std::pair<Tensor, Tensor>> layer_caches;

  /**
   * @brief Number of tokens currently cached
   */
  int cached_length = 0;

  /**
   * @brief Check if cache is initialized
   */
  bool is_initialized() const {
    return !layer_caches.empty() && cached_length > 0;
  }

  /**
   * @brief Clear the cache
   */
  void clear() {
    layer_caches.clear();
    cached_length = 0;
  }

  /**
   * @brief Reserve space for n_layers
   */
  void reserve(int n_layers) { layer_caches.reserve(n_layers); }
};

/**
 * @brief Configuration for Llama model architecture
 */
struct ModelConfig {
  int hidden_size;        ///< Hidden dimension (e.g., 2048 for TinyLlama)
  int num_layers;         ///< Number of transformer layers (e.g., 22)
  int num_heads;          ///< Number of attention heads (e.g., 32)
  int num_kv_heads;       ///< Number of KV heads for GQA (e.g., 4)
  int intermediate_size;  ///< MLP intermediate dimension (e.g., 5632)
  int vocab_size;         ///< Vocabulary size (e.g., 32000)
  int max_seq_len;        ///< Maximum sequence length (e.g., 2048)
  float norm_eps;         ///< RMSNorm epsilon (default: 1e-6)
  float rope_base;        ///< RoPE base frequency (default: 10000.0)

  /**
   * @brief Create default TinyLlama-1.1B config
   */
  static ModelConfig tinyllama_1_1b();

  /**
   * @brief Load config from JSON file
   */
  static ModelConfig from_json(const std::string& path);

  /**
   * @brief Load config from HuggingFace config.json format
   */
  static ModelConfig from_hf_config(const std::string& path);
};

/**
 * @brief Complete Llama model implementation
 *
 * Implements the full Llama architecture with:
 * - Token embeddings
 * - Transformer blocks (attention + MLP + normalization)
 * - Final normalization
 * - Language modeling head
 */
class LlamaModel {
 public:
  /**
   * @brief Construct Llama model from configuration
   * @param config Model architecture configuration
   */
  explicit LlamaModel(const ModelConfig& config);

  /**
   * @brief Forward pass through the model
   * @param input_ids Token IDs [batch, seq_len]
   * @param mask Optional attention mask [batch, 1, seq_len, seq_len]
   * @param kv_cache Optional KV cache for incremental inference
   * @return Logits [batch, seq_len, vocab_size]
   */
  Tensor forward(const Tensor& input_ids, const Tensor* mask = nullptr,
                 KVCache* kv_cache = nullptr);

  /**
   * @brief Load weights from safetensors file
   * @param path Path to .safetensors file
   * @return True if successful, false otherwise
   */
  bool load_weights(const std::string& path);

  /**
   * @brief Load weights from directory (HuggingFace format)
   *
   * Loads model.safetensors or pytorch_model.bin from directory
   * @param dir_path Path to model directory
   * @return True if successful, false otherwise
   */
  bool load_weights_from_dir(const std::string& dir_path);

  /**
   * @brief Load weights from MLX format
   *
   * Uses MLX's native load/save format
   * @param path Path to .npz file
   * @return True if successful, false otherwise
   */
  bool load_weights_mlx(const std::string& path);

  /**
   * @brief Get model configuration
   */
  const ModelConfig& config() const { return config_; }

  /**
   * @brief Get embedding layer
   */
  Tensor& embeddings() { return embed_tokens_; }
  const Tensor& embeddings() const { return embed_tokens_; }

  /**
   * @brief Get transformer blocks
   */
  std::vector<TransformerBlock>& blocks() { return blocks_; }
  const std::vector<TransformerBlock>& blocks() const { return blocks_; }

  /**
   * @brief Get final normalization layer
   */
  RMSNorm& norm() { return norm_; }
  const RMSNorm& norm() const { return norm_; }

  /**
   * @brief Get language modeling head
   */
  Tensor& lm_head() { return lm_head_; }
  const Tensor& lm_head() const { return lm_head_; }

 private:
  /**
   * @brief Load weights from safetensors format
   */
  bool load_safetensors(const std::string& path);

  /**
   * @brief Map weight name from HuggingFace format to internal format
   */
  std::string map_weight_name(const std::string& hf_name) const;

  /**
   * @brief Assign loaded weights to model parameters
   */
  bool assign_weights(const std::unordered_map<std::string, Tensor>& weights);

  ModelConfig config_;

  // Model components
  Tensor embed_tokens_;  ///< Token embeddings [vocab_size, hidden_size]
  std::vector<TransformerBlock> blocks_;  ///< Transformer blocks
  RMSNorm norm_;                          ///< Final normalization
  Tensor lm_head_;  ///< Output projection [vocab_size, hidden_size]
};

/**
 * @brief Load a Llama model from directory
 *
 * Convenience function that loads config and weights
 * @param model_dir Directory containing config.json and model weights
 * @return Unique pointer to loaded model, or nullptr on failure
 */
std::unique_ptr<LlamaModel> load_llama_model(const std::string& model_dir);

// Forward declarations
class CachedTransformerBlock;

}  // namespace graph

// Forward declaration for Pager
namespace runtime {
namespace kv {
class Pager;
}
}  // namespace runtime

namespace graph {

/**
 * @brief Llama model with Metal-accelerated cached attention
 *
 * Variant of LlamaModel that uses CachedTransformerBlock with paged
 * KV cache and Metal attention kernels for optimal performance.
 */
class CachedLlamaModel {
 public:
  /**
   * @brief Construct cached Llama model
   * @param config Model architecture configuration
   * @param pager KV cache pager (required)
   */
  CachedLlamaModel(const ModelConfig& config,
                   std::shared_ptr<runtime::kv::Pager> pager);

  /**
   * @brief Forward pass with cached attention
   * @param input_ids Token IDs [batch, seq_len]
   * @param seq_id Sequence ID for KV cache
   * @param start_pos Starting position in sequence
   * @param mask Optional attention mask
   * @return Logits [batch, seq_len, vocab_size]
   */
  Tensor forward(const Tensor& input_ids, int seq_id = 0, int start_pos = 0,
                 const Tensor* mask = nullptr);

  /**
   * @brief Load weights (delegates to internal LlamaModel)
   */
  bool load_weights(const std::string& path);
  bool load_weights_from_dir(const std::string& dir_path);

  /**
   * @brief Get model configuration
   */
  const ModelConfig& config() const { return config_; }

  /**
   * @brief Get pager
   */
  std::shared_ptr<runtime::kv::Pager> pager() { return pager_; }

  /**
   * @brief Get embedding layer
   */
  Tensor& embeddings() { return embed_tokens_; }
  const Tensor& embeddings() const { return embed_tokens_; }

  /**
   * @brief Get cached transformer blocks
   */
  std::vector<CachedTransformerBlock>& blocks() { return cached_blocks_; }

  /**
   * @brief Get final normalization layer
   */
  RMSNorm& norm() { return norm_; }

  /**
   * @brief Get language modeling head
   */
  Tensor& lm_head() { return lm_head_; }

 private:
  bool load_safetensors(const std::string& path);
  std::string map_weight_name(const std::string& hf_name) const;
  bool assign_weights(const std::unordered_map<std::string, Tensor>& weights);

  ModelConfig config_;
  std::shared_ptr<runtime::kv::Pager> pager_;

  // Model components
  Tensor embed_tokens_;
  std::vector<CachedTransformerBlock> cached_blocks_;
  RMSNorm norm_;
  Tensor lm_head_;
};

}  // namespace graph
}  // namespace mlxr
