// Copyright © 2025 MLXR Development
// SQLite-backed model registry

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare SQLite types to avoid header dependency
struct sqlite3;
struct sqlite3_stmt;

namespace mlxr {
namespace registry {

// Model format types
enum class ModelFormat {
  GGUF,         // GGUF/GGML format (llama.cpp)
  SAFETENSORS,  // HuggingFace safetensors
  MLX_NATIVE,   // MLX native format
  UNKNOWN
};

// Model quantization type
enum class QuantizationType {
  NONE,  // FP32/FP16
  Q4_0,
  Q4_1,
  Q5_0,
  Q5_1,
  Q8_0,
  Q2_K,
  Q3_K,
  Q4_K,
  Q5_K,
  Q6_K,
  Q8_K,
  IQ2_XXS,
  IQ2_XS,
  IQ3_XXS,
  MIXED  // Mixed quantization
};

// Model architecture family
enum class ModelArchitecture {
  LLAMA,
  MISTRAL,
  MIXTRAL,
  GEMMA,
  PHI,
  QWEN,
  UNKNOWN
};

// Model metadata
struct ModelInfo {
  // Identity
  int64_t id;            // Database ID
  std::string name;      // User-friendly name
  std::string model_id;  // Unique identifier (e.g., "llama-2-7b-chat")
  ModelArchitecture architecture;  // Architecture family

  // File info
  std::string file_path;  // Path to model file
  ModelFormat format;     // File format
  uint64_t file_size;     // Size in bytes
  std::string sha256;     // File checksum

  // Model parameters
  uint64_t param_count;   // Total parameters
  int context_length;     // Maximum context length
  int hidden_size;        // Hidden dimension
  int num_layers;         // Number of layers
  int num_heads;          // Number of attention heads
  int num_kv_heads;       // Number of KV heads (for GQA)
  int intermediate_size;  // FFN intermediate size
  int vocab_size;         // Vocabulary size

  // Quantization
  QuantizationType quant_type;  // Quantization type
  std::string quant_details;    // Additional quant info (JSON)

  // Tokenizer
  std::string tokenizer_type;  // "sentencepiece", "tiktoken", "hf"
  std::string tokenizer_path;  // Path to tokenizer file

  // RoPE configuration
  float rope_freq_base;           // RoPE frequency base
  float rope_scale;               // RoPE scaling factor
  std::string rope_scaling_type;  // "none", "linear", "ntk", "yarn"

  // Metadata
  std::string description;        // Human-readable description
  std::string license;            // License type
  std::string source_url;         // Download URL
  std::vector<std::string> tags;  // Tags for search

  // State
  bool is_loaded;               // Currently loaded in memory
  int64_t last_used_timestamp;  // Last access time (unix timestamp)
  int64_t created_timestamp;    // Creation time

  // Chat template
  std::string chat_template;  // Jinja2 chat template
};

// Adapter (LoRA) metadata
struct AdapterInfo {
  int64_t id;
  int64_t base_model_id;  // Parent model
  std::string name;
  std::string adapter_id;
  std::string file_path;
  std::string adapter_type;                 // "lora", "qlora", "ia3"
  int rank;                                 // LoRA rank
  float scale;                              // Adapter scale/alpha
  std::vector<std::string> target_modules;  // Which layers are adapted
  int64_t created_timestamp;
};

// Model tag for search and organization
struct ModelTag {
  int64_t model_id;
  std::string key;
  std::string value;
};

// Registry query options
struct QueryOptions {
  std::optional<ModelArchitecture> architecture;
  std::optional<ModelFormat> format;
  std::optional<QuantizationType> quant_type;
  std::optional<std::string> search_term;  // Search in name/description
  std::vector<std::string> required_tags;
  int limit = 100;
  int offset = 0;
  std::string order_by = "last_used_timestamp DESC";  // SQL ORDER BY clause
};

// Main model registry class
class ModelRegistry {
 public:
  /**
   * Create registry with database at given path
   * @param db_path Path to SQLite database file
   * @param create_if_missing Create database if it doesn't exist
   */
  explicit ModelRegistry(const std::string& db_path,
                         bool create_if_missing = true);

  ~ModelRegistry();

  // Non-copyable
  ModelRegistry(const ModelRegistry&) = delete;
  ModelRegistry& operator=(const ModelRegistry&) = delete;

  /**
   * Initialize database schema
   * @return true if successful
   */
  bool initialize();

  /**
   * Register a new model
   * @param info Model metadata
   * @return Model ID if successful, -1 otherwise
   */
  int64_t register_model(const ModelInfo& info);

  /**
   * Update existing model metadata
   * @param info Updated model info (must have valid id)
   * @return true if successful
   */
  bool update_model(const ModelInfo& info);

  /**
   * Remove model from registry
   * @param model_id Model ID to remove
   * @param delete_file Also delete the model file
   * @return true if successful
   */
  bool remove_model(int64_t model_id, bool delete_file = false);

  /**
   * Get model by ID
   * @param model_id Model ID
   * @return Model info if found
   */
  std::optional<ModelInfo> get_model(int64_t model_id) const;

  /**
   * Get model by unique identifier
   * @param model_id Model identifier string
   * @return Model info if found
   */
  std::optional<ModelInfo> get_model_by_identifier(
      const std::string& model_id) const;

  /**
   * List all models matching query
   * @param options Query filter options
   * @return Vector of matching models
   */
  std::vector<ModelInfo> list_models(const QueryOptions& options = {}) const;

  /**
   * Update model's last used timestamp
   * @param model_id Model ID
   */
  void touch_model(int64_t model_id);

  /**
   * Mark model as loaded/unloaded
   * @param model_id Model ID
   * @param loaded Load state
   */
  void set_model_loaded(int64_t model_id, bool loaded);

  /**
   * Add tags to model
   * @param model_id Model ID
   * @param tags Key-value tags
   */
  bool add_tags(int64_t model_id,
                const std::unordered_map<std::string, std::string>& tags);

  /**
   * Get tags for model
   * @param model_id Model ID
   * @return Map of tags
   */
  std::unordered_map<std::string, std::string> get_tags(int64_t model_id) const;

  /**
   * Register adapter for a model
   * @param info Adapter metadata
   * @return Adapter ID if successful, -1 otherwise
   */
  int64_t register_adapter(const AdapterInfo& info);

  /**
   * Get adapters for a model
   * @param base_model_id Base model ID
   * @return Vector of adapters
   */
  std::vector<AdapterInfo> get_adapters(int64_t base_model_id) const;

  /**
   * Remove adapter
   * @param adapter_id Adapter ID
   * @return true if successful
   */
  bool remove_adapter(int64_t adapter_id);

  /**
   * Get registry statistics
   * @return Map of stat name to value
   */
  std::unordered_map<std::string, int64_t> get_stats() const;

  /**
   * Check if database is healthy
   * @return true if database is accessible
   */
  bool health_check() const;

  /**
   * Close database connection
   */
  void close();

 private:
  // Database handle
  sqlite3* db_;
  std::string db_path_;
  mutable std::mutex mutex_;

  // Prepared statements for common queries
  sqlite3_stmt* stmt_insert_model_;
  sqlite3_stmt* stmt_update_model_;
  sqlite3_stmt* stmt_get_model_;
  sqlite3_stmt* stmt_touch_model_;

  // Helper methods
  bool create_schema();
  bool prepare_statements();
  void finalize_statements();
  ModelInfo row_to_model_info(sqlite3_stmt* stmt) const;
  AdapterInfo row_to_adapter_info(sqlite3_stmt* stmt) const;
  std::string architecture_to_string(ModelArchitecture arch) const;
  ModelArchitecture string_to_architecture(const std::string& str) const;
  std::string format_to_string(ModelFormat format) const;
  ModelFormat string_to_format(const std::string& str) const;
  std::string quant_type_to_string(QuantizationType type) const;
  QuantizationType string_to_quant_type(const std::string& str) const;
};

}  // namespace registry
}  // namespace mlxr
