/**
 * @file model.cpp
 * @brief Implementation of complete Llama model with weight loading
 */

#include "model.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../runtime/kv/pager.h"
#include "attention_cached.h"
#include "mlx/mlx.h"

// For JSON parsing (will use simple parsing or include nlohmann/json)
#include <algorithm>
#include <filesystem>

namespace mlxr {
namespace graph {

// C++17 compatible string methods
static bool ends_with(const std::string& str, const std::string& suffix) {
  if (suffix.length() > str.length()) return false;
  return str.compare(str.length() - suffix.length(), suffix.length(), suffix) ==
         0;
}

static bool starts_with(const std::string& str, const std::string& prefix) {
  if (prefix.length() > str.length()) return false;
  return str.compare(0, prefix.length(), prefix) == 0;
}

// ============================================================================
// ModelConfig Implementation
// ============================================================================

ModelConfig ModelConfig::tinyllama_1_1b() {
  ModelConfig config;
  config.hidden_size = 2048;
  config.num_layers = 22;
  config.num_heads = 32;
  config.num_kv_heads = 4;
  config.intermediate_size = 5632;
  config.vocab_size = 32000;
  config.max_seq_len = 2048;
  config.norm_eps = 1e-6f;
  config.rope_base = 10000.0f;
  return config;
}

// Simple JSON parser for config files
// TODO: Replace with proper JSON library (nlohmann/json) for production
static std::unordered_map<std::string, std::string> parse_simple_json(
    const std::string& json_str) {
  std::unordered_map<std::string, std::string> result;

  // Very simple parser - just extract key-value pairs
  // Format: "key": value or "key": "value"
  size_t pos = 0;
  while (pos < json_str.length()) {
    // Find key
    size_t key_start = json_str.find('"', pos);
    if (key_start == std::string::npos) break;
    key_start++;

    size_t key_end = json_str.find('"', key_start);
    if (key_end == std::string::npos) break;

    std::string key = json_str.substr(key_start, key_end - key_start);

    // Find colon
    size_t colon = json_str.find(':', key_end);
    if (colon == std::string::npos) break;

    // Find value
    size_t value_start = colon + 1;
    while (value_start < json_str.length() &&
           std::isspace(json_str[value_start])) {
      value_start++;
    }

    size_t value_end;
    if (json_str[value_start] == '"') {
      // String value
      value_start++;
      value_end = json_str.find('"', value_start);
    } else {
      // Numeric value
      value_end = json_str.find_first_of(",}", value_start);
    }

    if (value_end == std::string::npos) break;

    std::string value = json_str.substr(value_start, value_end - value_start);
    // Trim whitespace
    value.erase(0, value.find_first_not_of(" \t\n\r"));
    value.erase(value.find_last_not_of(" \t\n\r") + 1);

    result[key] = value;
    pos = value_end + 1;
  }

  return result;
}

ModelConfig ModelConfig::from_json(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json_str = buffer.str();

  auto json_map = parse_simple_json(json_str);

  ModelConfig config;
  config.hidden_size = std::stoi(json_map["hidden_size"]);
  config.num_layers = std::stoi(json_map["num_hidden_layers"]);
  config.num_heads = std::stoi(json_map["num_attention_heads"]);
  config.num_kv_heads = std::stoi(json_map["num_key_value_heads"]);
  config.intermediate_size = std::stoi(json_map["intermediate_size"]);
  config.vocab_size = std::stoi(json_map["vocab_size"]);
  config.max_seq_len = std::stoi(json_map["max_position_embeddings"]);

  // Optional fields with defaults
  if (json_map.count("rms_norm_eps")) {
    config.norm_eps = std::stof(json_map["rms_norm_eps"]);
  } else {
    config.norm_eps = 1e-6f;
  }

  if (json_map.count("rope_theta")) {
    config.rope_base = std::stof(json_map["rope_theta"]);
  } else {
    config.rope_base = 10000.0f;
  }

  return config;
}

ModelConfig ModelConfig::from_hf_config(const std::string& path) {
  // HuggingFace format is same as our JSON format
  return from_json(path);
}

// ============================================================================
// LlamaModel Implementation
// ============================================================================

LlamaModel::LlamaModel(const ModelConfig& config)
    : config_(config),
      embed_tokens_(mlxr::graph::zeros({config.vocab_size, config.hidden_size},
                                       mlx::core::float32)),
      norm_(config.hidden_size, config.norm_eps),
      lm_head_(mlxr::graph::zeros({config.vocab_size, config.hidden_size},
                                  mlx::core::float32)) {
  // Initialize transformer blocks
  blocks_.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    blocks_.emplace_back(config.hidden_size, config.num_heads,
                         config.intermediate_size, config.max_seq_len,
                         config.norm_eps, config.num_kv_heads);
  }
}

Tensor LlamaModel::forward(const Tensor& input_ids, const Tensor* mask,
                           KVCache* kv_cache) {
  // input_ids shape: [batch, seq_len]
  auto shape = input_ids.shape();
  int batch = shape[0];
  int seq_len = shape[1];

  // Embedding lookup: [batch, seq_len, hidden_size]
  // Use MLX's take operation for embedding lookup
  auto input_ids_arr = input_ids.array();
  auto embed_arr = embed_tokens_.array();

  // Flatten input_ids to 1D for indexing
  auto flat_ids = mlx::core::reshape(input_ids_arr, {batch * seq_len});

  // Take embeddings
  auto flat_embeds = mlx::core::take(embed_arr, flat_ids, 0);

  // Reshape back to [batch, seq_len, hidden_size]
  auto hidden_states = Tensor(
      mlx::core::reshape(flat_embeds, {batch, seq_len, config_.hidden_size}));

  // Pass through transformer blocks with KV cache
  for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
    hidden_states = blocks_[i].forward(hidden_states, mask, kv_cache, i);
  }

  // Update cache length after processing all layers
  if (kv_cache != nullptr) {
    kv_cache->cached_length += seq_len;
  }

  // Final normalization
  hidden_states = norm_.forward(hidden_states);

  // Project to vocabulary: [batch, seq_len, vocab_size]
  // lm_head weight shape: [vocab_size, hidden_size]
  // We need to compute hidden_states @ lm_head.T
  auto output = matmul(hidden_states, lm_head_.transpose());

  // Force evaluation of the output tensor before returning
  // MLX uses lazy evaluation, so we need to explicitly eval() to ensure
  // logits are fully computed before being passed to the sampler
  mlx::core::eval(output.array());

  return output;
}

bool LlamaModel::load_weights(const std::string& path) {
  if (ends_with(path, ".safetensors")) {
    return load_safetensors(path);
  } else if (ends_with(path, ".npz")) {
    return load_weights_mlx(path);
  } else {
    std::cerr << "Unknown weight format: " << path << std::endl;
    std::cerr << "Supported formats: .safetensors, .npz" << std::endl;
    return false;
  }
}

bool LlamaModel::load_weights_from_dir(const std::string& dir_path) {
  namespace fs = std::filesystem;

  // Check for model.safetensors
  fs::path safetensors_path = fs::path(dir_path) / "model.safetensors";
  if (fs::exists(safetensors_path)) {
    return load_safetensors(safetensors_path.string());
  }

  // Check for weights.npz (MLX format)
  fs::path npz_path = fs::path(dir_path) / "weights.npz";
  if (fs::exists(npz_path)) {
    return load_weights_mlx(npz_path.string());
  }

  // Check for pytorch_model.bin
  fs::path pytorch_path = fs::path(dir_path) / "pytorch_model.bin";
  if (fs::exists(pytorch_path)) {
    std::cerr << "PyTorch .bin format not yet supported. " << std::endl;
    std::cerr << "Please convert to safetensors or MLX format." << std::endl;
    return false;
  }

  std::cerr << "No compatible weight files found in: " << dir_path << std::endl;
  return false;
}

bool LlamaModel::load_weights_mlx(const std::string& path) {
  (void)path;  // Suppress unused warning
  try {
    // For MLX NPZ format, we need to use Python's numpy/npz loader
    // or implement custom NPZ parsing. For now, use safetensors instead.
    std::cerr << "NPZ loading not yet implemented. " << std::endl;
    std::cerr << "Please convert your model to safetensors format."
              << std::endl;
    std::cerr
        << "You can use: mlx.core.save_safetensors() or HF convert script."
        << std::endl;
    return false;

  } catch (const std::exception& e) {
    std::cerr << "Failed to load MLX weights: " << e.what() << std::endl;
    return false;
  }
}

bool LlamaModel::load_safetensors(const std::string& path) {
  try {
    // MLX has native safetensors support via load_safetensors()
    std::cout << "Loading weights from safetensors format: " << path
              << std::endl;

    // load_safetensors returns std::pair<unordered_map<string, array>,
    // metadata>
    auto loaded = mlx::core::load_safetensors(path);
    auto& weights_map = loaded.first;

    // Convert to our Tensor format
    std::unordered_map<std::string, Tensor> tensor_map;
    for (const auto& weight_pair : weights_map) {
      tensor_map[weight_pair.first] = Tensor(weight_pair.second);
    }

    std::cout << "Loaded " << tensor_map.size() << " weight tensors"
              << std::endl;
    return assign_weights(tensor_map);

  } catch (const std::exception& e) {
    std::cerr << "Failed to load safetensors: " << e.what() << std::endl;
    return false;
  }
}

std::string LlamaModel::map_weight_name(const std::string& hf_name) const {
  // Map HuggingFace naming to our internal naming
  // HF format: model.layers.0.self_attn.q_proj.weight
  // Our format: blocks.0.attention.q_proj.weight

  std::string mapped = hf_name;

  // Replace common prefixes
  if (starts_with(mapped, "model.")) {
    mapped = mapped.substr(6);  // Remove "model."
  }

  // Replace "layers" with "blocks"
  size_t pos = 0;
  while ((pos = mapped.find("layers.", pos)) != std::string::npos) {
    mapped.replace(pos, 7, "blocks.");
    pos += 7;
  }

  // Replace "self_attn" with "attention"
  pos = 0;
  while ((pos = mapped.find("self_attn.", pos)) != std::string::npos) {
    mapped.replace(pos, 10, "attention.");
    pos += 10;
  }

  // Replace "mlp" remains "mlp"
  // Replace "input_layernorm" remains "input_layernorm"
  // Replace "post_attention_layernorm" remains "post_attention_layernorm"

  return mapped;
}

bool LlamaModel::assign_weights(
    const std::unordered_map<std::string, Tensor>& weights) {
  int weights_assigned = 0;

  for (const auto& [name, tensor] : weights) {
    std::string mapped_name = map_weight_name(name);

    try {
      // Token embeddings
      if (mapped_name == "embed_tokens.weight") {
        embed_tokens_ = tensor;
        weights_assigned++;
        continue;
      }

      // Final norm
      if (mapped_name == "norm.weight") {
        norm_.weight() = tensor;
        weights_assigned++;
        continue;
      }

      // LM head
      if (mapped_name == "lm_head.weight") {
        lm_head_ = tensor;
        weights_assigned++;
        continue;
      }

      // Transformer blocks
      if (starts_with(mapped_name, "blocks.")) {
        // Extract layer number
        size_t layer_start = 7;  // After "blocks."
        size_t layer_end = mapped_name.find('.', layer_start);
        int layer_idx =
            std::stoi(mapped_name.substr(layer_start, layer_end - layer_start));

        if (layer_idx >= config_.num_layers) {
          std::cerr << "Layer index out of range: " << layer_idx << std::endl;
          continue;
        }

        std::string layer_suffix = mapped_name.substr(layer_end + 1);
        auto& block = blocks_[layer_idx];

        // Input layernorm
        if (layer_suffix == "input_layernorm.weight") {
          block.input_layernorm().weight() = tensor;
          weights_assigned++;
        }
        // Attention weights
        else if (layer_suffix == "attention.q_proj.weight") {
          block.attention().q_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.k_proj.weight") {
          block.attention().k_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.v_proj.weight") {
          block.attention().v_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.o_proj.weight") {
          block.attention().o_proj().weight() = tensor;
          weights_assigned++;
        }
        // Post-attention layernorm
        else if (layer_suffix == "post_attention_layernorm.weight") {
          block.post_attention_layernorm().weight() = tensor;
          weights_assigned++;
        }
        // MLP weights
        else if (layer_suffix == "mlp.gate_proj.weight") {
          block.mlp().gate_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "mlp.up_proj.weight") {
          block.mlp().up_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "mlp.down_proj.weight") {
          block.mlp().down_proj().weight() = tensor;
          weights_assigned++;
        }

        continue;
      }

      // If we get here, weight name wasn't recognized
      std::cerr << "Unrecognized weight name: " << name << " (mapped to "
                << mapped_name << ")" << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "Error assigning weight " << name << ": " << e.what()
                << std::endl;
      return false;
    }
  }

  std::cout << "Successfully assigned " << weights_assigned << " weights"
            << std::endl;

  // Verify we have all required weights
  int expected_weights =
      1 +                       // embed_tokens
      1 +                       // norm
      1 +                       // lm_head
      config_.num_layers * 10;  // Each layer: 2 norms + 4 attn + 3 mlp + 1
                                // input_norm

  if (weights_assigned < expected_weights * 0.9) {  // Allow 10% tolerance
    std::cerr << "Warning: Only assigned " << weights_assigned << " weights, "
              << "expected ~" << expected_weights << std::endl;
  }

  return true;
}

// ============================================================================
// Convenience Functions
// ============================================================================

std::unique_ptr<LlamaModel> load_llama_model(const std::string& model_dir) {
  namespace fs = std::filesystem;

  // Load config
  fs::path config_path = fs::path(model_dir) / "config.json";
  if (!fs::exists(config_path)) {
    std::cerr << "config.json not found in: " << model_dir << std::endl;
    return nullptr;
  }

  ModelConfig config;
  try {
    config = ModelConfig::from_hf_config(config_path.string());
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config: " << e.what() << std::endl;
    return nullptr;
  }

  // Create model
  auto model = std::make_unique<LlamaModel>(config);

  // Load weights
  if (!model->load_weights_from_dir(model_dir)) {
    std::cerr << "Failed to load weights from: " << model_dir << std::endl;
    return nullptr;
  }

  std::cout << "Successfully loaded Llama model from: " << model_dir
            << std::endl;
  std::cout << "  Hidden size: " << config.hidden_size << std::endl;
  std::cout << "  Layers: " << config.num_layers << std::endl;
  std::cout << "  Attention heads: " << config.num_heads << std::endl;
  std::cout << "  Vocab size: " << config.vocab_size << std::endl;

  return model;
}

// ============================================================================
// CachedLlamaModel Implementation
// ============================================================================

CachedLlamaModel::CachedLlamaModel(const ModelConfig& config,
                                   std::shared_ptr<runtime::kv::Pager> pager)
    : config_(config),
      pager_(pager),
      embed_tokens_(mlxr::graph::zeros({config.vocab_size, config.hidden_size},
                                       mlx::core::float32)),
      norm_(config.hidden_size, config.norm_eps),
      lm_head_(mlxr::graph::zeros({config.vocab_size, config.hidden_size},
                                  mlx::core::float32)) {
  if (!pager_) {
    throw std::invalid_argument("Pager cannot be null for CachedLlamaModel");
  }

  // Initialize cached transformer blocks with Metal kernel support
  cached_blocks_.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    cached_blocks_.emplace_back(config.hidden_size, config.num_heads,
                                config.num_kv_heads, config.intermediate_size,
                                config.max_seq_len,
                                i,  // layer_idx
                                pager_, config.norm_eps);
  }

  std::cout
      << "[CachedLlamaModel] Initialized with Metal attention kernels enabled"
      << std::endl;
}

Tensor CachedLlamaModel::forward(const Tensor& input_ids, int seq_id,
                                 int start_pos, const Tensor* mask) {
  // input_ids shape: [batch, seq_len]
  auto shape = input_ids.shape();
  int batch = shape[0];
  int seq_len = shape[1];

  // Embedding lookup
  auto input_ids_arr = input_ids.array();
  auto embed_arr = embed_tokens_.array();
  auto flat_ids = mlx::core::reshape(input_ids_arr, {batch * seq_len});
  auto flat_embeds = mlx::core::take(embed_arr, flat_ids, 0);
  auto hidden_states = Tensor(
      mlx::core::reshape(flat_embeds, {batch, seq_len, config_.hidden_size}));

  // Pass through cached transformer blocks
  // These will use Metal kernels for attention!
  for (int i = 0; i < static_cast<int>(cached_blocks_.size()); ++i) {
    hidden_states =
        cached_blocks_[i].forward(hidden_states, seq_id, start_pos, mask);
  }

  // Final normalization
  hidden_states = norm_.forward(hidden_states);

  // Project to vocabulary
  auto output = matmul(hidden_states, lm_head_.transpose());

  // Force evaluation of the output tensor before returning
  // MLX uses lazy evaluation, so we need to explicitly eval() to ensure
  // logits are fully computed before being passed to the sampler
  mlx::core::eval(output.array());

  return output;
}

bool CachedLlamaModel::load_weights(const std::string& path) {
  if (ends_with(path, ".safetensors")) {
    return load_safetensors(path);
  } else {
    std::cerr << "Unknown weight format: " << path << std::endl;
    return false;
  }
}

bool CachedLlamaModel::load_weights_from_dir(const std::string& dir_path) {
  namespace fs = std::filesystem;

  fs::path safetensors_path = fs::path(dir_path) / "model.safetensors";
  if (fs::exists(safetensors_path)) {
    return load_safetensors(safetensors_path.string());
  }

  std::cerr << "No compatible weight files found in: " << dir_path << std::endl;
  return false;
}

bool CachedLlamaModel::load_from_weight_map(
    const std::unordered_map<std::string, Tensor>& weights) {
  return assign_weights(weights);
}

bool CachedLlamaModel::load_safetensors(const std::string& path) {
  try {
    std::cout << "Loading weights from safetensors format: " << path
              << std::endl;

    auto loaded = mlx::core::load_safetensors(path);
    auto& weights_map = loaded.first;

    std::unordered_map<std::string, Tensor> tensor_map;
    for (const auto& weight_pair : weights_map) {
      tensor_map[weight_pair.first] = Tensor(weight_pair.second);
    }

    std::cout << "Loaded " << tensor_map.size() << " weight tensors"
              << std::endl;
    return assign_weights(tensor_map);

  } catch (const std::exception& e) {
    std::cerr << "Failed to load safetensors: " << e.what() << std::endl;
    return false;
  }
}

std::string CachedLlamaModel::map_weight_name(
    const std::string& hf_name) const {
  std::string mapped = hf_name;

  if (starts_with(mapped, "model.")) {
    mapped = mapped.substr(6);
  }

  size_t pos = 0;
  while ((pos = mapped.find("layers.", pos)) != std::string::npos) {
    mapped.replace(pos, 7, "blocks.");
    pos += 7;
  }

  pos = 0;
  while ((pos = mapped.find("self_attn.", pos)) != std::string::npos) {
    mapped.replace(pos, 10, "attention.");
    pos += 10;
  }

  return mapped;
}

bool CachedLlamaModel::assign_weights(
    const std::unordered_map<std::string, Tensor>& weights) {
  int weights_assigned = 0;

  for (const auto& [name, tensor] : weights) {
    std::string mapped_name = map_weight_name(name);

    try {
      // Token embeddings
      if (mapped_name == "embed_tokens.weight") {
        embed_tokens_ = tensor;
        weights_assigned++;
        continue;
      }

      // Final norm
      if (mapped_name == "norm.weight") {
        norm_.weight() = tensor;
        weights_assigned++;
        continue;
      }

      // LM head
      if (mapped_name == "lm_head.weight") {
        lm_head_ = tensor;
        weights_assigned++;
        continue;
      }

      // Cached transformer blocks
      if (starts_with(mapped_name, "blocks.")) {
        size_t layer_start = 7;
        size_t layer_end = mapped_name.find('.', layer_start);
        int layer_idx =
            std::stoi(mapped_name.substr(layer_start, layer_end - layer_start));

        if (layer_idx >= config_.num_layers) {
          std::cerr << "Layer index out of range: " << layer_idx << std::endl;
          continue;
        }

        std::string layer_suffix = mapped_name.substr(layer_end + 1);
        auto& block = cached_blocks_[layer_idx];

        // Input layernorm
        if (layer_suffix == "input_layernorm.weight") {
          block.input_layernorm().weight() = tensor;
          weights_assigned++;
        }
        // Attention weights (access via CachedAttention -> Attention)
        else if (layer_suffix == "attention.q_proj.weight") {
          block.attention().attention().q_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.k_proj.weight") {
          block.attention().attention().k_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.v_proj.weight") {
          block.attention().attention().v_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "attention.o_proj.weight") {
          block.attention().attention().o_proj().weight() = tensor;
          weights_assigned++;
        }
        // Post-attention layernorm
        else if (layer_suffix == "post_attention_layernorm.weight") {
          block.post_attention_layernorm().weight() = tensor;
          weights_assigned++;
        }
        // MLP weights
        else if (layer_suffix == "mlp.gate_proj.weight") {
          block.mlp().gate_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "mlp.up_proj.weight") {
          block.mlp().up_proj().weight() = tensor;
          weights_assigned++;
        } else if (layer_suffix == "mlp.down_proj.weight") {
          block.mlp().down_proj().weight() = tensor;
          weights_assigned++;
        }

        continue;
      }

      std::cerr << "Unrecognized weight name: " << name << " (mapped to "
                << mapped_name << ")" << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "Error assigning weight " << name << ": " << e.what()
                << std::endl;
      return false;
    }
  }

  std::cout << "Successfully assigned " << weights_assigned << " weights"
            << std::endl;

  int expected_weights = 1 + 1 + 1 + config_.num_layers * 10;
  if (weights_assigned < expected_weights * 0.9) {
    std::cerr << "Warning: Only assigned " << weights_assigned << " weights, "
              << "expected ~" << expected_weights << std::endl;
  }

  return true;
}

}  // namespace graph
}  // namespace mlxr
