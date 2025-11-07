// Copyright © 2025 MLXR Development
// Model loading utility implementation

#include "model_loader.h"

#include <filesystem>
#include <iostream>
#include <optional>

#include "../../core/graph/attention_cached.h"
#include "../../core/graph/model.h"
#include "../../core/runtime/engine.h"
#include "../../core/runtime/kv/arena.h"
#include "../../core/runtime/kv/eviction.h"
#include "../../core/runtime/kv/pager.h"
#include "../../core/runtime/tokenizer/tokenizer.h"
#include "../registry/gguf_parser.h"
#include "../registry/model_registry.h"

namespace mlxr {
namespace server {

ModelLoader::ModelLoader(std::shared_ptr<registry::ModelRegistry> registry)
    : registry_(registry) {}

//==============================================================================
// Private Helper Methods
//==============================================================================

std::shared_ptr<MMapWeightLoader> ModelLoader::load_weights(
    const std::string& file_path, bool prefetch, bool lock) {
  // Create loader
  auto loader = std::make_shared<MMapWeightLoader>(file_path, true);

  // Initialize (opens file, gets size)
  if (!loader->initialize()) {
    last_error_ = "Failed to initialize weight loader for: " + file_path;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }

  std::cerr << "[ModelLoader] Opened model file: " << file_path << " ("
            << (loader->file_size() / 1024 / 1024) << " MB)" << std::endl;

  // Optionally map entire file and prefetch/lock
  if (prefetch || lock) {
    auto region = loader->map_all(prefetch);
    if (!region.is_valid) {
      last_error_ = "Failed to map weights into memory";
      std::cerr << "[ModelLoader] " << last_error_ << std::endl;
      return nullptr;
    }

    if (lock) {
      if (!loader->lock_memory(region)) {
        std::cerr << "[ModelLoader] Warning: Failed to lock memory"
                  << std::endl;
      }
    }
  }

  return loader;
}

bool ModelLoader::load_gguf_tensors(std::shared_ptr<MMapWeightLoader> loader,
                                    const std::string& file_path) {
  // Parse GGUF file
  registry::GGUFFile gguf;
  if (!gguf.parse(file_path)) {
    last_error_ = "Failed to parse GGUF file: " + gguf.error();
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return false;
  }

  std::cerr << "[ModelLoader] Parsed GGUF: " << gguf.tensors().size()
            << " tensors, arch=" << gguf.get_arch() << std::endl;

  // Get data offset (where tensor data starts in file)
  uint64_t data_offset = gguf.data_offset();

  // Register each tensor with the weight loader
  for (const auto& tensor_info : gguf.tensors()) {
    WeightTensor wt;
    wt.name = tensor_info.name;

    // Convert dimensions from uint64_t to int64_t
    wt.shape.reserve(tensor_info.dimensions.size());
    for (auto dim : tensor_info.dimensions) {
      wt.shape.push_back(static_cast<int64_t>(dim));
    }

    wt.file_offset = data_offset + tensor_info.offset;
    wt.data_size = tensor_info.size;
    wt.dtype = registry::gguf_type_to_mlx_dtype(tensor_info.type);

    // Quantization metadata (if quantized)
    if (static_cast<uint32_t>(tensor_info.type) >=
        static_cast<uint32_t>(registry::GGUFTensorType::Q4_0)) {
      wt.quant_type = registry::gguf_type_name(tensor_info.type);
      wt.quant_block_size =
          static_cast<int>(registry::gguf_block_size(tensor_info.type));
    }

    loader->register_tensor(wt);
  }

  std::cerr << "[ModelLoader] Registered " << gguf.tensors().size()
            << " tensors with loader" << std::endl;

  return true;
}

std::shared_ptr<runtime::Tokenizer> ModelLoader::load_tokenizer(
    const registry::ModelInfo& info) {
  if (info.tokenizer_path.empty()) {
    last_error_ = "No tokenizer path in model info";
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }

  try {
    std::cerr << "[ModelLoader] Loading tokenizer from: " << info.tokenizer_path
              << std::endl;

    // Use runtime::create_tokenizer factory
    auto tokenizer = runtime::create_tokenizer(info.tokenizer_path);

    std::cerr << "[ModelLoader] Tokenizer loaded, vocab_size="
              << tokenizer->vocab_size() << std::endl;

    return tokenizer;

  } catch (const std::exception& e) {
    last_error_ = "Failed to load tokenizer: " + std::string(e.what());
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }
}

std::shared_ptr<runtime::kv::Pager> ModelLoader::create_pager(
    const LoadModelConfig& config) {
  // Create arena config
  runtime::kv::ArenaConfig arena_config;
  arena_config.num_layers = config.kv_num_layers;
  arena_config.num_kv_heads = config.kv_num_heads;
  arena_config.head_dim = config.kv_head_dim;
  arena_config.block_size_tokens = config.kv_block_size;
  arena_config.num_blocks = config.kv_num_blocks;

  std::cerr << "[ModelLoader] Creating Arena: " << arena_config.num_blocks
            << " blocks x " << arena_config.block_size_tokens
            << " tokens/block = "
            << (arena_config.num_blocks * arena_config.block_size_tokens)
            << " token capacity" << std::endl;

  // Create arena
  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);

  // Create pager
  auto pager = std::make_shared<runtime::kv::Pager>(arena);

  std::cerr << "[ModelLoader] Pager created successfully" << std::endl;

  return pager;
}

std::shared_ptr<graph::CachedLlamaModel>
ModelLoader::load_model_from_safetensors(
    std::shared_ptr<graph::CachedLlamaModel> model,
    const registry::ModelInfo& info) {
  std::cerr << "[ModelLoader] Loading safetensors weights..." << std::endl;

  // Get directory containing safetensors file
  std::filesystem::path file_path(info.file_path);
  std::filesystem::path dir_path = file_path.parent_path();

  // Use model's built-in safetensors loader
  if (!model->load_weights_from_dir(dir_path.string())) {
    last_error_ = "Failed to load safetensors weights";
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }

  std::cerr << "[ModelLoader] Safetensors weights loaded successfully"
            << std::endl;

  return model;
}

std::shared_ptr<graph::CachedLlamaModel> ModelLoader::load_model_from_gguf_mmap(
    std::shared_ptr<graph::CachedLlamaModel> model,
    std::shared_ptr<MMapWeightLoader> loader,
    const registry::ModelInfo& info) {
  std::cerr << "[ModelLoader] Loading GGUF weights from mmap..." << std::endl;

  // Build weight map: tensor_name → MLX Tensor
  std::unordered_map<std::string, graph::Tensor> weight_map;

  auto tensor_names = loader->list_tensors();
  std::cerr << "[ModelLoader] Processing " << tensor_names.size()
            << " tensors..." << std::endl;

  int loaded = 0;
  int skipped = 0;

  for (const auto& tensor_name : tensor_names) {
    auto tensor_info_opt = loader->get_tensor_info(tensor_name);
    if (!tensor_info_opt.has_value()) {
      std::cerr << "[ModelLoader] Warning: No info for tensor: " << tensor_name
                << std::endl;
      skipped++;
      continue;
    }

    const auto& tensor_info = tensor_info_opt.value();

    // Map the tensor into memory
    auto region = loader->map_tensor(tensor_name, true);  // with prefetch
    if (!region.is_valid) {
      last_error_ = "Failed to map tensor: " + tensor_name;
      std::cerr << "[ModelLoader] " << last_error_ << std::endl;
      return nullptr;
    }

    // Determine MLX dtype and handle quantized types
    mlx::core::Dtype mlx_dtype = mlx::core::float32;  // Default

    if (tensor_info.dtype == "float16") {
      mlx_dtype = mlx::core::float16;
    } else if (tensor_info.dtype == "float32") {
      mlx_dtype = mlx::core::float32;
    } else if (tensor_info.dtype == "int32") {
      mlx_dtype = mlx::core::int32;
    } else if (tensor_info.dtype == "int64") {
      mlx_dtype = mlx::core::int64;
    } else {
      // Quantized types need dequantization
      // For MVP: just log warning and skip
      std::cerr << "[ModelLoader] Warning: Quantized dtype " << tensor_info.dtype
                << " for " << tensor_name
                << " - dequantization not yet implemented, skipping"
                << std::endl;
      skipped++;
      continue;
    }

    // Convert shape vector<int64_t> to Shape
    std::vector<int> shape_vec;
    shape_vec.reserve(tensor_info.shape.size());
    for (auto dim : tensor_info.shape) {
      shape_vec.push_back(static_cast<int>(dim));
    }
    auto mlx_shape = graph::to_shape(shape_vec);

    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : shape_vec) {
      total_elements *= dim;
    }

    // Create MLX array by COPYING from mmap'd memory
    // This is safer and ensures model works even if loader is destroyed
    std::optional<mlx::core::array> arr_opt;

    if (mlx_dtype == mlx::core::float32) {
      // Copy float32 data
      std::vector<float> data_vec(static_cast<const float*>(region.data),
                                  static_cast<const float*>(region.data) + total_elements);
      arr_opt = mlx::core::array(data_vec.data(), mlx_shape, mlx_dtype);
    } else if (mlx_dtype == mlx::core::float16) {
      // For float16, we need to copy the raw bytes
      // MLX expects fp16 data as uint16_t
      std::vector<uint16_t> data_vec(static_cast<const uint16_t*>(region.data),
                                     static_cast<const uint16_t*>(region.data) + total_elements);
      arr_opt = mlx::core::array(data_vec.data(), mlx_shape, mlx_dtype);
    } else if (mlx_dtype == mlx::core::int32) {
      std::vector<int32_t> data_vec(static_cast<const int32_t*>(region.data),
                                    static_cast<const int32_t*>(region.data) + total_elements);
      arr_opt = mlx::core::array(data_vec.data(), mlx_shape, mlx_dtype);
    } else if (mlx_dtype == mlx::core::int64) {
      std::vector<int64_t> data_vec(static_cast<const int64_t*>(region.data),
                                    static_cast<const int64_t*>(region.data) + total_elements);
      arr_opt = mlx::core::array(data_vec.data(), mlx_shape, mlx_dtype);
    } else {
      std::cerr << "[ModelLoader] Warning: Unsupported dtype for " << tensor_name
                << ", skipping" << std::endl;
      skipped++;
      continue;
    }

    // Force evaluation to trigger the copy
    mlx::core::eval(arr_opt.value());

    // Store in weight map
    weight_map[tensor_name] = graph::Tensor(arr_opt.value());
    loaded++;

    if (loaded % 50 == 0) {
      std::cerr << "[ModelLoader] Loaded " << loaded << "/"
                << tensor_names.size() << " tensors..." << std::endl;
    }
  }

  std::cerr << "[ModelLoader] Finished loading: " << loaded << " loaded, "
            << skipped << " skipped" << std::endl;

  // Use model's load_from_weight_map() method
  std::cerr << "[ModelLoader] Assigning weights to model layers..." << std::endl;

  if (!model->load_from_weight_map(weight_map)) {
    last_error_ = "Failed to assign weights to model layers";
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }

  std::cerr << "[ModelLoader] GGUF weights loaded and assigned successfully"
            << std::endl;

  return model;
}

std::shared_ptr<graph::CachedLlamaModel> ModelLoader::create_cached_model(
    std::shared_ptr<MMapWeightLoader> loader,
    const registry::ModelInfo& info,
    std::shared_ptr<runtime::kv::Pager> pager) {
  // Build ModelConfig from registry info
  graph::ModelConfig model_config;
  model_config.hidden_size = info.hidden_size;
  model_config.num_layers = info.num_layers;
  model_config.num_heads = info.num_heads;
  model_config.num_kv_heads = info.num_kv_heads;
  model_config.intermediate_size = info.intermediate_size;
  model_config.vocab_size = info.vocab_size;
  model_config.max_seq_len = info.context_length;
  model_config.norm_eps = 1e-6f;  // Default, could be in info
  model_config.rope_base = info.rope_freq_base;

  std::cerr << "[ModelLoader] Model config: " << model_config.num_layers
            << " layers, " << model_config.num_heads << " heads, "
            << model_config.num_kv_heads << " KV heads, "
            << "hidden=" << model_config.hidden_size << std::endl;

  // Create model
  auto model = std::make_shared<graph::CachedLlamaModel>(model_config, pager);

  // Load weights based on format
  if (info.format == registry::ModelFormat::SAFETENSORS) {
    return load_model_from_safetensors(model, info);
  } else if (info.format == registry::ModelFormat::GGUF) {
    return load_model_from_gguf_mmap(model, loader, info);
  } else {
    last_error_ = "Unsupported model format";
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return nullptr;
  }
}

//==============================================================================
// Public API
//==============================================================================

std::optional<LoadedModel> ModelLoader::load_model(
    const std::string& model_name, const LoadModelConfig& config) {
  last_error_.clear();

  // STEP 1: Query registry for model metadata
  auto model_info_opt = registry_->get_model_by_identifier(model_name);
  if (!model_info_opt.has_value()) {
    last_error_ = "Model not found in registry: " + model_name;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  const auto& info = model_info_opt.value();
  std::cerr << "[ModelLoader] Found model: " << info.name << " at "
            << info.file_path << std::endl;
  std::cerr << "[ModelLoader] Format: "
            << (info.format == registry::ModelFormat::GGUF ? "GGUF"
                : info.format == registry::ModelFormat::SAFETENSORS
                    ? "SAFETENSORS"
                    : "UNKNOWN")
            << std::endl;

  // STEP 2: Load weights via mmap
  auto loader =
      load_weights(info.file_path, config.prefetch_weights, config.lock_weights);
  if (!loader) {
    return std::nullopt;  // last_error_ set by load_weights()
  }

  // STEP 3: Register GGUF tensors (if GGUF format)
  if (info.format == registry::ModelFormat::GGUF) {
    if (!load_gguf_tensors(loader, info.file_path)) {
      return std::nullopt;  // last_error_ set by load_gguf_tensors()
    }
  }

  // STEP 4: Load tokenizer
  auto tokenizer = load_tokenizer(info);
  if (!tokenizer) {
    return std::nullopt;
  }

  // STEP 5: Create KV cache pager
  // Update config with model info
  LoadModelConfig updated_config = config;
  updated_config.kv_num_layers = info.num_layers;
  updated_config.kv_num_heads = info.num_kv_heads;
  updated_config.kv_head_dim = info.hidden_size / info.num_heads;

  auto pager = create_pager(updated_config);
  if (!pager) {
    last_error_ = "Failed to create pager";
    return std::nullopt;
  }

  // STEP 6: Create CachedLlamaModel and load weights
  auto model = create_cached_model(loader, info, pager);
  if (!model) {
    return std::nullopt;
  }

  // STEP 7: Create Engine
  runtime::GenerationConfig gen_config;
  gen_config.max_seq_len = info.context_length;
  gen_config.max_new_tokens = config.max_new_tokens;
  gen_config.use_cached_attention = config.use_cached_attention;
  gen_config.kv_block_size = config.kv_block_size;
  gen_config.kv_num_blocks = config.kv_num_blocks;

  auto engine =
      std::make_shared<runtime::Engine>(model, pager, tokenizer, gen_config);

  // STEP 8: Package result
  LoadedModel result;
  result.model = model;
  result.pager = pager;
  result.tokenizer = tokenizer;
  result.engine = engine;
  result.info = info;
  result.loader = loader;  // Keep alive for mmap
  result.config = gen_config;

  // Update registry
  registry_->touch_model(info.id);
  registry_->set_model_loaded(info.id, true);

  std::cerr << "[ModelLoader] Successfully loaded model: " << info.name
            << std::endl;

  return result;
}

std::optional<LoadedModel> ModelLoader::load_model_by_id(
    int64_t model_id, const LoadModelConfig& config) {
  // Get model info from registry by ID
  auto model_info_opt = registry_->get_model(model_id);
  if (!model_info_opt.has_value()) {
    last_error_ = "Model not found with ID: " + std::to_string(model_id);
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  // Delegate to load_model() with model_id string
  return load_model(model_info_opt->model_id, config);
}

}  // namespace server
}  // namespace mlxr
