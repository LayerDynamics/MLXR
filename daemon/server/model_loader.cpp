// Copyright © 2025 MLXR Development
// Model loading utility implementation

#include "model_loader.h"

#include <iostream>

#include "../../core/graph/attention_cached.h"
#include "../../core/runtime/kv/arena.h"
#include "../../core/runtime/kv/eviction.h"
#include "../../core/runtime/kv/pager.h"
#include "../../core/runtime/tokenizer/tokenizer.h"
#include "../registry/gguf_parser.h"

namespace mlxr {
namespace server {

ModelLoader::ModelLoader(std::shared_ptr<registry::ModelRegistry> registry)
    : registry_(registry) {}

std::optional<LoadedModel> ModelLoader::load_model(
    const std::string& model_name, const LoadModelConfig& config) {
  last_error_.clear();

  // Query registry for model metadata
  auto model_info_opt = registry_->get_model_by_name(model_name);
  if (!model_info_opt.has_value()) {
    last_error_ = "Model not found in registry: " + model_name;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  return load_model_by_id(model_info_opt->id, config);
}

std::optional<LoadedModel> ModelLoader::load_model_by_id(
    int64_t model_id, const LoadModelConfig& config) {
  last_error_.clear();

  // Get model metadata
  auto model_info_opt = registry_->get_model(model_id);
  if (!model_info_opt.has_value()) {
    last_error_ = "Model ID not found in registry: " + std::to_string(model_id);
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  const auto& info = model_info_opt.value();
  std::cout << "[ModelLoader] Loading model: " << info.name << " ("
            << info.model_id << ")" << std::endl;
  std::cout << "[ModelLoader]   Format: "
            << (info.format == registry::ModelFormat::GGUF
                    ? "GGUF"
                    : info.format == registry::ModelFormat::SAFETENSORS
                          ? "SAFETENSORS"
                          : "MLX_NATIVE")
            << std::endl;
  std::cout << "[ModelLoader]   File: " << info.file_path << std::endl;
  std::cout << "[ModelLoader]   Params: " << info.param_count << std::endl;
  std::cout << "[ModelLoader]   Context: " << info.context_length << std::endl;

  // Load weights via mmap
  auto loader =
      load_weights(info.file_path, config.prefetch_weights, config.lock_weights);
  if (!loader) {
    last_error_ = "Failed to load weights from: " + info.file_path;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  std::cout << "[ModelLoader] Weights loaded via mmap: "
            << loader->file_size() / (1024 * 1024) << " MB" << std::endl;

  // Load tokenizer
  auto tokenizer = load_tokenizer(info);
  if (!tokenizer) {
    last_error_ = "Failed to load tokenizer for: " + info.name;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  std::cout << "[ModelLoader] Tokenizer loaded: " << info.tokenizer_type
            << std::endl;

  LoadedModel result;
  result.info = info;
  result.loader = loader;
  result.tokenizer = tokenizer;

  // Create generation config
  result.config.max_new_tokens = config.max_new_tokens;
  result.config.max_seq_len = config.max_seq_len;
  result.config.use_cached_attention = config.use_cached_attention;
  result.config.kv_block_size = config.kv_block_size;
  result.config.kv_num_blocks = config.kv_num_blocks;

  if (config.use_cached_attention) {
    std::cout << "[ModelLoader] Creating paged KV cache..." << std::endl;

    // Create KV cache pager with model-specific config
    LoadModelConfig model_config = config;
    model_config.kv_num_layers = info.num_layers;
    model_config.kv_num_heads = info.num_kv_heads;
    model_config.kv_head_dim = info.hidden_size / info.num_heads;

    result.pager = create_pager(model_config);
    if (!result.pager) {
      last_error_ = "Failed to create KV cache pager";
      std::cerr << "[ModelLoader] " << last_error_ << std::endl;
      return std::nullopt;
    }

    std::cout << "[ModelLoader] Pager created: " << config.kv_num_blocks
              << " blocks x " << config.kv_block_size << " tokens/block = "
              << (config.kv_num_blocks * config.kv_block_size)
              << " total tokens" << std::endl;

    // Create CachedLlamaModel
    result.model = create_cached_model(loader, info, result.pager);
    if (!result.model) {
      last_error_ = "Failed to create CachedLlamaModel";
      std::cerr << "[ModelLoader] " << last_error_ << std::endl;
      return std::nullopt;
    }

    std::cout << "[ModelLoader] CachedLlamaModel created (with Metal kernels)"
              << std::endl;

    // Create engine with cached model
    result.engine = std::make_shared<runtime::Engine>(
        result.model, result.pager, tokenizer, result.config);

  } else {
    std::cout << "[ModelLoader] Using simple LlamaModel (no Metal kernels)"
              << std::endl;

    // Create simple LlamaModel
    auto simple_model = create_simple_model(loader, info);
    if (!simple_model) {
      last_error_ = "Failed to create LlamaModel";
      std::cerr << "[ModelLoader] " << last_error_ << std::endl;
      return std::nullopt;
    }

    // Create engine with simple model
    result.engine =
        std::make_shared<runtime::Engine>(simple_model, tokenizer, result.config);
  }

  std::cout << "[ModelLoader] Engine created successfully" << std::endl;

  // Update registry to mark model as loaded
  registry::ModelInfo updated_info = info;
  updated_info.is_loaded = true;
  updated_info.last_used_timestamp = std::time(nullptr);
  registry_->update_model(updated_info);

  std::cout << "[ModelLoader] Model loaded and ready for inference" << std::endl;

  return result;
}

std::shared_ptr<runtime::Tokenizer> ModelLoader::load_tokenizer(
    const registry::ModelInfo& info) {
  std::cout << "[ModelLoader] Loading tokenizer: " << info.tokenizer_type
            << std::endl;

  if (info.tokenizer_type == "sentencepiece") {
    auto tokenizer = std::make_shared<runtime::SentencePieceTokenizer>();
    if (!tokenizer->load(info.tokenizer_path)) {
      std::cerr << "[ModelLoader] Failed to load SentencePiece tokenizer from: "
                << info.tokenizer_path << std::endl;
      return nullptr;
    }
    return tokenizer;
  }

  // TODO: Add support for other tokenizer types (HuggingFace, tiktoken)
  if (info.tokenizer_type == "huggingface" || info.tokenizer_type == "hf") {
    std::cerr << "[ModelLoader] HuggingFace tokenizer not yet implemented"
              << std::endl;
    return nullptr;
  }

  if (info.tokenizer_type == "tiktoken") {
    std::cerr << "[ModelLoader] tiktoken tokenizer not yet implemented"
              << std::endl;
    return nullptr;
  }

  std::cerr << "[ModelLoader] Unknown tokenizer type: " << info.tokenizer_type
            << std::endl;
  return nullptr;
}

std::shared_ptr<runtime::kv::Pager> ModelLoader::create_pager(
    const LoadModelConfig& config) {
  // Create arena configuration
  runtime::kv::ArenaConfig arena_config;
  arena_config.block_size_tokens = config.kv_block_size;
  arena_config.max_blocks = config.kv_num_blocks;
  arena_config.num_layers = config.kv_num_layers;
  arena_config.num_kv_heads = config.kv_num_heads;
  arena_config.head_dim = config.kv_head_dim;
  arena_config.dtype_size = 2;  // FP16 = 2 bytes

  std::cout << "[ModelLoader] Arena config: " << config.kv_num_layers
            << " layers, " << config.kv_num_heads << " KV heads, "
            << config.kv_head_dim << " dim" << std::endl;

  // Create arena
  auto arena = std::make_shared<runtime::kv::Arena>(arena_config);
  if (!arena->initialize()) {
    std::cerr << "[ModelLoader] Failed to initialize KV cache arena"
              << std::endl;
    return nullptr;
  }

  // Create eviction policy (LRU)
  auto eviction_policy = std::make_shared<runtime::kv::LRUEvictionPolicy>(
      arena_config.max_blocks, config.kv_block_size);

  // Create pager
  auto pager = std::make_shared<runtime::kv::Pager>(arena, eviction_policy);

  std::cout << "[ModelLoader] Pager created with "
            << (arena_config.block_size_tokens * arena_config.max_blocks)
            << " token capacity" << std::endl;

  return pager;
}

std::shared_ptr<MMapWeightLoader> ModelLoader::load_weights(
    const std::string& file_path, bool prefetch, bool lock) {
  auto loader = std::make_shared<MMapWeightLoader>(file_path, /*read_only=*/true);

  if (!loader->initialize()) {
    std::cerr << "[ModelLoader] Failed to initialize weight loader for: "
              << file_path << std::endl;
    return nullptr;
  }

  // Check if this is a GGUF file and register tensors if so
  if (file_path.ends_with(".gguf") || file_path.ends_with(".GGUF")) {
    std::cout << "[ModelLoader] Detected GGUF format, parsing tensor metadata..."
              << std::endl;

    if (!load_gguf_tensors(loader, file_path)) {
      std::cerr << "[ModelLoader] Failed to parse GGUF file" << std::endl;
      return nullptr;
    }
  }

  // Map entire file (or specific tensors as needed)
  // For now, we map the entire file for simplicity
  auto region = loader->map_all(prefetch);
  if (!region.is_valid) {
    std::cerr << "[ModelLoader] Failed to mmap model file: " << file_path
              << std::endl;
    return nullptr;
  }

  std::cout << "[ModelLoader] Mapped " << region.size / (1024 * 1024)
            << " MB into memory" << std::endl;

  // Optionally lock in physical memory
  if (lock) {
    if (loader->lock_memory(region)) {
      std::cout << "[ModelLoader] Weights locked in physical memory"
                << std::endl;
    } else {
      std::cerr << "[ModelLoader] Warning: Failed to lock weights in memory"
                << std::endl;
    }
  }

  // Give advice for sequential access (good for initial load)
  loader->advise(region, MMapWeightLoader::AdvicePattern::SEQUENTIAL);

  return loader;
}

bool ModelLoader::load_gguf_tensors(std::shared_ptr<MMapWeightLoader> loader,
                                     const std::string& file_path) {
  // Parse GGUF file to get tensor metadata
  registry::GGUFFile gguf;
  if (!gguf.parse(file_path)) {
    std::cerr << "[ModelLoader] Failed to parse GGUF file: " << gguf.error()
              << std::endl;
    return false;
  }

  std::cout << "[ModelLoader] GGUF file parsed successfully:" << std::endl;
  std::cout << "[ModelLoader]   Version: " << gguf.header().version << std::endl;
  std::cout << "[ModelLoader]   Tensors: " << gguf.header().tensor_count
            << std::endl;
  std::cout << "[ModelLoader]   Architecture: " << gguf.get_arch() << std::endl;
  std::cout << "[ModelLoader]   Context length: " << gguf.get_context_length()
            << std::endl;
  std::cout << "[ModelLoader]   Attention heads: "
            << gguf.get_attention_head_count() << " ("
            << gguf.get_attention_head_count_kv() << " KV heads)" << std::endl;

  // Register each tensor with the weight loader
  const auto& tensors = gguf.tensors();
  uint64_t data_offset = gguf.data_offset();

  std::cout << "[ModelLoader] Registering " << tensors.size()
            << " tensors with weight loader..." << std::endl;

  for (const auto& tensor_info : tensors) {
    // Convert GGUF tensor info to WeightTensor format
    WeightTensor weight_tensor;
    weight_tensor.name = tensor_info.name;

    // Convert dimensions from uint64_t to int64_t
    weight_tensor.shape.reserve(tensor_info.dimensions.size());
    for (uint64_t dim : tensor_info.dimensions) {
      weight_tensor.shape.push_back(static_cast<int64_t>(dim));
    }

    // Calculate absolute file offset (data_offset + tensor offset)
    weight_tensor.file_offset = data_offset + tensor_info.offset;
    weight_tensor.data_size = tensor_info.size;

    // Convert GGUF type to dtype string
    weight_tensor.dtype = registry::gguf_type_to_mlx_dtype(tensor_info.type);

    // Set quantization info if applicable
    if (tensor_info.type >= registry::GGUFTensorType::Q4_0 &&
        tensor_info.type <= registry::GGUFTensorType::Q8_K) {
      weight_tensor.quant_type = registry::gguf_type_name(tensor_info.type);
      weight_tensor.quant_block_size =
          registry::gguf_block_size(tensor_info.type);
    }

    // Register tensor with loader
    loader->register_tensor(weight_tensor);
  }

  std::cout << "[ModelLoader] All tensors registered successfully" << std::endl;

  return true;
}

std::shared_ptr<graph::CachedLlamaModel> ModelLoader::create_cached_model(
    std::shared_ptr<MMapWeightLoader> loader, const registry::ModelInfo& info,
    std::shared_ptr<runtime::kv::Pager> pager) {
  // Create model configuration from registry info
  graph::LlamaConfig model_config;
  model_config.hidden_size = info.hidden_size;
  model_config.num_layers = info.num_layers;
  model_config.num_heads = info.num_heads;
  model_config.num_kv_heads = info.num_kv_heads;
  model_config.intermediate_size = info.intermediate_size;
  model_config.vocab_size = info.vocab_size;
  model_config.max_seq_len = info.context_length;
  model_config.norm_eps = 1e-5f;  // Default RMS norm epsilon
  model_config.rope_theta = info.rope_freq_base;

  std::cout << "[ModelLoader] Model config: " << model_config.num_layers
            << " layers, " << model_config.num_heads << " heads ("
            << model_config.num_kv_heads << " KV heads)" << std::endl;

  // Create CachedLlamaModel
  auto model = std::make_shared<graph::CachedLlamaModel>(model_config, pager);

  // Load weights based on format
  std::cout << "[ModelLoader] Loading weights from: " << info.file_path
            << std::endl;

  if (info.format == registry::ModelFormat::SAFETENSORS) {
    // For safetensors, use MLX's native loading
    if (!model->load_weights(info.file_path)) {
      std::cerr << "[ModelLoader] Failed to load safetensors weights"
                << std::endl;
      return nullptr;
    }
    std::cout << "[ModelLoader] Safetensors weights loaded successfully"
              << std::endl;

  } else if (info.format == registry::ModelFormat::GGUF) {
    // For GGUF, we need to create MLX arrays from mmap'd tensors
    std::cout << "[ModelLoader] Converting GGUF tensors to MLX arrays..."
              << std::endl;

    // Get list of registered tensors
    auto tensor_names = loader->list_tensors();
    std::cout << "[ModelLoader] Found " << tensor_names.size()
              << " tensors in GGUF file" << std::endl;

    // Create weight map from mmap'd tensors
    std::unordered_map<std::string, graph::Tensor> weight_map;

    for (const auto& name : tensor_names) {
      auto tensor_info_opt = loader->get_tensor_info(name);
      if (!tensor_info_opt.has_value()) {
        continue;
      }

      const auto& tensor_info = tensor_info_opt.value();

      // Map tensor to MLX array
      // TODO: For quantized tensors (q4_0, q5_k, etc.), need dequantization
      // For now, only support FP16/FP32 tensors
      if (tensor_info.dtype != "fp16" && tensor_info.dtype != "fp32") {
        std::cout << "[ModelLoader] Skipping quantized tensor: " << name
                  << " (type: " << tensor_info.dtype << ")" << std::endl;
        std::cout << "[ModelLoader] Quantized weight loading not yet "
                     "implemented"
                  << std::endl;
        std::cout
            << "[ModelLoader] Please convert model to FP16 safetensors format"
            << std::endl;
        return nullptr;
      }

      // Map the tensor region
      auto region = loader->map_tensor(name, /*prefetch=*/true);
      if (!region.is_valid) {
        std::cerr << "[ModelLoader] Failed to map tensor: " << name << std::endl;
        return nullptr;
      }

      // Create MLX array from mmap'd region
      // Note: This creates a copy since MLX arrays manage their own memory
      mlx::core::Dtype mlx_dtype =
          (tensor_info.dtype == "fp16") ? mlx::core::float16 : mlx::core::float32;

      auto mlx_array =
          mlx::core::array(region.data, tensor_info.shape, mlx_dtype);

      // Force evaluation to ensure data is copied
      mlx::core::eval(mlx_array);

      weight_map[name] = graph::Tensor(mlx_array);
    }

    std::cout << "[ModelLoader] Created " << weight_map.size()
              << " MLX arrays from GGUF tensors" << std::endl;

    // Assign weights to model
    // Note: CachedLlamaModel's load_from_weight_map handles HF name mapping
    if (!model->load_from_weight_map(weight_map)) {
      std::cerr << "[ModelLoader] Failed to assign weights to model"
                << std::endl;
      return nullptr;
    }

    std::cout << "[ModelLoader] GGUF weights loaded successfully" << std::endl;

  } else {
    std::cerr << "[ModelLoader] Unsupported model format" << std::endl;
    return nullptr;
  }

  return model;
}

std::shared_ptr<graph::LlamaModel> ModelLoader::create_simple_model(
    std::shared_ptr<MMapWeightLoader> loader, const registry::ModelInfo& info) {
  // Create model configuration
  graph::LlamaConfig model_config;
  model_config.hidden_size = info.hidden_size;
  model_config.num_layers = info.num_layers;
  model_config.num_heads = info.num_heads;
  model_config.num_kv_heads = info.num_kv_heads;
  model_config.intermediate_size = info.intermediate_size;
  model_config.vocab_size = info.vocab_size;
  model_config.max_seq_len = info.context_length;
  model_config.norm_eps = 1e-5f;
  model_config.rope_theta = info.rope_freq_base;

  // Create LlamaModel
  auto model = std::make_shared<graph::LlamaModel>(model_config);

  // Load weights based on format
  std::cout << "[ModelLoader] Loading weights for simple model from: "
            << info.file_path << std::endl;

  if (info.format == registry::ModelFormat::SAFETENSORS) {
    // For safetensors, use MLX's native loading
    if (!model->load_weights(info.file_path)) {
      std::cerr << "[ModelLoader] Failed to load safetensors weights"
                << std::endl;
      return nullptr;
    }
    std::cout << "[ModelLoader] Safetensors weights loaded successfully"
              << std::endl;

  } else if (info.format == registry::ModelFormat::GGUF) {
    std::cout << "[ModelLoader] GGUF format not yet supported for simple model"
              << std::endl;
    std::cout << "[ModelLoader] Please use safetensors format or enable cached "
                 "attention"
              << std::endl;
    return nullptr;

  } else {
    std::cerr << "[ModelLoader] Unsupported model format" << std::endl;
    return nullptr;
  }

  return model;
}

}  // namespace server
}  // namespace mlxr
