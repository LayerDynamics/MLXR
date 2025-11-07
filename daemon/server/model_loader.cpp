// Copyright © 2025 MLXR Development
// Model loading utility implementation

#include "model_loader.h"

#include <iostream>

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

std::optional<LoadedModel> ModelLoader::load_model(
    const std::string& model_name, const LoadModelConfig& config) {
  last_error_.clear();

  // Query registry for model metadata
  auto model_info_opt = registry_->get_model_by_identifier(model_name);
  if (!model_info_opt.has_value()) {
    last_error_ = "Model not found in registry: " + model_name;
    std::cerr << "[ModelLoader] " << last_error_ << std::endl;
    return std::nullopt;
  }

  const auto& info = model_info_opt.value();

  // TODO: Implement full model loading pipeline
  // This requires proper API integration with:
  // - Tokenizer loading (SentencePieceTokenizer constructor takes path)
  // - Arena creation and initialization
  // - Pager creation (constructor signature needs checking)
  // - Weight loading from GGUF/safetensors
  // - CachedLlamaModel creation with ModelConfig (not LlamaConfig)
  //
  // For now, return empty to allow compilation

  last_error_ = "Model loading not yet fully implemented";
  std::cerr << "[ModelLoader] " << last_error_ << std::endl;
  std::cerr << "[ModelLoader] Found model: " << info.name << " at "
            << info.file_path << std::endl;
  std::cerr << "[ModelLoader] Format: "
            << (info.format == registry::ModelFormat::GGUF ? "GGUF"
                                                            : "SAFETENSORS")
            << std::endl;

  (void)config;  // Suppress unused warning
  return std::nullopt;
}

// Note: last_error() is implemented inline in the header

}  // namespace server
}  // namespace mlxr
