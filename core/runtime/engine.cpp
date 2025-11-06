/**
 * @file engine.cpp
 * @brief Implementation of inference engine for text generation
 */

#include "engine.h"

#include <iostream>
#include <stdexcept>

#include "mlx/mlx.h"

namespace mlxr {
namespace runtime {

// Constructor for simple LlamaModel
Engine::Engine(std::shared_ptr<graph::LlamaModel> model,
               std::shared_ptr<Tokenizer> tokenizer,
               const GenerationConfig& config)
    : simple_model_(model),
      cached_model_(nullptr),
      pager_(nullptr),
      use_cached_(false),
      tokenizer_(tokenizer),
      config_(config),
      next_seq_id_(0) {
  if (!simple_model_) {
    throw std::invalid_argument("Model cannot be null");
  }
  if (!tokenizer_) {
    throw std::invalid_argument("Tokenizer cannot be null");
  }

  std::cout
      << "[Engine] Initialized with simple LlamaModel (basic concatenation)"
      << std::endl;
}

// Constructor for CachedLlamaModel
Engine::Engine(std::shared_ptr<graph::CachedLlamaModel> model,
               std::shared_ptr<kv::Pager> pager,
               std::shared_ptr<Tokenizer> tokenizer,
               const GenerationConfig& config)
    : simple_model_(nullptr),
      cached_model_(model),
      pager_(pager),
      use_cached_(true),
      tokenizer_(tokenizer),
      config_(config),
      next_seq_id_(0) {
  if (!cached_model_) {
    throw std::invalid_argument("CachedModel cannot be null");
  }
  if (!pager_) {
    throw std::invalid_argument("Pager cannot be null");
  }
  if (!tokenizer_) {
    throw std::invalid_argument("Tokenizer cannot be null");
  }

  std::cout << "[Engine] Initialized with CachedLlamaModel (zero-copy Metal "
               "attention)"
            << std::endl;
}

void Engine::allocate_cache_sequence(InferenceCache* cache, int num_tokens) {
  if (!use_cached_ || !pager_) {
    return;  // Not using cached model, no sequence allocation needed
  }

  // Allocate new sequence ID
  cache->seq_id = next_seq_id_++;

  // Create sequence in pager
  if (!pager_->create_sequence(cache->seq_id)) {
    throw std::runtime_error("Failed to create sequence " +
                             std::to_string(cache->seq_id));
  }

  // Allocate blocks for the sequence
  if (!pager_->allocate_blocks_for_sequence(cache->seq_id, num_tokens)) {
    pager_->delete_sequence(cache->seq_id);
    throw std::runtime_error("Failed to allocate blocks for sequence " +
                             std::to_string(cache->seq_id));
  }
}

void Engine::release_cache_sequence(InferenceCache* cache) {
  if (!use_cached_ || !pager_ || cache->seq_id < 0) {
    return;  // Not using cached model or no sequence allocated
  }

  pager_->delete_sequence(cache->seq_id);
  cache->seq_id = -1;
}

std::vector<int> Engine::encode(const std::string& text) {
  return tokenizer_->encode(text);
}

std::string Engine::decode(const std::vector<int>& token_ids) {
  return tokenizer_->decode(token_ids);
}

graph::Tensor Engine::forward(const std::vector<int>& input_ids) {
  if (use_cached_) {
    throw std::runtime_error(
        "forward() without cache not supported with CachedLlamaModel. Use "
        "forward_prefill/forward_decode instead.");
  }

  // Convert input_ids to tensor
  // Shape: [batch=1, seq_len]
  int seq_len = static_cast<int>(input_ids.size());

  // Create MLX array from input_ids
  auto input_arr =
      mlx::core::array(input_ids.begin(), {1, seq_len}, mlx::core::int32);
  graph::Tensor input_tensor(input_arr);

  // Forward pass through simple model (no cache)
  auto logits = simple_model_->forward(input_tensor, nullptr, nullptr);

  // Get logits for last position
  // logits shape: [batch=1, seq_len, vocab_size]
  // We want: [vocab_size]
  auto logits_arr = logits.array();

  // Force evaluation before slicing to ensure logits are computed
  mlx::core::eval(logits_arr);

  // Slice to get last position: [1, vocab_size]
  auto last_logits =
      mlx::core::slice(logits_arr, {0, seq_len - 1, 0},
                       {1, seq_len, logits.shape()[2]}, {1, 1, 1});

  // Reshape to [vocab_size]
  auto vocab_size = logits.shape()[2];
  auto last_logits_reshaped = mlx::core::reshape(last_logits, {vocab_size});

  return graph::Tensor(last_logits_reshaped);
}

graph::Tensor Engine::forward_prefill(const std::vector<int>& input_ids,
                                      InferenceCache* cache) {
  if (!cache) {
    throw std::invalid_argument("Cache cannot be null for forward_prefill");
  }

  // Convert input_ids to tensor [batch=1, seq_len]
  int seq_len = static_cast<int>(input_ids.size());
  auto input_arr =
      mlx::core::array(input_ids.begin(), {1, seq_len}, mlx::core::int32);
  graph::Tensor input_tensor(input_arr);

  graph::Tensor logits;

  if (use_cached_) {
    // Cached model path: Use paged KV cache with Metal kernels

    // Allocate sequence in pager
    allocate_cache_sequence(cache, seq_len);

    // Forward pass through cached model
    // start_pos=0 for prefill (starting from beginning)
    logits = cached_model_->forward(input_tensor, cache->seq_id, 0, nullptr);

    // Update cache metadata
    auto* seq = pager_->get_sequence(cache->seq_id);
    if (seq) {
      seq->set_num_tokens(seq_len);
      cache->cached_tokens = seq_len;
    }
  } else {
    // Simple model path: Use concatenation-based KV cache

    // Forward pass through simple model WITH KV cache
    // The model will populate the cache during this forward pass
    logits = simple_model_->forward(input_tensor, nullptr, &cache->kv_cache);

    // Update cache metadata
    cache->cached_tokens = cache->kv_cache.cached_length;
  }

  cache->initialized = true;

  // Extract logits for last position [vocab_size]
  auto logits_arr = logits.array();

  // Force evaluation before slicing to ensure logits are computed
  mlx::core::eval(logits_arr);

  auto last_logits =
      mlx::core::slice(logits_arr, {0, seq_len - 1, 0},
                       {1, seq_len, logits.shape()[2]}, {1, 1, 1});
  auto vocab_size = logits.shape()[2];
  auto last_logits_reshaped = mlx::core::reshape(last_logits, {vocab_size});

  return graph::Tensor(last_logits_reshaped);
}

graph::Tensor Engine::forward_decode(int token_id, InferenceCache* cache) {
  if (!cache) {
    throw std::invalid_argument("Cache cannot be null for forward_decode");
  }

  if (!cache->initialized) {
    throw std::runtime_error(
        "Cache not initialized - call forward_prefill first");
  }

  // Create input tensor for single token [batch=1, seq_len=1]
  std::vector<int> token_vec = {token_id};
  auto input_arr =
      mlx::core::array(token_vec.begin(), {1, 1}, mlx::core::int32);
  graph::Tensor input_tensor(input_arr);

  graph::Tensor logits;

  if (use_cached_) {
    // Cached model path: Use paged KV cache with Metal kernels

    // Get current sequence length (start_pos for this new token)
    int start_pos = cache->cached_tokens;

    // Allocate additional blocks if needed
    int new_num_tokens = start_pos + 1;
    if (!pager_->allocate_blocks_for_sequence(cache->seq_id, new_num_tokens)) {
      throw std::runtime_error("Failed to allocate blocks for decode");
    }

    // Forward pass through cached model
    logits =
        cached_model_->forward(input_tensor, cache->seq_id, start_pos, nullptr);

    // Update cache metadata
    auto* seq = pager_->get_sequence(cache->seq_id);
    if (seq) {
      seq->set_num_tokens(new_num_tokens);
      cache->cached_tokens = new_num_tokens;
    }
  } else {
    // Simple model path: Use concatenation-based KV cache

    // Forward pass through simple model WITH KV cache
    // The model will use existing cache and append this token's K,V
    logits = simple_model_->forward(input_tensor, nullptr, &cache->kv_cache);

    // Update cache metadata
    cache->cached_tokens = cache->kv_cache.cached_length;
  }

  // Extract logits for the single position [vocab_size]
  auto logits_arr = logits.array();

  // Force evaluation before slicing to ensure logits are computed
  mlx::core::eval(logits_arr);

  // Logits shape: [batch=1, seq_len=1, vocab_size]
  // Extract: [vocab_size]
  auto last_logits = mlx::core::slice(logits_arr, {0, 0, 0},
                                      {1, 1, logits.shape()[2]}, {1, 1, 1});
  auto vocab_size = logits.shape()[2];
  auto last_logits_reshaped = mlx::core::reshape(last_logits, {vocab_size});

  return graph::Tensor(last_logits_reshaped);
}

bool Engine::is_stop_token(int token_id, const std::vector<int>& stop_tokens) {
  return std::find(stop_tokens.begin(), stop_tokens.end(), token_id) !=
         stop_tokens.end();
}

std::vector<int> Engine::generate_tokens(const std::vector<int>& input_ids,
                                         const GenerationConfig* config) {
  // Use provided config or default
  const GenerationConfig& gen_config = config ? *config : config_;

  // Initialize generated tokens with prompt
  std::vector<int> generated = input_ids;

  // Add EOS token to stop tokens if not already present
  std::vector<int> stop_tokens = gen_config.stop_tokens;
  int eos_token = tokenizer_->eos_token_id();
  if (eos_token >= 0 && std::find(stop_tokens.begin(), stop_tokens.end(),
                                  eos_token) == stop_tokens.end()) {
    stop_tokens.push_back(eos_token);
  }

  // Create sampler
  Sampler sampler(gen_config.sampler_config);

  if (gen_config.verbose) {
    std::cout << "Starting generation (prompt length: " << input_ids.size()
              << " tokens)" << std::endl;
  }

  // Create cache for prefill/decode path (used with CachedLlamaModel)
  InferenceCache cache;
  bool use_cached_path = (cached_model_ != nullptr);

  // Generation loop
  for (int i = 0; i < gen_config.max_new_tokens; ++i) {
    // Check max sequence length
    if (static_cast<int>(generated.size()) >= gen_config.max_seq_len) {
      if (gen_config.verbose) {
        std::cout << "Reached max sequence length" << std::endl;
      }
      break;
    }

    // Forward pass - use prefill/decode path if CachedLlamaModel is available
    graph::Tensor logits;
    if (use_cached_path) {
      if (i == 0) {
        // First iteration: prefill with entire prompt
        logits = forward_prefill(generated, &cache);
      } else {
        // Subsequent iterations: decode with single token
        logits = forward_decode(generated.back(), &cache);
      }
    } else {
      // Fallback to simple forward (reprocesses entire sequence each time)
      logits = forward(generated);
    }

    // Sample next token
    int next_token = sampler.sample(logits, generated);

    // Add to generated sequence
    generated.push_back(next_token);

    // Print token if verbose
    if (gen_config.verbose) {
      std::string token_str = tokenizer_->id_to_token(next_token);
      std::cout << token_str << std::flush;
    }

    // Check for stop tokens
    if (is_stop_token(next_token, stop_tokens)) {
      if (gen_config.verbose) {
        std::cout << std::endl << "Hit stop token" << std::endl;
      }
      break;
    }
  }

  if (gen_config.verbose) {
    std::cout << std::endl
              << "Generated " << (generated.size() - input_ids.size())
              << " tokens" << std::endl;
  }

  // Return generated tokens (with or without prompt)
  if (!gen_config.echo_prompt && generated.size() > input_ids.size()) {
    return std::vector<int>(generated.begin() + input_ids.size(),
                            generated.end());
  }

  return generated;
}

std::string Engine::generate(const std::string& prompt,
                             const GenerationConfig* config) {
  // Encode prompt
  auto input_ids = encode(prompt);

  if (input_ids.empty()) {
    throw std::runtime_error("Failed to encode prompt");
  }

  // Generate tokens
  auto generated_ids = generate_tokens(input_ids, config);

  // Decode to text
  return decode(generated_ids);
}

std::unique_ptr<Engine> load_engine(const std::string& model_dir,
                                    const std::string& tokenizer_path,
                                    const GenerationConfig& config) {
  // Load tokenizer first
  std::shared_ptr<Tokenizer> tokenizer;
  try {
    tokenizer = create_tokenizer(tokenizer_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
    return nullptr;
  }

  // Create engine based on config flag
  try {
    if (config.use_cached_attention) {
      std::cout << "[load_engine] Creating CachedLlamaModel with paged KV cache"
                << std::endl;

      // Load model config to get architecture parameters
      auto simple_model = graph::load_llama_model(model_dir);
      if (!simple_model) {
        std::cerr << "Failed to load model from: " << model_dir << std::endl;
        return nullptr;
      }

      // Get model config for arena setup
      const auto& model_config = simple_model->config();
      std::cout << "[load_engine] Model config: " << model_config.num_layers
                << " layers, " << model_config.num_kv_heads << " KV heads, "
                << "head_dim="
                << (model_config.hidden_size / model_config.num_heads)
                << std::endl;

      // Create Arena for paged KV cache
      kv::ArenaConfig arena_config;
      arena_config.num_layers = model_config.num_layers;
      arena_config.num_kv_heads = model_config.num_kv_heads;
      arena_config.head_dim = model_config.hidden_size / model_config.num_heads;
      arena_config.block_size_tokens = config.kv_block_size;
      arena_config.num_blocks = config.kv_num_blocks;

      std::cout << "[load_engine] Creating Arena: " << arena_config.num_blocks
                << " blocks, " << arena_config.block_size_tokens
                << " tokens/block" << std::endl;
      auto arena = std::make_shared<kv::Arena>(arena_config);
      std::cout << "[load_engine] Arena created successfully" << std::endl;

      // Create Pager
      std::cout << "[load_engine] Creating Pager..." << std::endl;
      auto pager = std::make_shared<kv::Pager>(arena);
      std::cout << "[load_engine] Pager created successfully" << std::endl;

      // Create CachedLlamaModel
      std::cout << "[load_engine] Creating CachedLlamaModel..." << std::endl;
      auto cached_model =
          std::make_shared<graph::CachedLlamaModel>(model_config, pager);
      std::cout << "[load_engine] CachedLlamaModel created successfully"
                << std::endl;

      // Load weights into cached model
      std::cout << "[load_engine] Loading weights into CachedLlamaModel..."
                << std::endl;
      if (!cached_model->load_weights_from_dir(model_dir)) {
        std::cerr << "Failed to load weights from: " << model_dir << std::endl;
        return nullptr;
      }
      std::cout << "[load_engine] Weights loaded successfully" << std::endl;

      // Create Engine with cached model
      std::cout << "[load_engine] Creating Engine with CachedLlamaModel..."
                << std::endl;
      return std::make_unique<Engine>(cached_model, pager, tokenizer, config);

    } else {
      // Use simple LlamaModel
      auto model = graph::load_llama_model(model_dir);
      if (!model) {
        std::cerr << "Failed to load model from: " << model_dir << std::endl;
        return nullptr;
      }

      // Create Engine with simple model
      return std::make_unique<Engine>(std::move(model), tokenizer, config);
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to create engine: " << e.what() << std::endl;
    return nullptr;
  }
}

}  // namespace runtime
}  // namespace mlxr
