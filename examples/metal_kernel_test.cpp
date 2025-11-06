/**
 * @file metal_kernel_test.cpp
 * @brief Test Metal attention kernels with CachedLlamaModel
 *
 * This validates that the Metal attention kernels are properly invoked
 * and provides performance measurements.
 *
 * Usage:
 *   ./metal_kernel_test <model_dir> <tokenizer_path> "<prompt>"
 *
 * Example:
 *   ./metal_kernel_test ~/models/llm/tinyllama-1.1b \
 *                       ~/models/llm/tinyllama-1.1b/tokenizer.model \
 *                       "Once upon a time"
 */

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "graph/attention_cached.h"
#include "graph/model.h"
#include "graph/tensor.h"
#include "mlx/mlx.h"
#include "runtime/kv/arena.h"
#include "runtime/kv/pager.h"
#include "runtime/sampler.h"
#include "runtime/tokenizer/tokenizer.h"

using namespace mlxr;
using namespace mlxr::graph;
using namespace mlxr::runtime;
using namespace std::chrono;

void print_separator() { std::cout << std::string(80, '=') << std::endl; }

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <tokenizer_path> \"<prompt>\"" << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << argv[0] << " ~/models/llm/tinyllama-1.1b"
              << " ~/models/llm/tinyllama-1.1b/tokenizer.model"
              << " \"Once upon a time\"" << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string tokenizer_path = argv[2];
  std::string prompt = argv[3];

  print_separator();
  std::cout << "MLXR Metal Kernel Validation Test" << std::endl;
  print_separator();
  std::cout << "Model directory: " << model_dir << std::endl;
  std::cout << "Tokenizer: " << tokenizer_path << std::endl;
  std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
  std::cout << std::endl;

  // Load model config
  std::cout << "Loading model configuration..." << std::endl;
  auto config = ModelConfig::from_hf_config(model_dir + "/config.json");

  // Create KV cache arena and pager
  std::cout << "Initializing KV cache system..." << std::endl;
  kv::ArenaConfig arena_config;
  arena_config.block_size_tokens = 32;
  arena_config.num_blocks = 256;
  arena_config.num_layers = config.num_layers;
  arena_config.num_kv_heads = config.num_kv_heads;
  arena_config.head_dim = config.hidden_size / config.num_heads;

  auto arena = std::make_shared<kv::Arena>(arena_config);
  auto pager = std::make_shared<kv::Pager>(arena);

  // Create cached model with Metal kernels enabled
  std::cout << "Creating CachedLlamaModel (Metal kernels enabled)..."
            << std::endl;
  auto model = std::make_shared<CachedLlamaModel>(config, pager);

  // Load weights
  std::cout << "Loading model weights..." << std::endl;
  if (!model->load_weights_from_dir(model_dir)) {
    std::cerr << "Failed to load model weights" << std::endl;
    return 1;
  }

  // Load tokenizer
  std::cout << "Loading tokenizer..." << std::endl;
  auto tokenizer = std::make_unique<SentencePieceTokenizer>(tokenizer_path);

  print_separator();
  std::cout << "Encoding prompt..." << std::endl;
  auto input_ids = tokenizer->encode(prompt);
  std::cout << "Prompt tokens: " << input_ids.size() << std::endl;

  // Create sequence for this generation
  int seq_id = 0;
  if (!pager->create_sequence(seq_id)) {
    std::cerr << "Failed to create sequence" << std::endl;
    return 1;
  }
  std::cout << "Created sequence ID: " << seq_id << std::endl;

  print_separator();
  std::cout << "Running PREFILL with Metal kernels..." << std::endl;
  print_separator();

  // Convert to tensor [batch=1, seq_len]
  int seq_len = static_cast<int>(input_ids.size());
  auto input_arr =
      mlx::core::array(input_ids.data(), {1, seq_len}, mlx::core::int32);
  Tensor input_tensor(input_arr);

  // Prefill forward pass
  auto prefill_start = high_resolution_clock::now();
  auto logits = model->forward(input_tensor, seq_id, 0);
  mlx::core::eval(logits.array());  // Force evaluation
  auto prefill_end = high_resolution_clock::now();
  auto prefill_ms =
      duration_cast<milliseconds>(prefill_end - prefill_start).count();

  std::cout << "\nPrefill latency: " << prefill_ms << " ms" << std::endl;

  // Sample first token
  SamplerConfig sampler_config;
  sampler_config.temperature = 0.7f;
  sampler_config.top_p = 0.9f;
  Sampler sampler(sampler_config);

  // Get last logits
  auto logits_arr = logits.array();
  auto last_logits =
      mlx::core::slice(logits_arr, {0, seq_len - 1, 0},
                       {1, seq_len, logits.shape()[2]}, {1, 1, 1});
  auto vocab_size = logits.shape()[2];
  auto last_logits_reshaped = mlx::core::reshape(last_logits, {vocab_size});
  Tensor last_logits_tensor(last_logits_reshaped);

  int token = sampler.sample(last_logits_tensor, input_ids);
  std::cout << "\nGenerated tokens: ";
  std::cout << tokenizer->decode({token}) << std::flush;

  print_separator();
  std::cout << "Running DECODE with Metal kernels..." << std::endl;
  print_separator();

  // Decode loop (generate 10 more tokens)
  std::vector<long long> decode_latencies;
  int num_decode_tokens = 10;

  for (int i = 0; i < num_decode_tokens; ++i) {
    // Create single token input [batch=1, seq_len=1]
    auto decode_input_arr = mlx::core::array(&token, {1, 1}, mlx::core::int32);
    Tensor decode_input(decode_input_arr);

    int start_pos = seq_len + i;

    auto decode_start = high_resolution_clock::now();
    auto decode_logits = model->forward(decode_input, seq_id, start_pos);
    mlx::core::eval(decode_logits.array());  // Force evaluation
    auto decode_end = high_resolution_clock::now();

    auto decode_ms =
        duration_cast<milliseconds>(decode_end - decode_start).count();
    decode_latencies.push_back(decode_ms);

    // Sample next token
    auto decode_logits_arr = decode_logits.array();
    auto decode_last =
        mlx::core::slice(decode_logits_arr, {0, 0, 0},
                         {1, 1, decode_logits.shape()[2]}, {1, 1, 1});
    auto decode_reshaped = mlx::core::reshape(decode_last, {vocab_size});
    Tensor decode_logits_tensor(decode_reshaped);

    input_ids.push_back(token);
    token = sampler.sample(decode_logits_tensor, input_ids);

    std::cout << tokenizer->decode({token}) << std::flush;
  }

  std::cout << std::endl;

  // Calculate statistics
  long long total_decode = 0;
  long long min_decode = decode_latencies[0];
  long long max_decode = decode_latencies[0];

  for (auto ms : decode_latencies) {
    total_decode += ms;
    if (ms < min_decode) min_decode = ms;
    if (ms > max_decode) max_decode = ms;
  }

  double avg_decode =
      static_cast<double>(total_decode) / decode_latencies.size();

  print_separator();
  std::cout << "Performance Results" << std::endl;
  print_separator();
  std::cout << "Prefill:" << std::endl;
  std::cout << "  Latency: " << prefill_ms << " ms (" << seq_len << " tokens)"
            << std::endl;
  std::cout << "  Per-token: " << (prefill_ms / (double)seq_len) << " ms/token"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Decode (" << decode_latencies.size()
            << " tokens):" << std::endl;
  std::cout << "  Min: " << min_decode << " ms" << std::endl;
  std::cout << "  Max: " << max_decode << " ms" << std::endl;
  std::cout << "  Avg: " << avg_decode << " ms/token" << std::endl;
  std::cout << "  Throughput: " << (1000.0 / avg_decode) << " tokens/sec"
            << std::endl;

  print_separator();
  std::cout << "KV Cache Statistics" << std::endl;
  print_separator();
  auto stats = arena->get_stats();
  std::cout << "  Total blocks: " << stats.total_blocks << std::endl;
  std::cout << "  Allocated blocks: " << stats.allocated_blocks << std::endl;
  std::cout << "  Free GPU blocks: " << stats.free_gpu_blocks << std::endl;
  std::cout << "  GPU memory: " << (stats.gpu_memory_bytes / 1024.0 / 1024.0)
            << " MB" << std::endl;

  print_separator();

  // Check if Metal kernels were actually used
  std::cout << "\nNOTE: Check the logs above for:" << std::endl;
  std::cout << "  [AttentionCached] PREFILL: Using Metal kernel path"
            << std::endl;
  std::cout << "  [AttentionCached] DECODE: Using Metal kernel path"
            << std::endl;
  std::cout << "\nIf you see these messages, Metal kernels are working! ✅"
            << std::endl;

  print_separator();

  // Cleanup
  pager->delete_sequence(seq_id);

  return 0;
}
