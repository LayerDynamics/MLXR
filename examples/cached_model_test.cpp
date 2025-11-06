/**
 * @file cached_model_test.cpp
 * @brief Test zero-copy optimization by running inference with verbose logging
 *
 * This test runs standard inference and checks logs for Metal kernel usage.
 * When zero-copy is working, you should see:
 * - [AttentionPrefill] logs during prefill
 * - [AttentionDecode] logs during decode
 * -[RMSNorm] logs throughout
 *
 * Usage:
 *   ./cached_model_test <model_dir> <tokenizer_path>
 *
 * Example:
 *   ./cached_model_test ~/models/llm/tinyllama-1.1b
 * ~/models/llm/tinyllama-1.1b/tokenizer.model
 */

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "graph/tensor.h"
#include "mlx/mlx.h"
#include "runtime/engine.h"
#include "runtime/sampler.h"

using namespace mlxr;
using namespace mlxr::runtime;
using namespace std::chrono;

void print_separator() { std::cout << std::string(80, '=') << std::endl; }

int main(int argc, char* argv[]) {
  // Parse command line arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_dir> <tokenizer_path>"
              << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << argv[0]
              << " ~/models/llm/tinyllama-1.1b "
                 "~/models/llm/tinyllama-1.1b/tokenizer.model"
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string tokenizer_path = argv[2];

  print_separator();
  std::cout << "MLXR Zero-Copy Optimization Verification" << std::endl;
  print_separator();
  std::cout << "Model directory: " << model_dir << std::endl;
  std::cout << "Tokenizer: " << tokenizer_path << std::endl;
  std::cout << std::endl;

  std::cout << "IMPORTANT: Watch the logs for Metal kernel usage:" << std::endl;
  std::cout << "  [AttentionPrefill] - Metal prefill kernel (should appear)"
            << std::endl;
  std::cout << "  [AttentionDecode] - Metal decode kernel (should appear)"
            << std::endl;
  std::cout << "  [RMSNorm] - Metal normalization kernel (should appear)"
            << std::endl;
  std::cout << "\nIf you see these logs, the zero-copy optimization is working!"
            << std::endl;
  print_separator();

  // Configure generation
  GenerationConfig config;
  config.max_new_tokens = 10;                // Generate 10 tokens
  config.sampler_config.temperature = 0.0f;  // Greedy for determinism
  config.sampler_config.top_p = 1.0f;
  config.sampler_config.top_k = 0;
  config.echo_prompt = false;
  config.verbose = true;      // Enable verbose logging to see kernel calls
  config.kv_num_blocks = 32;  // Reduce from 256 to 32 blocks for faster init

  // Load engine
  std::cout << "\nLoading model..." << std::endl;
  auto engine = load_engine(model_dir, tokenizer_path, config);

  if (!engine) {
    std::cerr << "Failed to load engine" << std::endl;
    return 1;
  }

  std::cout << "Model loaded successfully!" << std::endl;
  print_separator();

  // Test prompt
  std::string prompt = "The quick brown fox";
  std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
  std::cout << "Generating " << config.max_new_tokens << " tokens...\n"
            << std::endl;
  print_separator();

  // Encode prompt
  auto input_ids = engine->encode(prompt);
  std::cout << "Encoded prompt: " << input_ids.size() << " tokens" << std::endl;

  // Run generation with Metal kernel logging
  InferenceCache cache;
  Sampler sampler(config.sampler_config);

  // Prefill pass
  std::cout << "\n" << std::string(80, '-') << std::endl;
  std::cout << "PREFILL PASS" << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  auto prefill_start = high_resolution_clock::now();
  auto prefill_logits = engine->forward_prefill(input_ids, &cache);
  auto prefill_end = high_resolution_clock::now();
  auto prefill_ms =
      duration_cast<milliseconds>(prefill_end - prefill_start).count();

  std::cout << "\n✓ Prefill completed in " << prefill_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << (1000.0 * input_ids.size() / prefill_ms)
            << " tokens/sec" << std::endl;

  // Sample first token
  int next_token = sampler.sample(prefill_logits);
  std::vector<int> generated_ids = {next_token};

  // Decode passes
  std::cout << "\n" << std::string(80, '-') << std::endl;
  std::cout << "DECODE PASS" << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  std::vector<double> decode_latencies;

  std::cout << "\nGenerated text: ";
  std::cout << engine->decode({next_token});
  std::cout.flush();

  for (int i = 1; i < config.max_new_tokens; ++i) {
    auto decode_start = high_resolution_clock::now();
    auto logits = engine->forward_decode(next_token, &cache);
    auto decode_end = high_resolution_clock::now();

    auto decode_us =
        duration_cast<microseconds>(decode_end - decode_start).count();
    decode_latencies.push_back(decode_us / 1000.0);  // Convert to ms

    next_token = sampler.sample(logits);
    generated_ids.push_back(next_token);

    // Print token
    std::cout << engine->decode({next_token});
    std::cout.flush();
  }

  std::cout << std::endl;

  // Report performance metrics
  print_separator();
  std::cout << "PERFORMANCE METRICS" << std::endl;
  print_separator();

  std::cout << "Prefill:" << std::endl;
  std::cout << "  Latency: " << prefill_ms << " ms" << std::endl;
  std::cout << "  Tokens: " << input_ids.size() << std::endl;
  std::cout << "  Throughput: " << (1000.0 * input_ids.size() / prefill_ms)
            << " tokens/sec" << std::endl;

  if (!decode_latencies.empty()) {
    double sum = 0.0;
    double min_lat = decode_latencies[0];
    double max_lat = decode_latencies[0];

    for (double lat : decode_latencies) {
      sum += lat;
      min_lat = std::min(min_lat, lat);
      max_lat = std::max(max_lat, lat);
    }

    double avg_lat = sum / decode_latencies.size();

    std::cout << "\nDecode:" << std::endl;
    std::cout << "  Average: " << avg_lat << " ms/token" << std::endl;
    std::cout << "  Min: " << min_lat << " ms/token" << std::endl;
    std::cout << "  Max: " << max_lat << " ms/token" << std::endl;
    std::cout << "  Throughput: " << (1000.0 / avg_lat) << " tokens/sec"
              << std::endl;
    std::cout << "  Tokens generated: " << generated_ids.size() << std::endl;
  }

  // Expected performance
  print_separator();
  std::cout << "EXPECTED PERFORMANCE (TinyLlama on M4)" << std::endl;
  print_separator();
  std::cout << "With zero-copy optimization:" << std::endl;
  std::cout << "  Prefill: ~200-250 ms (target: match simple Attention)"
            << std::endl;
  std::cout << "  Decode: ~40-50 ms/token" << std::endl;
  std::cout << "  Throughput: ~20-25 tokens/sec" << std::endl;
  std::cout << "\nBaseline (simple Attention):" << std::endl;
  std::cout << "  Prefill: ~198 ms" << std::endl;
  std::cout << "  Decode: ~53 ms/token" << std::endl;
  std::cout << "  Throughput: ~18.87 tokens/sec" << std::endl;

  print_separator();
  std::cout << "VERIFICATION" << std::endl;
  print_separator();
  std::cout << "Check the logs above for Metal kernel messages:" << std::endl;
  std::cout << "  ✓ [AttentionPrefill] - Indicates Metal prefill kernel"
            << std::endl;
  std::cout << "  ✓ [AttentionDecode] - Indicates Metal decode kernel"
            << std::endl;
  std::cout << "  ✓ [RMSNorm] - Indicates Metal RMSNorm kernel" << std::endl;
  std::cout << "\nIf these appear, zero-copy optimization is active!"
            << std::endl;
  std::cout << "If only [RMSNorm] appears, CachedAttention is not being used."
            << std::endl;
  print_separator();

  return 0;
}
