/**
 * @file simple_generation.cpp
 * @brief Simple example of text generation with MLXR
 *
 * Usage:
 *   ./simple_generation <model_dir> <tokenizer_path> <prompt>
 *
 * Example:
 *   ./simple_generation ./models/TinyLlama-1.1B ./models/tokenizer.model "Once
 * upon a time"
 */

#include <iostream>
#include <string>

#include "runtime/engine.h"

int main(int argc, char* argv[]) {
  // Parse command line arguments
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <tokenizer_path> <prompt>" << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << argv[0]
              << " ./models/TinyLlama-1.1B ./models/tokenizer.model \"Once "
                 "upon a time\""
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string tokenizer_path = argv[2];
  std::string prompt = argv[3];

  std::cout << "=== MLXR Simple Generation Example ===" << std::endl;
  std::cout << "Model directory: " << model_dir << std::endl;
  std::cout << "Tokenizer: " << tokenizer_path << std::endl;
  std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
  std::cout << std::endl;

  // Configure generation
  mlxr::runtime::GenerationConfig config;
  config.max_new_tokens = 50;
  config.sampler_config.temperature = 0.7f;
  config.sampler_config.top_p = 0.9f;
  config.echo_prompt = true;
  config.verbose = true;

  // Load engine
  std::cout << "Loading model..." << std::endl;
  auto engine = mlxr::runtime::load_engine(model_dir, tokenizer_path, config);

  if (!engine) {
    std::cerr << "Failed to load engine" << std::endl;
    return 1;
  }

  std::cout << "Model loaded successfully!" << std::endl;
  std::cout << std::endl;

  // Generate text
  std::cout << "Generating..." << std::endl;
  std::cout << "---" << std::endl;

  try {
    std::string generated = engine->generate(prompt);
    std::cout << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "\nGenerated text:" << std::endl;
    std::cout << generated << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Generation failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
