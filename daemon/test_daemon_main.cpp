// Test daemon main - HTTP server with scheduler integration
// This is a test file to verify the full inference pipeline

#include <dirent.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

#include "../core/graph/model.h"
#include "../core/runtime/engine.h"
#include "../core/runtime/tokenizer/tokenizer.h"
#include "registry/model_registry.h"
#include "scheduler/scheduler.h"
#include "server/rest_server.h"
#include "server/scheduler_worker.h"

using namespace mlxr;
using namespace mlxr::server;

std::atomic<bool> keep_running{true};

void signal_handler(int signal) {
  std::cout << "\nReceived signal " << signal << ", shutting down..."
            << std::endl;
  keep_running = false;
}

// Helper function to check if a path is a directory
bool is_directory(const std::string& path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

// Helper function to check if a file exists
bool file_exists(const std::string& path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

// Helper function to get file size
uint64_t get_file_size(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return st.st_size;
  }
  return 0;
}

// Scan models directory and register discovered models
void scan_and_register_models(registry::ModelRegistry& registry,
                              const std::string& models_dir) {
  std::cout << "Scanning models directory: " << models_dir << std::endl;

  if (!is_directory(models_dir)) {
    std::cerr << "Models directory does not exist: " << models_dir << std::endl;
    return;
  }

  DIR* dir = opendir(models_dir.c_str());
  if (!dir) {
    std::cerr << "Failed to open models directory" << std::endl;
    return;
  }

  int models_found = 0;
  struct dirent* entry;

  while ((entry = readdir(dir)) != nullptr) {
    std::string model_name = entry->d_name;

    // Skip . and ..
    if (model_name == "." || model_name == "..") {
      continue;
    }

    std::string model_path = models_dir + "/" + model_name;

    // Check if it's a directory
    if (!is_directory(model_path)) {
      continue;
    }

    // Look for model files (safetensors or GGUF)
    std::string safetensors_path = model_path + "/model.safetensors";
    std::string gguf_path = model_path + "/ggml-model-f16.gguf";

    registry::ModelInfo info;
    info.name = model_name;
    info.model_id = model_name;
    info.architecture = registry::ModelArchitecture::LLAMA;  // Default to Llama

    // Check for safetensors format
    if (file_exists(safetensors_path)) {
      info.file_path = safetensors_path;
      info.format = registry::ModelFormat::SAFETENSORS;
      info.file_size = get_file_size(safetensors_path);
      info.quant_type =
          registry::QuantizationType::NONE;  // Safetensors usually FP16
    }
    // Check for GGUF format
    else if (file_exists(gguf_path)) {
      info.file_path = gguf_path;
      info.format = registry::ModelFormat::GGUF;
      info.file_size = get_file_size(gguf_path);
      info.quant_type = registry::QuantizationType::Q4_K;  // Default assumption
    } else {
      // No recognized model file found
      continue;
    }

    // Look for tokenizer
    std::string tokenizer_path = model_path + "/tokenizer.model";
    if (file_exists(tokenizer_path)) {
      info.tokenizer_type = "sentencepiece";
      info.tokenizer_path = tokenizer_path;
    }

    // Set some defaults
    info.param_count =
        1100000000;  // ~1.1B, will be updated when model is loaded
    info.context_length = 2048;
    info.hidden_size = 2048;
    info.num_layers = 22;
    info.num_heads = 32;
    info.num_kv_heads = 4;
    info.intermediate_size = 5632;
    info.vocab_size = 32000;
    info.rope_freq_base = 10000.0f;
    info.rope_scale = 1.0f;
    info.rope_scaling_type = "none";
    info.description = "Discovered model: " + model_name;
    info.is_loaded = false;

    // Register the model
    int64_t model_id = registry.register_model(info);
    if (model_id >= 0) {
      std::cout << "  ✓ Registered model: " << model_name << " (id=" << model_id
                << ", format="
                << (info.format == registry::ModelFormat::SAFETENSORS
                        ? "safetensors"
                        : "gguf")
                << ")" << std::endl;
      models_found++;
    } else {
      std::cerr << "  ✗ Failed to register model: " << model_name << std::endl;
    }
  }

  closedir(dir);

  std::cout << "Model scan complete. Found " << models_found << " model(s)"
            << std::endl;
}

int main() {
  // Setup signal handlers for graceful shutdown
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  std::cout << "Starting MLXR Test Daemon..." << std::endl;

  // Get home directory
  const char* home = std::getenv("HOME");
  if (!home) {
    std::cerr << "HOME environment variable not set" << std::endl;
    return 1;
  }

  // Initialize model registry
  std::cout << "Initializing model registry..." << std::endl;
  std::string registry_dir =
      std::string(home) + "/Library/Application Support/MLXRunner";
  std::string registry_path = registry_dir + "/models.db";

  // Create registry directory if it doesn't exist
  system(("mkdir -p \"" + registry_dir + "\"").c_str());

  auto registry =
      std::make_shared<registry::ModelRegistry>(registry_path, true);

  if (!registry->initialize()) {
    std::cerr << "Failed to initialize model registry" << std::endl;
    return 1;
  }
  std::cout << "Model registry initialized at: " << registry_path << std::endl;

  // Scan and register models from disk
  std::string models_dir = std::string(home) + "/models/llm";
  scan_and_register_models(*registry, models_dir);

  // Display registered models
  auto registered_models = registry->list_models();
  std::cout << "\nRegistered models (" << registered_models.size()
            << " total):" << std::endl;
  for (const auto& model_info : registered_models) {
    std::cout << "  - " << model_info.name << " (" << model_info.file_path
              << ")" << std::endl;
  }
  std::cout << std::endl;

  // Create scheduler
  std::cout << "Initializing scheduler..." << std::endl;
  scheduler::SchedulerConfig sched_config;
  sched_config.max_batch_tokens = 4096;
  sched_config.max_batch_size = 64;
  auto scheduler = std::make_shared<scheduler::Scheduler>(sched_config);

  // Load model and tokenizer
  std::cout << "Loading TinyLlama model..." << std::endl;
  std::string model_dir = std::string(home) + "/models/llm/tinyllama-1.1b";
  std::string tokenizer_path = model_dir + "/tokenizer.model";

  // Load model
  auto model = graph::load_llama_model(model_dir);
  if (!model) {
    std::cerr << "Failed to load model. Running in mock mode (no inference)."
              << std::endl;
  }

  // Load tokenizer
  std::shared_ptr<runtime::Tokenizer> tokenizer;
  try {
    tokenizer = runtime::create_tokenizer(tokenizer_path);
    std::cout << "Tokenizer loaded successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
    std::cerr << "Running in mock mode (no inference)." << std::endl;
  }

  // Create engine if model and tokenizer are loaded
  std::shared_ptr<runtime::Engine> engine;
  if (model && tokenizer) {
    runtime::GenerationConfig gen_config;
    gen_config.max_new_tokens = 512;
    gen_config.max_seq_len = 2048;
    gen_config.sampler_config.temperature = 0.7f;
    gen_config.sampler_config.top_p = 0.9f;
    gen_config.verbose = false;

    engine = std::make_shared<runtime::Engine>(std::move(model), tokenizer,
                                               gen_config);
    std::cout << "Inference engine created successfully!" << std::endl;
  } else {
    engine = nullptr;
    std::cout << "Note: Running without loaded model (mock mode)" << std::endl;
  }

  // Create scheduler worker thread
  std::cout << "Starting scheduler worker..." << std::endl;
  auto worker = std::make_unique<SchedulerWorker>(scheduler, engine);
  worker->start();

  // Create REST server configuration
  ServerConfig config;
  config.bind_address = "127.0.0.1";
  config.port = 11434;  // Ollama-compatible default port
  config.enable_cors = true;
  config.api_key = "";  // No auth for testing

  // Create and initialize REST server
  RestServer server(config);

  // Wire scheduler, tokenizer, engine, and registry to server
  server.set_scheduler(scheduler);
  server.set_registry(registry);
  if (tokenizer) {
    server.set_tokenizer(tokenizer);
  }
  if (engine) {
    server.set_engine(engine);
  }

  std::cout << "Initializing HTTP server..." << std::endl;
  if (!server.initialize()) {
    std::cerr << "Failed to initialize REST server" << std::endl;
    return 1;
  }

  std::cout << "Starting HTTP server on " << config.bind_address << ":"
            << config.port << std::endl;

  if (!server.start()) {
    std::cerr << "Failed to start REST server" << std::endl;
    return 1;
  }

  std::cout << "HTTP server started successfully!" << std::endl;
  std::cout << "Scheduler worker running in background" << std::endl;
  std::cout << "\nTest endpoints:" << std::endl;
  std::cout << "  GET  http://127.0.0.1:11434/health" << std::endl;
  std::cout << "  GET  http://127.0.0.1:11434/v1/models" << std::endl;
  std::cout << "  POST http://127.0.0.1:11434/v1/chat/completions" << std::endl;
  std::cout << "\nPress Ctrl+C to stop..." << std::endl;

  // Wait for shutdown signal
  while (keep_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "Stopping server..." << std::endl;
  server.stop();

  std::cout << "Stopping scheduler worker..." << std::endl;
  worker->stop();

  std::cout << "Shutting down scheduler..." << std::endl;
  scheduler->shutdown();

  std::cout << "Daemon stopped cleanly" << std::endl;
  return 0;
}
