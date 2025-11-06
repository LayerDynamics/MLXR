// Copyright © 2025 MLXR Development
// Ollama-compatible API endpoints

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlxr {

// Forward declarations
class LlamaModel;

namespace runtime {
class Tokenizer;
class Engine;
}  // namespace runtime

namespace registry {
class ModelRegistry;
}

using runtime::Engine;
using runtime::Tokenizer;

namespace server {

// Forward declaration
using StreamCallback = std::function<bool(const std::string& chunk)>;

// ==============================================================================
// Ollama Request/Response Data Structures
// ==============================================================================

// /api/generate request
struct OllamaGenerateRequest {
  std::string model;
  std::string prompt;
  std::optional<std::string> system;
  std::optional<std::string> template_;
  std::optional<std::string> context;
  std::optional<bool> stream;
  std::optional<bool> raw;
  std::optional<std::string> format;  // "json"

  // Model parameters
  std::optional<int> num_predict;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> top_k;
  std::optional<float> repeat_penalty;
  std::optional<int> seed;
  std::optional<std::vector<std::string>> stop;
};

// /api/generate response (non-streaming)
struct OllamaGenerateResponse {
  std::string model;
  std::string created_at;
  std::string response;
  bool done;
  std::optional<std::string> context;
  std::optional<int64_t> total_duration;
  std::optional<int64_t> load_duration;
  std::optional<int> prompt_eval_count;
  std::optional<int64_t> prompt_eval_duration;
  std::optional<int> eval_count;
  std::optional<int64_t> eval_duration;
};

// /api/chat message
struct OllamaChatMessage {
  std::string role;  // "system", "user", "assistant"
  std::string content;
  std::optional<std::vector<std::string>> images;  // Base64 encoded
};

// /api/chat request
struct OllamaChatRequest {
  std::string model;
  std::vector<OllamaChatMessage> messages;
  std::optional<bool> stream;
  std::optional<std::string> format;  // "json"

  // Model parameters
  std::optional<int> num_predict;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> top_k;
  std::optional<float> repeat_penalty;
  std::optional<int> seed;
  std::optional<std::vector<std::string>> stop;
};

// /api/chat response (non-streaming)
struct OllamaChatResponse {
  std::string model;
  std::string created_at;
  OllamaChatMessage message;
  bool done;
  std::optional<int64_t> total_duration;
  std::optional<int64_t> load_duration;
  std::optional<int> prompt_eval_count;
  std::optional<int64_t> prompt_eval_duration;
  std::optional<int> eval_count;
  std::optional<int64_t> eval_duration;
};

// /api/embeddings request
struct OllamaEmbeddingsRequest {
  std::string model;
  std::string prompt;
};

// /api/embeddings response
struct OllamaEmbeddingsResponse {
  std::vector<float> embedding;
};

// /api/pull request
struct OllamaPullRequest {
  std::string name;
  std::optional<bool> insecure;
  std::optional<bool> stream;
};

// /api/pull response (streaming)
struct OllamaPullResponse {
  std::string status;
  std::optional<std::string> digest;
  std::optional<int64_t> total;
  std::optional<int64_t> completed;
};

// /api/create request
struct OllamaCreateRequest {
  std::string name;
  std::optional<std::string> modelfile;
  std::optional<std::string> path;
  std::optional<bool> stream;
};

// /api/create response (streaming)
struct OllamaCreateResponse {
  std::string status;
};

// /api/tags response (model list)
struct OllamaModelInfo {
  std::string name;
  std::string modified_at;
  int64_t size;
  std::string digest;

  struct Details {
    std::string format;
    std::string family;
    std::vector<std::string> families;
    std::string parameter_size;
    std::string quantization_level;
  };

  std::optional<Details> details;
};

struct OllamaTagsResponse {
  std::vector<OllamaModelInfo> models;
};

// /api/ps response (running models)
struct OllamaRunningModel {
  std::string name;
  std::string model;
  int64_t size;
  std::string digest;

  struct Details {
    std::string format;
    std::string family;
    std::vector<std::string> families;
    std::string parameter_size;
    std::string quantization_level;
  };

  std::optional<Details> details;
  std::optional<std::string> expires_at;
  std::optional<int64_t> size_vram;
};

struct OllamaProcessResponse {
  std::vector<OllamaRunningModel> models;
};

// /api/show request
struct OllamaShowRequest {
  std::string name;
};

// /api/show response
struct OllamaShowResponse {
  std::string modelfile;
  std::string parameters;
  std::string template_;

  struct Details {
    std::string format;
    std::string family;
    std::vector<std::string> families;
    std::string parameter_size;
    std::string quantization_level;
  };

  std::optional<Details> details;
};

// /api/copy request
struct OllamaCopyRequest {
  std::string source;
  std::string destination;
};

// /api/delete request
struct OllamaDeleteRequest {
  std::string name;
};

// ==============================================================================
// Ollama API Handler
// ==============================================================================

class OllamaAPIHandler {
 public:
  explicit OllamaAPIHandler();
  ~OllamaAPIHandler();

  // Delete copy operations
  OllamaAPIHandler(const OllamaAPIHandler&) = delete;
  OllamaAPIHandler& operator=(const OllamaAPIHandler&) = delete;

  // Set model and inference components
  void set_model(std::shared_ptr<LlamaModel> model);
  void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer);
  void set_engine(std::shared_ptr<Engine> engine);
  void set_registry(std::shared_ptr<registry::ModelRegistry> registry);

  // Endpoint handlers
  std::string handle_generate(const std::string& json_request,
                              StreamCallback stream_callback = nullptr);
  std::string handle_chat(const std::string& json_request,
                          StreamCallback stream_callback = nullptr);
  std::string handle_embeddings(const std::string& json_request);
  std::string handle_pull(const std::string& json_request,
                          StreamCallback stream_callback = nullptr);
  std::string handle_create(const std::string& json_request,
                            StreamCallback stream_callback = nullptr);
  std::string handle_tags();
  std::string handle_ps();
  std::string handle_show(const std::string& json_request);
  std::string handle_copy(const std::string& json_request);
  std::string handle_delete(const std::string& json_request);

 private:
  // Model and inference components
  std::shared_ptr<LlamaModel> model_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Engine> engine_;
  std::shared_ptr<registry::ModelRegistry> registry_;

  // Request parsing
  std::optional<OllamaGenerateRequest> parse_generate_request(
      const std::string& json);
  std::optional<OllamaChatRequest> parse_chat_request(const std::string& json);
  std::optional<OllamaEmbeddingsRequest> parse_embeddings_request(
      const std::string& json);
  std::optional<OllamaPullRequest> parse_pull_request(const std::string& json);
  std::optional<OllamaCreateRequest> parse_create_request(
      const std::string& json);
  std::optional<OllamaShowRequest> parse_show_request(const std::string& json);
  std::optional<OllamaCopyRequest> parse_copy_request(const std::string& json);
  std::optional<OllamaDeleteRequest> parse_delete_request(
      const std::string& json);

  // Response serialization
  std::string serialize_generate_response(
      const OllamaGenerateResponse& response);
  std::string serialize_chat_response(const OllamaChatResponse& response);
  std::string serialize_embeddings_response(
      const OllamaEmbeddingsResponse& response);
  std::string serialize_pull_response(const OllamaPullResponse& response);
  std::string serialize_create_response(const OllamaCreateResponse& response);
  std::string serialize_tags_response(const OllamaTagsResponse& response);
  std::string serialize_ps_response(const OllamaProcessResponse& response);
  std::string serialize_show_response(const OllamaShowResponse& response);

  // Streaming support
  void stream_generate(const OllamaGenerateRequest& request,
                       StreamCallback callback);
  void stream_chat(const OllamaChatRequest& request, StreamCallback callback);
  void stream_pull(const OllamaPullRequest& request, StreamCallback callback);
  void stream_create(const OllamaCreateRequest& request,
                     StreamCallback callback);

  // Utility methods
  std::string current_timestamp_iso8601();
  std::string create_error_response(const std::string& error);
};

}  // namespace server
}  // namespace mlxr
