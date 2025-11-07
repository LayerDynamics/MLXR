// Copyright © 2025 MLXR Development
// REST server with OpenAI-compatible API endpoints

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

namespace scheduler {
class Scheduler;
}

namespace registry {
class ModelRegistry;
}

using runtime::Engine;
using runtime::Tokenizer;

namespace server {

// Forward declaration
class OllamaAPIHandler;

// ==============================================================================
// Request/Response Data Structures
// ==============================================================================

// Chat completion message
struct ChatMessage {
  std::string role;  // "system", "user", "assistant", "function"
  std::string content;
  std::optional<std::string> name;
  std::optional<std::string> function_call;
};

// Function definition for function calling
struct FunctionDefinition {
  std::string name;
  std::string description;
  std::string parameters_json;  // JSON schema
};

// Tool definition
struct ToolDefinition {
  std::string type;  // "function"
  FunctionDefinition function;
};

// Chat completion request (OpenAI-compatible)
struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;

  // Optional parameters
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<float> repetition_penalty;
  std::optional<int> max_tokens;
  std::optional<bool> stream;
  std::optional<std::vector<std::string>> stop;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;
  std::optional<int> n;
  std::optional<std::string> user;
  std::optional<std::vector<ToolDefinition>> tools;
  std::optional<std::string> tool_choice;
  std::optional<int> seed;
};

// Usage statistics
struct UsageInfo {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
};

// Chat completion choice
struct ChatCompletionChoice {
  int index = 0;
  ChatMessage message;
  std::string
      finish_reason;  // "stop", "length", "function_call", "content_filter"
};

// Chat completion response
struct ChatCompletionResponse {
  std::string id;
  std::string object = "chat.completion";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionChoice> choices;
  UsageInfo usage;
};

// Streaming chunk delta
struct ChatCompletionDelta {
  std::optional<std::string> role;
  std::optional<std::string> content;
  std::optional<std::string> function_call;
};

// Streaming chunk choice
struct ChatCompletionStreamChoice {
  int index = 0;
  ChatCompletionDelta delta;
  std::string finish_reason;
};

// Streaming chunk
struct ChatCompletionChunk {
  std::string id;
  std::string object = "chat.completion.chunk";
  int64_t created = 0;
  std::string model;
  std::vector<ChatCompletionStreamChoice> choices;
};

// Completion request (non-chat)
struct CompletionRequest {
  std::string model;
  std::string prompt;

  // Optional parameters
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<float> repetition_penalty;
  std::optional<int> max_tokens;
  std::optional<bool> stream;
  std::optional<std::vector<std::string>> stop;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;
  std::optional<int> n;
  std::optional<std::string> suffix;
  std::optional<int> seed;
};

// Completion choice
struct CompletionChoice {
  int index = 0;
  std::string text;
  std::string finish_reason;
};

// Completion response
struct CompletionResponse {
  std::string id;
  std::string object = "text_completion";
  int64_t created = 0;
  std::string model;
  std::vector<CompletionChoice> choices;
  UsageInfo usage;
};

// Embedding request
struct EmbeddingRequest {
  std::string model;
  std::string input;  // Could be string or array of strings
  std::optional<std::string> encoding_format;  // "float" or "base64"
  std::optional<std::string> user;
};

// Single embedding
struct EmbeddingObject {
  int index = 0;
  std::vector<float> embedding;
  std::string object = "embedding";
};

// Embedding response
struct EmbeddingResponse {
  std::string object = "list";
  std::vector<EmbeddingObject> data;
  std::string model;
  UsageInfo usage;
};

// Model info
struct ModelInfo {
  std::string id;
  std::string object = "model";
  int64_t created = 0;
  std::string owned_by = "mlxr";
};

// Model list response
struct ModelListResponse {
  std::string object = "list";
  std::vector<ModelInfo> data;
};

// Error response
struct ErrorResponse {
  struct ErrorDetail {
    std::string message;
    std::string type;
    std::optional<std::string> code;
  };

  ErrorDetail error;
};

// ==============================================================================
// HTTP Request/Response Structures
// ==============================================================================

struct HttpRequest {
  std::string method;  // GET, POST, etc.
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
  std::map<std::string, std::string> query_params;
};

struct HttpResponse {
  int status_code = 200;
  std::map<std::string, std::string> headers;
  std::string body;
};

// ==============================================================================
// REST Server Configuration
// ==============================================================================

struct ServerConfig {
  std::string bind_address = "127.0.0.1";
  int port = 8080;
  bool enable_unix_socket = true;
  std::string unix_socket_path =
      "~/Library/Application Support/MLXRunner/run/mlxrunner.sock";
  bool enable_cors = true;
  int max_connections = 100;
  int thread_pool_size = 4;
  std::string api_key;  // Optional API key for authentication
  bool enable_metrics = true;
  std::string log_level = "info";

  // Connection timeout settings
  int read_timeout_sec = 30;        // Read timeout in seconds (default: 30s)
  int write_timeout_sec = 30;       // Write timeout in seconds (default: 30s)
  int keep_alive_max_count = 100;   // Max requests per connection (default: 100)
  int keep_alive_timeout_sec = 5;   // Keep-alive timeout in seconds (default: 5s)
  size_t payload_max_length = 100 * 1024 * 1024;  // Max payload size (default: 100MB)
};

// ==============================================================================
// REST Server Class
// ==============================================================================

// Callback type for streaming responses
// Returns true if chunk was sent successfully, false if connection closed
using StreamCallback = std::function<bool(const std::string& chunk)>;

class RestServer {
 public:
  explicit RestServer(const ServerConfig& config);
  ~RestServer();

  // Delete copy operations
  RestServer(const RestServer&) = delete;
  RestServer& operator=(const RestServer&) = delete;

  // Initialize server
  bool initialize();

  // Start/stop server
  bool start();
  void stop();

  // Check if server is running
  bool is_running() const { return running_; }

  // Get server configuration
  const ServerConfig& config() const { return config_; }

  // Set model and inference engine
  void set_model(std::shared_ptr<LlamaModel> model);
  void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer);
  void set_engine(std::shared_ptr<Engine> engine);
  void set_scheduler(std::shared_ptr<scheduler::Scheduler> scheduler);
  void set_registry(std::shared_ptr<registry::ModelRegistry> registry);

  // Endpoint handlers (can be overridden for custom behavior)
  virtual HttpResponse handle_chat_completion(const HttpRequest& request);
  virtual HttpResponse handle_completion(const HttpRequest& request);
  virtual HttpResponse handle_embedding(const HttpRequest& request);
  virtual HttpResponse handle_models(const HttpRequest& request);
  virtual HttpResponse handle_model_info(const HttpRequest& request);

 private:
  // Configuration
  ServerConfig config_;

  // Server state
  bool running_;
  bool initialized_;

  // Model and inference components
  std::shared_ptr<LlamaModel> model_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Engine> engine_;
  std::shared_ptr<scheduler::Scheduler> scheduler_;
  std::shared_ptr<registry::ModelRegistry> registry_;

  // API handlers
  std::unique_ptr<OllamaAPIHandler> ollama_handler_;

  // Request routing
  HttpResponse route_request(const HttpRequest& request);

  // Request parsing
  std::optional<ChatCompletionRequest> parse_chat_completion_request(
      const std::string& json);
  std::optional<CompletionRequest> parse_completion_request(
      const std::string& json);
  std::optional<EmbeddingRequest> parse_embedding_request(
      const std::string& json);

  // Response serialization
  std::string serialize_chat_completion_response(
      const ChatCompletionResponse& response);
  std::string serialize_completion_response(const CompletionResponse& response);
  std::string serialize_embedding_response(const EmbeddingResponse& response);
  std::string serialize_model_list_response(const ModelListResponse& response);
  std::string serialize_error_response(const ErrorResponse& response);
  std::string serialize_chat_completion_chunk(const ChatCompletionChunk& chunk);

  // Streaming support
  void stream_chat_completion(const ChatCompletionRequest& request,
                              StreamCallback callback);

  void stream_completion(const CompletionRequest& request,
                         StreamCallback callback);

  // Utility methods
  std::string generate_request_id();
  int64_t current_timestamp();
  HttpResponse create_error_response(int status_code,
                                     const std::string& message);
  bool validate_api_key(const HttpRequest& request);

  // Server implementation details (platform-specific)
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
}  // namespace mlxr
