// Copyright © 2025 MLXR Development
// REST server implementation

// Include cpp-httplib FIRST before any other includes (fixes namespace
// conflicts)
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "rest_server.h"

#include <httplib.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

// Socket headers for advanced configuration
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "graph/model.h"
#include "model_loader.h"
#include "ollama_api.h"
#include "runtime/engine.h"
#include "runtime/tokenizer/tokenizer.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "scheduler_worker.h"
#include "sse_stream.h"

// Simple JSON parsing/serialization helpers
// In production, would use nlohmann/json or similar
namespace {

std::string escape_json_string(const std::string& str) {
  std::string result;
  result.reserve(str.size());

  for (char c : str) {
    switch (c) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (c < 0x20) {
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::setfill('0') << std::setw(4)
              << static_cast<int>(c);
          result += oss.str();
        } else {
          result += c;
        }
    }
  }

  return result;
}

// Simple JSON value extraction (for minimal parsing)
std::string extract_json_string(const std::string& json,
                                const std::string& key) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) {
    return "";
  }

  pos += search.length();

  // Skip whitespace
  while (pos < json.length() && std::isspace(json[pos])) {
    pos++;
  }

  if (pos >= json.length() || json[pos] != '"') {
    return "";
  }

  pos++;  // Skip opening quote

  std::string result;
  while (pos < json.length() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.length()) {
      pos++;
      switch (json[pos]) {
        case 'n':
          result += '\n';
          break;
        case 't':
          result += '\t';
          break;
        case 'r':
          result += '\r';
          break;
        case '\\':
          result += '\\';
          break;
        case '"':
          result += '"';
          break;
        default:
          result += json[pos];
      }
    } else {
      result += json[pos];
    }
    pos++;
  }

  return result;
}

std::optional<int> extract_json_int(const std::string& json,
                                    const std::string& key) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) {
    return std::nullopt;
  }

  pos += search.length();

  // Skip whitespace
  while (pos < json.length() && std::isspace(json[pos])) {
    pos++;
  }

  std::string num_str;
  while (pos < json.length() && (std::isdigit(json[pos]) || json[pos] == '-')) {
    num_str += json[pos];
    pos++;
  }

  if (num_str.empty()) {
    return std::nullopt;
  }

  return std::stoi(num_str);
}

std::optional<float> extract_json_float(const std::string& json,
                                        const std::string& key) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) {
    return std::nullopt;
  }

  pos += search.length();

  // Skip whitespace
  while (pos < json.length() && std::isspace(json[pos])) {
    pos++;
  }

  std::string num_str;
  while (pos < json.length() &&
         (std::isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.' ||
          json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+')) {
    num_str += json[pos];
    pos++;
  }

  if (num_str.empty()) {
    return std::nullopt;
  }

  return std::stof(num_str);
}

std::optional<bool> extract_json_bool(const std::string& json,
                                      const std::string& key) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) {
    return std::nullopt;
  }

  pos += search.length();

  // Skip whitespace
  while (pos < json.length() && std::isspace(json[pos])) {
    pos++;
  }

  if (json.substr(pos, 4) == "true") {
    return true;
  } else if (json.substr(pos, 5) == "false") {
    return false;
  }

  return std::nullopt;
}

}  // anonymous namespace

namespace mlxr {
namespace server {

// ==============================================================================
// Server Implementation Details
// ==============================================================================

struct RestServer::Impl {
  std::unique_ptr<httplib::Server> http_server;
  std::thread server_thread;
  bool running = false;

  // HTTP server runner thread
  void run_server(RestServer* server) {
    std::cout << "Starting REST server on " << server->config_.bind_address
              << ":" << server->config_.port << std::endl;

    // Create HTTP server instance
    http_server = std::make_unique<httplib::Server>();

    // Configure thread pool for concurrent requests
    http_server->new_task_queue = [&server] {
      return new httplib::ThreadPool(server->config_.thread_pool_size);
    };

    // Set socket options for better connection handling
    http_server->set_socket_options([](socket_t sock) {
      // Enable SO_REUSEADDR to allow quick restart
      int yes = 1;
      setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&yes), sizeof(yes));

      // Set TCP_NODELAY to disable Nagle's algorithm for lower latency
      setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&yes), sizeof(yes));

      // Set socket send/receive buffer sizes
      int buffer_size = 256 * 1024; // 256KB
      setsockopt(sock, SOL_SOCKET, SO_SNDBUF,
                 reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));
      setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
                 reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));
    });

    // Set connection timeouts using configurable values
    http_server->set_read_timeout(server->config_.read_timeout_sec, 0);
    http_server->set_write_timeout(server->config_.write_timeout_sec, 0);

    // Set keep-alive settings for connection reuse
    http_server->set_keep_alive_max_count(server->config_.keep_alive_max_count);
    http_server->set_keep_alive_timeout(server->config_.keep_alive_timeout_sec);

    // Set payload size limits
    http_server->set_payload_max_length(server->config_.payload_max_length);

    // Setup CORS headers if enabled
    if (server->config_.enable_cors) {
      http_server->set_default_headers({
          {"Access-Control-Allow-Origin", "*"},
          {"Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"},
          {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
      });
    }

    // Set up error handler for server errors
    http_server->set_error_handler([](const httplib::Request& req,
                                       httplib::Response& res) {
      std::cerr << "[HTTP ERROR] " << res.status << " for " << req.method
                << " " << req.path << std::endl;

      std::string error_msg = "{\"error\":\"Internal server error\",\"status\":" +
                              std::to_string(res.status) + "}";
      res.set_content(error_msg, "application/json");
    });

    // Set up exception handler to catch unhandled exceptions
    http_server->set_exception_handler([](const httplib::Request& req,
                                           httplib::Response& res,
                                           std::exception_ptr ep) {
      std::string error_msg = "Unknown exception";
      try {
        std::rethrow_exception(ep);
      } catch (std::exception& e) {
        error_msg = e.what();
      } catch (...) {
        error_msg = "Unknown exception type";
      }

      std::cerr << "[HTTP EXCEPTION] " << req.method << " " << req.path
                << " - " << error_msg << std::endl;

      res.status = 500;
      res.set_content("{\"error\":\"" + error_msg + "\"}", "application/json");
    });

    // Set up logger for request/response tracking
    http_server->set_logger([](const httplib::Request& req,
                                const httplib::Response& res) {
      std::cout << "[HTTP] " << req.method << " " << req.path
                << " - " << res.status << std::endl;
    });

    // Register route handlers
    register_routes(server);

    // Start listening
    bool listen_success = http_server->listen(
        server->config_.bind_address.c_str(), server->config_.port);

    if (!listen_success) {
      std::cerr << "Failed to start HTTP server on "
                << server->config_.bind_address << ":" << server->config_.port
                << std::endl;
    }

    std::cout << "REST server stopped" << std::endl;
  }

  // Ollama API handlers
  static void handle_ollama_generate(RestServer* server,
                                     const httplib::Request& req,
                                     httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    // Check if streaming is requested
    bool is_streaming = req.body.find("\"stream\":true") != std::string::npos;

    if (is_streaming) {
      // Set up SSE headers
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");

      // Create stream callback
      auto stream_callback = [&res](const std::string& chunk) -> bool {
        res.body += "data: " + chunk + "\n\n";
        return true;
      };

      std::string response =
          server->ollama_handler_->handle_generate(req.body, stream_callback);
      res.body += "data: " + response + "\n\n";
      res.body += "data: [DONE]\n\n";
    } else {
      std::string response = server->ollama_handler_->handle_generate(req.body);
      res.set_content(response, "application/json");
    }
  }

  static void handle_ollama_chat(RestServer* server,
                                 const httplib::Request& req,
                                 httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    // Check if streaming is requested
    bool is_streaming = req.body.find("\"stream\":true") != std::string::npos;

    if (is_streaming) {
      // Set up SSE headers
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");

      // Create stream callback
      auto stream_callback = [&res](const std::string& chunk) -> bool {
        res.body += "data: " + chunk + "\n\n";
        return true;
      };

      std::string response =
          server->ollama_handler_->handle_chat(req.body, stream_callback);
      res.body += "data: " + response + "\n\n";
      res.body += "data: [DONE]\n\n";
    } else {
      std::string response = server->ollama_handler_->handle_chat(req.body);
      res.set_content(response, "application/json");
    }
  }

  static void handle_ollama_embeddings(RestServer* server,
                                       const httplib::Request& req,
                                       httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_embeddings(req.body);
    res.set_content(response, "application/json");
  }

  static void handle_ollama_pull(RestServer* server,
                                 const httplib::Request& req,
                                 httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    // Check if streaming is requested
    bool is_streaming = req.body.find("\"stream\":true") != std::string::npos;

    if (is_streaming) {
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");

      auto stream_callback = [&res](const std::string& chunk) -> bool {
        res.body += chunk + "\n";
        return true;
      };

      std::string response =
          server->ollama_handler_->handle_pull(req.body, stream_callback);
      res.body += response + "\n";
    } else {
      std::string response = server->ollama_handler_->handle_pull(req.body);
      res.set_content(response, "application/json");
    }
  }

  static void handle_ollama_create(RestServer* server,
                                   const httplib::Request& req,
                                   httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    // Check if streaming is requested
    bool is_streaming = req.body.find("\"stream\":true") != std::string::npos;

    if (is_streaming) {
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");

      auto stream_callback = [&res](const std::string& chunk) -> bool {
        res.body += chunk + "\n";
        return true;
      };

      std::string response =
          server->ollama_handler_->handle_create(req.body, stream_callback);
      res.body += response + "\n";
    } else {
      std::string response = server->ollama_handler_->handle_create(req.body);
      res.set_content(response, "application/json");
    }
  }

  static void handle_ollama_tags(RestServer* server, httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_tags();
    res.set_content(response, "application/json");
  }

  static void handle_ollama_ps(RestServer* server, httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_ps();
    res.set_content(response, "application/json");
  }

  static void handle_ollama_show(RestServer* server,
                                 const httplib::Request& req,
                                 httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_show(req.body);
    res.set_content(response, "application/json");
  }

  static void handle_ollama_copy(RestServer* server,
                                 const httplib::Request& req,
                                 httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_copy(req.body);
    res.set_content(response, "application/json");
  }

  static void handle_ollama_delete(RestServer* server,
                                   const httplib::Request& req,
                                   httplib::Response& res) {
    if (!server->ollama_handler_) {
      res.status = 500;
      res.set_content("{\"error\":\"Ollama handler not initialized\"}",
                      "application/json");
      return;
    }

    std::string response = server->ollama_handler_->handle_delete(req.body);
    res.set_content(response, "application/json");
  }

  // Register all API routes
  void register_routes(RestServer* server) {
    // OpenAI-compatible endpoints
    http_server->Post(
        "/v1/chat/completions",
        [server](const httplib::Request& req, httplib::Response& res) {
          handle_http_request(server, req, res,
                              &RestServer::handle_chat_completion);
        });

    http_server->Post("/v1/completions", [server](const httplib::Request& req,
                                                  httplib::Response& res) {
      handle_http_request(server, req, res, &RestServer::handle_completion);
    });

    http_server->Post("/v1/embeddings", [server](const httplib::Request& req,
                                                 httplib::Response& res) {
      handle_http_request(server, req, res, &RestServer::handle_embedding);
    });

    http_server->Get("/v1/models", [server](const httplib::Request& req,
                                            httplib::Response& res) {
      handle_http_request(server, req, res, &RestServer::handle_models);
    });

    http_server->Get(
        "/v1/models/:model_id",
        [server](const httplib::Request& req, httplib::Response& res) {
          handle_http_request(server, req, res, &RestServer::handle_model_info);
        });

    // Health check endpoint
    http_server->Get(
        "/health", [](const httplib::Request&, httplib::Response& res) {
          res.set_content("{\"status\":\"ok\"}", "application/json");
        });

    // Ollama-compatible endpoints
    http_server->Post("/api/generate", [server](const httplib::Request& req,
                                                httplib::Response& res) {
      handle_ollama_generate(server, req, res);
    });

    http_server->Post("/api/chat", [server](const httplib::Request& req,
                                            httplib::Response& res) {
      handle_ollama_chat(server, req, res);
    });

    http_server->Post("/api/embeddings", [server](const httplib::Request& req,
                                                  httplib::Response& res) {
      handle_ollama_embeddings(server, req, res);
    });

    http_server->Post("/api/pull", [server](const httplib::Request& req,
                                            httplib::Response& res) {
      handle_ollama_pull(server, req, res);
    });

    http_server->Post("/api/create", [server](const httplib::Request& req,
                                              httplib::Response& res) {
      handle_ollama_create(server, req, res);
    });

    http_server->Get("/api/tags",
                     [server](const httplib::Request&, httplib::Response& res) {
                       handle_ollama_tags(server, res);
                     });

    http_server->Get("/api/ps",
                     [server](const httplib::Request&, httplib::Response& res) {
                       handle_ollama_ps(server, res);
                     });

    http_server->Post("/api/show", [server](const httplib::Request& req,
                                            httplib::Response& res) {
      handle_ollama_show(server, req, res);
    });

    http_server->Post("/api/copy", [server](const httplib::Request& req,
                                            httplib::Response& res) {
      handle_ollama_copy(server, req, res);
    });

    http_server->Delete("/api/delete", [server](const httplib::Request& req,
                                                httplib::Response& res) {
      handle_ollama_delete(server, req, res);
    });

    // OPTIONS for CORS preflight
    http_server->Options(".*",
                         [](const httplib::Request&, httplib::Response& res) {
                           res.status = 204;
                         });
  }

  // Helper to convert cpp-httplib request to our HttpRequest format
  static HttpRequest convert_request(const httplib::Request& req) {
    HttpRequest http_req;
    http_req.method = req.method;
    http_req.path = req.path;
    http_req.body = req.body;

    // Copy headers
    for (const auto& [key, value] : req.headers) {
      http_req.headers[key] = value;
    }

    // Copy query params
    for (const auto& [key, value] : req.params) {
      http_req.query_params[key] = value;
    }

    return http_req;
  }

  // Helper to convert our HttpResponse to cpp-httplib response
  static void convert_response(const HttpResponse& http_res,
                               httplib::Response& res) {
    res.status = http_res.status_code;
    res.body = http_res.body;

    // Copy headers
    for (const auto& [key, value] : http_res.headers) {
      res.set_header(key.c_str(), value.c_str());
    }

    // Set content type if not already set
    if (http_res.headers.find("Content-Type") == http_res.headers.end()) {
      res.set_header("Content-Type", "application/json");
    }
  }

  // Generic handler wrapper
  using HandlerFunc = HttpResponse (RestServer::*)(const HttpRequest&);

  static void handle_http_request(RestServer* server,
                                  const httplib::Request& req,
                                  httplib::Response& res, HandlerFunc handler) {
    try {
      // Convert request format
      HttpRequest http_req = convert_request(req);

      // Validate API key if configured
      if (!server->config_.api_key.empty()) {
        auto auth_header = http_req.headers.find("Authorization");
        if (auth_header == http_req.headers.end() ||
            auth_header->second != "Bearer " + server->config_.api_key) {
          res.status = 401;
          res.set_content("{\"error\":\"Unauthorized\"}", "application/json");
          return;
        }
      }

      // Call the handler
      HttpResponse http_res = (server->*handler)(http_req);

      // Convert response format
      convert_response(http_res, res);

    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content("{\"error\":\"Internal server error: " +
                          std::string(e.what()) + "\"}",
                      "application/json");
    }
  }

  // Graceful shutdown
  void shutdown() {
    if (http_server) {
      http_server->stop();
    }
  }
};

// ==============================================================================
// RestServer Implementation
// ==============================================================================

RestServer::RestServer(const ServerConfig& config)
    : config_(config),
      running_(false),
      initialized_(false),
      ollama_handler_(std::make_unique<OllamaAPIHandler>()),
      impl_(std::make_unique<Impl>()) {}

RestServer::~RestServer() { stop(); }

bool RestServer::initialize() {
  if (initialized_) {
    return true;
  }

  // Initialize server components
  std::cout << "Initializing REST server..." << std::endl;

  // Validate configuration
  if (config_.port < 1 || config_.port > 65535) {
    std::cerr << "Invalid port: " << config_.port << std::endl;
    return false;
  }

  initialized_ = true;
  return true;
}

bool RestServer::start() {
  if (!initialized_) {
    std::cerr << "Server not initialized" << std::endl;
    return false;
  }

  if (running_) {
    std::cerr << "Server already running" << std::endl;
    return false;
  }

  std::cout << "Starting REST server..." << std::endl;

  running_ = true;
  impl_->running = true;

  // Start server thread
  impl_->server_thread = std::thread(&Impl::run_server, impl_.get(), this);

  return true;
}

void RestServer::stop() {
  if (!running_) {
    return;
  }

  std::cout << "Stopping REST server..." << std::endl;

  running_ = false;
  impl_->running = false;

  // Gracefully shutdown HTTP server
  impl_->shutdown();

  // Wait for server thread to finish
  if (impl_->server_thread.joinable()) {
    impl_->server_thread.join();
  }

  std::cout << "REST server stopped successfully" << std::endl;
}

void RestServer::set_model(std::shared_ptr<LlamaModel> model) {
  model_ = model;
  if (ollama_handler_) {
    ollama_handler_->set_model(model);
  }
}

void RestServer::set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) {
  tokenizer_ = tokenizer;
  if (ollama_handler_) {
    ollama_handler_->set_tokenizer(tokenizer);
  }
}

void RestServer::set_engine(std::shared_ptr<Engine> engine) {
  engine_ = engine;
  if (ollama_handler_) {
    ollama_handler_->set_engine(engine);
  }
}

void RestServer::set_scheduler(
    std::shared_ptr<scheduler::Scheduler> scheduler) {
  scheduler_ = scheduler;
}

void RestServer::set_worker(std::shared_ptr<SchedulerWorker> worker) {
  worker_ = worker;
}

void RestServer::set_registry(
    std::shared_ptr<registry::ModelRegistry> registry) {
  registry_ = registry;
  if (ollama_handler_) {
    ollama_handler_->set_registry(registry);
  }
}

// ==============================================================================
// Request Routing
// ==============================================================================

HttpResponse RestServer::route_request(const HttpRequest& request) {
  // Validate API key if configured
  if (!config_.api_key.empty() && !validate_api_key(request)) {
    return create_error_response(401, "Invalid API key");
  }

  // Route to appropriate handler
  if (request.method == "POST") {
    if (request.path == "/v1/chat/completions") {
      return handle_chat_completion(request);
    } else if (request.path == "/v1/completions") {
      return handle_completion(request);
    } else if (request.path == "/v1/embeddings") {
      return handle_embedding(request);
    }
  } else if (request.method == "GET") {
    if (request.path == "/v1/models") {
      return handle_models(request);
    } else if (request.path.find("/v1/models/") == 0) {
      return handle_model_info(request);
    }
  }

  return create_error_response(404, "Endpoint not found");
}

// ==============================================================================
// Endpoint Handlers
// ==============================================================================

HttpResponse RestServer::handle_chat_completion(const HttpRequest& request) {
  auto req = parse_chat_completion_request(request.body);
  if (!req.has_value()) {
    return create_error_response(400, "Invalid request format");
  }

  // Check if scheduler is available
  if (!scheduler_) {
    return create_error_response(503, "Scheduler not initialized");
  }

  // Check if tokenizer is loaded
  if (!tokenizer_) {
    return create_error_response(503, "Tokenizer not loaded");
  }

  // Build prompt from messages
  std::string prompt;
  for (const auto& msg : req->messages) {
    prompt += msg.role + ": " + msg.content + "\n";
  }

  // Tokenize prompt
  std::vector<int> prompt_tokens = tokenizer_->encode(prompt);

  // Create sampling parameters with improved defaults
  scheduler::SamplingParams sampling_params;
  sampling_params.temperature =
      req->temperature.value_or(0.7f);  // More focused (was 1.0f)
  sampling_params.top_p =
      req->top_p.value_or(0.9f);  // Nucleus sampling (was 1.0f)
  sampling_params.top_k = req->top_k.value_or(40);  // Top-k filtering
  sampling_params.repetition_penalty =
      req->repetition_penalty.value_or(1.1f);  // Reduce repetition
  sampling_params.max_tokens = req->max_tokens.value_or(512);

  // Parse stop sequences from request
  if (req->stop.has_value()) {
    for (const auto& stop_str : req->stop.value()) {
      // Encode each stop string to token IDs
      std::vector<int> stop_tokens = tokenizer_->encode(stop_str);
      // For simplicity, use the last token as the stop token
      // (In production, would need more sophisticated stop sequence matching)
      if (!stop_tokens.empty()) {
        sampling_params.stop_token_ids.push_back(stop_tokens.back());
      }
    }
  }

  // Generate unique request ID
  std::string request_id = generate_request_id();

  // Create scheduler request
  auto sched_request = std::make_shared<scheduler::Request>(
      request_id, prompt, prompt_tokens, sampling_params);

  // Handle streaming vs non-streaming
  bool stream = req->stream.value_or(false);

  if (stream) {
    // Implement SSE streaming
    std::string sse_content;
    std::mutex stream_mutex;
    bool stream_finished = false;
    std::condition_variable stream_cv;

    // Set up streaming callback
    sched_request->token_callback = [&](int token_id, bool finished) {
      std::lock_guard<std::mutex> lock(stream_mutex);

      // Decode single token to text
      std::string token_text = tokenizer_->decode({token_id});

      // Create SSE chunk
      ChatCompletionChunk chunk;
      chunk.id = request_id;
      chunk.created = current_timestamp();
      chunk.model = req->model;

      ChatCompletionStreamChoice choice;
      choice.index = 0;
      choice.delta.content = token_text;
      choice.finish_reason = finished ? "stop" : "";

      chunk.choices.push_back(choice);

      // Serialize and append to SSE content
      std::string chunk_json = serialize_chat_completion_chunk(chunk);
      sse_content += "data: " + chunk_json + "\n\n";

      if (finished) {
        sse_content += "data: [DONE]\n\n";
        stream_finished = true;
        stream_cv.notify_one();
      }
    };

    // Submit request to scheduler
    bool submitted = scheduler_->submit_request(sched_request);
    if (!submitted) {
      return create_error_response(503, "Request queue full");
    }

    // Wait for streaming to complete (with timeout)
    {
      std::unique_lock<std::mutex> lock(stream_mutex);
      bool success = stream_cv.wait_for(lock, std::chrono::seconds(120),
                                        [&] { return stream_finished; });

      if (!success) {
        scheduler_->cancel_request(request_id);
        return create_error_response(504, "Request timeout");
      }
    }

    // Return SSE response
    HttpResponse http_response;
    http_response.status_code = 200;
    http_response.headers["Content-Type"] = "text/event-stream";
    http_response.headers["Cache-Control"] = "no-cache";
    http_response.headers["Connection"] = "keep-alive";
    http_response.body = sse_content;

    return http_response;
  }

  // Non-streaming: Wait for completion
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  bool completed = false;
  std::string error_msg;

  // Set up completion callback
  sched_request->token_callback = [&](int token_id, bool finished) {
    if (finished) {
      std::lock_guard<std::mutex> lock(completion_mutex);
      completed = true;
      completion_cv.notify_one();
    }
  };

  // Submit request to scheduler
  bool submitted = scheduler_->submit_request(sched_request);
  if (!submitted) {
    return create_error_response(503, "Request queue full");
  }

  // Wait for completion (with timeout)
  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    bool success = completion_cv.wait_for(lock, std::chrono::seconds(60),
                                          [&] { return completed; });

    if (!success) {
      scheduler_->cancel_request(request_id);
      return create_error_response(504, "Request timeout");
    }
  }

  // Check if request failed
  if (sched_request->state == scheduler::RequestState::FAILED) {
    return create_error_response(500, sched_request->error_message);
  }

  // Decode generated tokens to text
  std::string generated_text =
      tokenizer_->decode(sched_request->generated_token_ids);

  // Create response
  ChatCompletionResponse response;
  response.id = request_id;
  response.created = current_timestamp();
  response.model = req->model;

  ChatCompletionChoice choice;
  choice.index = 0;
  choice.message.role = "assistant";
  choice.message.content = generated_text;

  // Map finish reason
  switch (sched_request->finish_reason) {
    case scheduler::FinishReason::STOP:
    case scheduler::FinishReason::EOS:
      choice.finish_reason = "stop";
      break;
    case scheduler::FinishReason::LENGTH:
      choice.finish_reason = "length";
      break;
    default:
      choice.finish_reason = "stop";
  }

  response.choices.push_back(choice);

  response.usage.prompt_tokens = sched_request->num_prompt_tokens;
  response.usage.completion_tokens = sched_request->num_generated_tokens;
  response.usage.total_tokens =
      response.usage.prompt_tokens + response.usage.completion_tokens;

  // Serialize response
  HttpResponse http_response;
  http_response.status_code = 200;
  http_response.headers["Content-Type"] = "application/json";
  http_response.body = serialize_chat_completion_response(response);

  return http_response;
}

HttpResponse RestServer::handle_completion(const HttpRequest& request) {
  auto req = parse_completion_request(request.body);
  if (!req.has_value()) {
    return create_error_response(400, "Invalid request format");
  }

  // Check if scheduler is available
  if (!scheduler_) {
    return create_error_response(503, "Scheduler not initialized");
  }

  // Check if tokenizer is loaded
  if (!tokenizer_) {
    return create_error_response(503, "Tokenizer not loaded");
  }

  // Tokenize prompt
  std::vector<int> prompt_tokens = tokenizer_->encode(req->prompt);

  // Create sampling parameters with improved defaults
  scheduler::SamplingParams sampling_params;
  sampling_params.temperature =
      req->temperature.value_or(0.7f);  // More focused (was 1.0f)
  sampling_params.top_p =
      req->top_p.value_or(0.9f);  // Nucleus sampling (was 1.0f)
  sampling_params.top_k = req->top_k.value_or(40);  // Top-k filtering
  sampling_params.repetition_penalty =
      req->repetition_penalty.value_or(1.1f);  // Reduce repetition
  sampling_params.max_tokens = req->max_tokens.value_or(512);

  // Parse stop sequences
  if (req->stop.has_value()) {
    for (const auto& stop_str : req->stop.value()) {
      std::vector<int> stop_tokens = tokenizer_->encode(stop_str);
      if (!stop_tokens.empty()) {
        sampling_params.stop_token_ids.push_back(stop_tokens.back());
      }
    }
  }

  // Generate unique request ID
  std::string request_id = generate_request_id();

  // Create scheduler request
  auto sched_request = std::make_shared<scheduler::Request>(
      request_id, req->prompt, prompt_tokens, sampling_params);

  // Handle streaming
  bool stream = req->stream.value_or(false);
  if (stream) {
    // Implement SSE streaming for completions
    std::string sse_content;
    std::mutex stream_mutex;
    bool stream_finished = false;
    std::condition_variable stream_cv;

    // Set up streaming callback
    sched_request->token_callback = [&](int token_id, bool finished) {
      std::lock_guard<std::mutex> lock(stream_mutex);

      // Decode single token to text
      std::string token_text = tokenizer_->decode({token_id});

      // Create simple completion chunk (non-chat format)
      std::string chunk_json =
          "{\"text\":\"" + escape_json_string(token_text) + "\"";
      if (finished) {
        chunk_json += ",\"finish_reason\":\"stop\"";
      }
      chunk_json += "}";

      sse_content += "data: " + chunk_json + "\n\n";

      if (finished) {
        sse_content += "data: [DONE]\n\n";
        stream_finished = true;
        stream_cv.notify_one();
      }
    };

    // Submit request to scheduler
    bool submitted = scheduler_->submit_request(sched_request);
    if (!submitted) {
      return create_error_response(503, "Request queue full");
    }

    // Wait for streaming to complete
    {
      std::unique_lock<std::mutex> lock(stream_mutex);
      bool success = stream_cv.wait_for(lock, std::chrono::seconds(120),
                                        [&] { return stream_finished; });

      if (!success) {
        scheduler_->cancel_request(request_id);
        return create_error_response(504, "Request timeout");
      }
    }

    // Return SSE response
    HttpResponse http_response;
    http_response.status_code = 200;
    http_response.headers["Content-Type"] = "text/event-stream";
    http_response.headers["Cache-Control"] = "no-cache";
    http_response.headers["Connection"] = "keep-alive";
    http_response.body = sse_content;

    return http_response;
  }

  // Non-streaming: Wait for completion
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  bool completed = false;

  // Set up completion callback
  sched_request->token_callback = [&](int token_id, bool finished) {
    if (finished) {
      std::lock_guard<std::mutex> lock(completion_mutex);
      completed = true;
      completion_cv.notify_one();
    }
  };

  // Submit request to scheduler
  bool submitted = scheduler_->submit_request(sched_request);
  if (!submitted) {
    return create_error_response(503, "Request queue full");
  }

  // Wait for completion (with timeout)
  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    bool success = completion_cv.wait_for(lock, std::chrono::seconds(60),
                                          [&] { return completed; });

    if (!success) {
      scheduler_->cancel_request(request_id);
      return create_error_response(504, "Request timeout");
    }
  }

  // Check if request failed
  if (sched_request->state == scheduler::RequestState::FAILED) {
    return create_error_response(500, sched_request->error_message);
  }

  // Decode generated tokens to text
  std::string generated_text =
      tokenizer_->decode(sched_request->generated_token_ids);

  // Create response
  CompletionResponse response;
  response.id = request_id;
  response.created = current_timestamp();
  response.model = req->model;

  CompletionChoice choice;
  choice.index = 0;
  choice.text = generated_text;

  // Map finish reason
  switch (sched_request->finish_reason) {
    case scheduler::FinishReason::STOP:
    case scheduler::FinishReason::EOS:
      choice.finish_reason = "stop";
      break;
    case scheduler::FinishReason::LENGTH:
      choice.finish_reason = "length";
      break;
    default:
      choice.finish_reason = "stop";
  }

  response.choices.push_back(choice);

  response.usage.prompt_tokens = sched_request->num_prompt_tokens;
  response.usage.completion_tokens = sched_request->num_generated_tokens;
  response.usage.total_tokens =
      response.usage.prompt_tokens + response.usage.completion_tokens;

  HttpResponse http_response;
  http_response.status_code = 200;
  http_response.headers["Content-Type"] = "application/json";
  http_response.body = serialize_completion_response(response);

  return http_response;
}

HttpResponse RestServer::handle_embedding(const HttpRequest& request) {
  auto req = parse_embedding_request(request.body);
  if (!req.has_value()) {
    return create_error_response(400, "Invalid request format");
  }

  // Check if model is loaded
  if (!model_ || !tokenizer_) {
    return create_error_response(503, "Model not loaded");
  }

  // Tokenize input
  std::vector<int> tokens = tokenizer_->encode(req->input);

  // Generate embedding (placeholder)
  // In production, would use model to generate actual embeddings
  std::vector<float> embedding(768, 0.0f);

  // Mock embedding: just fill with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(0.0f, 1.0f);

  for (float& val : embedding) {
    val = dis(gen);
  }

  // Normalize
  float norm = 0.0f;
  for (float val : embedding) {
    norm += val * val;
  }
  norm = std::sqrt(norm);

  if (norm > 0.0f) {
    for (float& val : embedding) {
      val /= norm;
    }
  }

  // Create response
  EmbeddingResponse response;
  response.model = req->model;

  EmbeddingObject emb_obj;
  emb_obj.index = 0;
  emb_obj.embedding = std::move(embedding);

  response.data.push_back(emb_obj);

  response.usage.prompt_tokens = static_cast<int>(tokens.size());
  response.usage.total_tokens = response.usage.prompt_tokens;

  HttpResponse http_response;
  http_response.status_code = 200;
  http_response.headers["Content-Type"] = "application/json";
  http_response.body = serialize_embedding_response(response);

  return http_response;
}

HttpResponse RestServer::handle_models(const HttpRequest& request) {
  ModelListResponse response;

  // Return available models
  // In production, would query model registry
  if (model_) {
    ModelInfo info;
    info.id = "llama-7b";
    info.created = current_timestamp();
    response.data.push_back(info);
  }

  HttpResponse http_response;
  http_response.status_code = 200;
  http_response.headers["Content-Type"] = "application/json";
  http_response.body = serialize_model_list_response(response);

  return http_response;
}

HttpResponse RestServer::handle_model_info(const HttpRequest& request) {
  // Extract model ID from path
  std::string model_id = request.path.substr(strlen("/v1/models/"));

  if (!model_) {
    return create_error_response(404, "Model not found");
  }

  ModelInfo info;
  info.id = model_id;
  info.created = current_timestamp();

  // Serialize single model info
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json_string(info.id) << "\",";
  oss << "\"object\":\"" << escape_json_string(info.object) << "\",";
  oss << "\"created\":" << info.created << ",";
  oss << "\"owned_by\":\"" << escape_json_string(info.owned_by) << "\"";
  oss << "}";

  HttpResponse http_response;
  http_response.status_code = 200;
  http_response.headers["Content-Type"] = "application/json";
  http_response.body = oss.str();

  return http_response;
}

// ==============================================================================
// Request Parsing
// ==============================================================================

std::optional<ChatCompletionRequest> RestServer::parse_chat_completion_request(
    const std::string& json) {
  ChatCompletionRequest req;

  // Extract required fields
  req.model = extract_json_string(json, "model");
  if (req.model.empty()) {
    return std::nullopt;
  }

  // Extract optional fields
  req.temperature = extract_json_float(json, "temperature");
  req.top_p = extract_json_float(json, "top_p");
  req.max_tokens = extract_json_int(json, "max_tokens");
  req.stream = extract_json_bool(json, "stream");

  // Parse messages (simplified)
  // In production, would use proper JSON parser
  size_t messages_pos = json.find("\"messages\":");
  if (messages_pos != std::string::npos) {
    // Mock: just create one user message
    ChatMessage msg;
    msg.role = "user";
    msg.content = extract_json_string(json, "content");
    if (msg.content.empty()) {
      msg.content = "Hello";
    }
    req.messages.push_back(msg);
  }

  return req;
}

std::optional<CompletionRequest> RestServer::parse_completion_request(
    const std::string& json) {
  CompletionRequest req;

  req.model = extract_json_string(json, "model");
  if (req.model.empty()) {
    return std::nullopt;
  }

  req.prompt = extract_json_string(json, "prompt");
  if (req.prompt.empty()) {
    return std::nullopt;
  }

  req.temperature = extract_json_float(json, "temperature");
  req.top_p = extract_json_float(json, "top_p");
  req.max_tokens = extract_json_int(json, "max_tokens");
  req.stream = extract_json_bool(json, "stream");

  return req;
}

std::optional<EmbeddingRequest> RestServer::parse_embedding_request(
    const std::string& json) {
  EmbeddingRequest req;

  req.model = extract_json_string(json, "model");
  if (req.model.empty()) {
    return std::nullopt;
  }

  req.input = extract_json_string(json, "input");
  if (req.input.empty()) {
    return std::nullopt;
  }

  return req;
}

// ==============================================================================
// Response Serialization
// ==============================================================================

std::string RestServer::serialize_chat_completion_response(
    const ChatCompletionResponse& response) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json_string(response.id) << "\",";
  oss << "\"object\":\"" << escape_json_string(response.object) << "\",";
  oss << "\"created\":" << response.created << ",";
  oss << "\"model\":\"" << escape_json_string(response.model) << "\",";

  oss << "\"choices\":[";
  for (size_t i = 0; i < response.choices.size(); i++) {
    const auto& choice = response.choices[i];
    if (i > 0) oss << ",";

    oss << "{";
    oss << "\"index\":" << choice.index << ",";
    oss << "\"message\":{";
    oss << "\"role\":\"" << escape_json_string(choice.message.role) << "\",";
    oss << "\"content\":\"" << escape_json_string(choice.message.content)
        << "\"";
    oss << "},";
    oss << "\"finish_reason\":\"" << escape_json_string(choice.finish_reason)
        << "\"";
    oss << "}";
  }
  oss << "],";

  oss << "\"usage\":{";
  oss << "\"prompt_tokens\":" << response.usage.prompt_tokens << ",";
  oss << "\"completion_tokens\":" << response.usage.completion_tokens << ",";
  oss << "\"total_tokens\":" << response.usage.total_tokens;
  oss << "}";

  oss << "}";

  return oss.str();
}

std::string RestServer::serialize_completion_response(
    const CompletionResponse& response) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json_string(response.id) << "\",";
  oss << "\"object\":\"" << escape_json_string(response.object) << "\",";
  oss << "\"created\":" << response.created << ",";
  oss << "\"model\":\"" << escape_json_string(response.model) << "\",";

  oss << "\"choices\":[";
  for (size_t i = 0; i < response.choices.size(); i++) {
    const auto& choice = response.choices[i];
    if (i > 0) oss << ",";

    oss << "{";
    oss << "\"index\":" << choice.index << ",";
    oss << "\"text\":\"" << escape_json_string(choice.text) << "\",";
    oss << "\"finish_reason\":\"" << escape_json_string(choice.finish_reason)
        << "\"";
    oss << "}";
  }
  oss << "],";

  oss << "\"usage\":{";
  oss << "\"prompt_tokens\":" << response.usage.prompt_tokens << ",";
  oss << "\"completion_tokens\":" << response.usage.completion_tokens << ",";
  oss << "\"total_tokens\":" << response.usage.total_tokens;
  oss << "}";

  oss << "}";

  return oss.str();
}

std::string RestServer::serialize_embedding_response(
    const EmbeddingResponse& response) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"object\":\"" << escape_json_string(response.object) << "\",";
  oss << "\"model\":\"" << escape_json_string(response.model) << "\",";

  oss << "\"data\":[";
  for (size_t i = 0; i < response.data.size(); i++) {
    const auto& emb = response.data[i];
    if (i > 0) oss << ",";

    oss << "{";
    oss << "\"object\":\"" << escape_json_string(emb.object) << "\",";
    oss << "\"index\":" << emb.index << ",";
    oss << "\"embedding\":[";

    for (size_t j = 0; j < emb.embedding.size(); j++) {
      if (j > 0) oss << ",";
      oss << emb.embedding[j];
    }

    oss << "]";
    oss << "}";
  }
  oss << "],";

  oss << "\"usage\":{";
  oss << "\"prompt_tokens\":" << response.usage.prompt_tokens << ",";
  oss << "\"total_tokens\":" << response.usage.total_tokens;
  oss << "}";

  oss << "}";

  return oss.str();
}

std::string RestServer::serialize_model_list_response(
    const ModelListResponse& response) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"object\":\"" << escape_json_string(response.object) << "\",";
  oss << "\"data\":[";

  for (size_t i = 0; i < response.data.size(); i++) {
    const auto& model = response.data[i];
    if (i > 0) oss << ",";

    oss << "{";
    oss << "\"id\":\"" << escape_json_string(model.id) << "\",";
    oss << "\"object\":\"" << escape_json_string(model.object) << "\",";
    oss << "\"created\":" << model.created << ",";
    oss << "\"owned_by\":\"" << escape_json_string(model.owned_by) << "\"";
    oss << "}";
  }

  oss << "]";
  oss << "}";

  return oss.str();
}

std::string RestServer::serialize_error_response(
    const ErrorResponse& response) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"error\":{";
  oss << "\"message\":\"" << escape_json_string(response.error.message)
      << "\",";
  oss << "\"type\":\"" << escape_json_string(response.error.type) << "\"";
  if (response.error.code.has_value()) {
    oss << ",\"code\":\"" << escape_json_string(response.error.code.value())
        << "\"";
  }
  oss << "}";
  oss << "}";

  return oss.str();
}

std::string RestServer::serialize_chat_completion_chunk(
    const ChatCompletionChunk& chunk) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json_string(chunk.id) << "\",";
  oss << "\"object\":\"" << escape_json_string(chunk.object) << "\",";
  oss << "\"created\":" << chunk.created << ",";
  oss << "\"model\":\"" << escape_json_string(chunk.model) << "\",";

  oss << "\"choices\":[";
  for (size_t i = 0; i < chunk.choices.size(); i++) {
    const auto& choice = chunk.choices[i];
    if (i > 0) oss << ",";

    oss << "{";
    oss << "\"index\":" << choice.index << ",";
    oss << "\"delta\":{";

    bool first = true;
    if (choice.delta.role.has_value()) {
      oss << "\"role\":\"" << escape_json_string(choice.delta.role.value())
          << "\"";
      first = false;
    }
    if (choice.delta.content.has_value()) {
      if (!first) oss << ",";
      oss << "\"content\":\""
          << escape_json_string(choice.delta.content.value()) << "\"";
      first = false;
    }

    oss << "},";
    oss << "\"finish_reason\":";
    if (choice.finish_reason.empty()) {
      oss << "null";
    } else {
      oss << "\"" << escape_json_string(choice.finish_reason) << "\"";
    }
    oss << "}";
  }
  oss << "]";

  oss << "}";

  return oss.str();
}

// ==============================================================================
// Streaming Support
// ==============================================================================

void RestServer::stream_chat_completion(const ChatCompletionRequest& request,
                                        StreamCallback callback) {
  // Create SSE stream
  auto sse_stream = std::make_shared<SSEStream>(callback);

  // Create formatter
  std::string request_id = generate_request_id();
  ChatCompletionStreamFormatter formatter(request_id, request.model);

  // Send initial role chunk
  std::string role_chunk = formatter.format_role("assistant");
  sse_stream->send_data(role_chunk);

  // Build prompt from messages
  std::string prompt;
  for (const auto& msg : request.messages) {
    prompt += msg.role + ": " + msg.content + "\n";
  }

  // Tokenize prompt
  if (!tokenizer_) {
    sse_stream->send_data(formatter.format_finish("error"));
    sse_stream->send_done();
    return;
  }

  std::vector<int> prompt_tokens = tokenizer_->encode(prompt);

  // Get parameters
  int max_tokens = request.max_tokens.value_or(100);
  float temperature = request.temperature.value_or(0.7f);

  // Placeholder: Mock token streaming
  // In production, would use inference engine
  for (int i = 0; i < std::min(max_tokens, 20); i++) {
    std::string token_text = "token" + std::to_string(i) + " ";

    // Send content delta
    std::string content_chunk = formatter.format_content(token_text);
    if (!sse_stream->send_data(content_chunk)) {
      break;  // Client disconnected
    }

    // Simulate generation delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Send finish reason
  std::string finish_chunk = formatter.format_finish("stop");
  sse_stream->send_data(finish_chunk);

  // Send [DONE] marker
  sse_stream->send_done();
}

void RestServer::stream_completion(const CompletionRequest& request,
                                   StreamCallback callback) {
  // Create SSE stream
  auto sse_stream = std::make_shared<SSEStream>(callback);

  // Create formatter
  std::string request_id = generate_request_id();
  CompletionStreamFormatter formatter(request_id, request.model);

  // Tokenize prompt
  if (!tokenizer_) {
    sse_stream->send_data(formatter.format_finish("error"));
    sse_stream->send_done();
    return;
  }

  std::vector<int> prompt_tokens = tokenizer_->encode(request.prompt);

  // Get parameters
  int max_tokens = request.max_tokens.value_or(100);
  float temperature = request.temperature.value_or(0.7f);

  // Placeholder: Mock token streaming
  for (int i = 0; i < std::min(max_tokens, 20); i++) {
    std::string token_text = "token" + std::to_string(i) + " ";

    // Send text delta
    std::string text_chunk = formatter.format_text(token_text);
    if (!sse_stream->send_data(text_chunk)) {
      break;  // Client disconnected
    }

    // Simulate generation delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Send finish reason
  std::string finish_chunk = formatter.format_finish("stop");
  sse_stream->send_data(finish_chunk);

  // Send [DONE] marker
  sse_stream->send_done();
}

// ==============================================================================
// Utility Methods
// ==============================================================================

std::string RestServer::generate_request_id() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);

  const char* hex = "0123456789abcdef";
  std::string id = "chatcmpl-";

  for (int i = 0; i < 24; i++) {
    id += hex[dis(gen)];
  }

  return id;
}

int64_t RestServer::current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

HttpResponse RestServer::create_error_response(int status_code,
                                               const std::string& message) {
  ErrorResponse error;
  error.error.message = message;

  switch (status_code) {
    case 400:
      error.error.type = "invalid_request_error";
      break;
    case 401:
      error.error.type = "authentication_error";
      break;
    case 403:
      error.error.type = "permission_error";
      break;
    case 404:
      error.error.type = "not_found_error";
      break;
    case 429:
      error.error.type = "rate_limit_error";
      break;
    case 500:
      error.error.type = "server_error";
      break;
    case 503:
      error.error.type = "service_unavailable";
      break;
    default:
      error.error.type = "error";
      break;
  }

  HttpResponse response;
  response.status_code = status_code;
  response.headers["Content-Type"] = "application/json";
  response.body = serialize_error_response(error);

  return response;
}

bool RestServer::validate_api_key(const HttpRequest& request) {
  // Check Authorization header
  auto it = request.headers.find("Authorization");
  if (it == request.headers.end()) {
    return false;
  }

  std::string auth = it->second;
  if (auth.find("Bearer ") != 0) {
    return false;
  }

  std::string provided_key = auth.substr(7);
  return provided_key == config_.api_key;
}

// ============================================================================
// Model Loading Methods
// ============================================================================

bool RestServer::load_model(const std::string& model_name) {
  std::lock_guard<std::mutex> lock(model_mutex_);

  std::cout << "[RestServer] Loading model: " << model_name << std::endl;

  if (!registry_) {
    std::cerr << "[RestServer] Error: Model registry not set" << std::endl;
    return false;
  }

  // Create ModelLoader
  auto model_loader = std::make_shared<ModelLoader>(registry_);

  // Load the model with default configuration
  LoadModelConfig config;
  config.use_cached_attention = true;  // Enable Metal kernels
  config.prefetch_weights = true;
  config.lock_weights = false;

  auto loaded_model_opt = model_loader->load_model(model_name, config);
  if (!loaded_model_opt.has_value()) {
    std::cerr << "[RestServer] Failed to load model: "
              << model_loader->last_error() << std::endl;
    return false;
  }

  auto& loaded_model = loaded_model_opt.value();

  // Update server components
  // Note: model is stored in engine, no need to keep separate reference
  tokenizer_ = loaded_model.tokenizer;
  engine_ = loaded_model.engine;
  current_model_name_ = model_name;

  std::cout << "[RestServer] Model loaded successfully: " << model_name
            << std::endl;
  std::cout << "[RestServer]   Format: "
            << (loaded_model.info.format == registry::ModelFormat::GGUF
                    ? "GGUF"
                    : "SAFETENSORS")
            << std::endl;
  std::cout << "[RestServer]   Params: " << loaded_model.info.param_count
            << std::endl;
  std::cout << "[RestServer]   Context: " << loaded_model.info.context_length
            << std::endl;

  // If a worker exists, update its engine (thread-safe)
  if (worker_) {
    worker_->set_engine(engine_);
    std::cout << "[RestServer] Updated worker engine" << std::endl;
  }

  return true;
}

bool RestServer::unload_model(const std::string& model_name) {
  std::lock_guard<std::mutex> lock(model_mutex_);

  std::cout << "[RestServer] Unloading model: " << model_name << std::endl;

  if (current_model_name_ != model_name) {
    std::cerr << "[RestServer] Model not currently loaded: " << model_name
              << std::endl;
    return false;
  }

  // Clear current model (model is stored in engine)
  engine_.reset();
  tokenizer_.reset();
  current_model_name_.clear();

  std::cout << "[RestServer] Model unloaded successfully" << std::endl;
  return true;
}

std::string RestServer::current_model() const {
  // Need lock since std::string is not thread-safe (even for reads)
  std::lock_guard<std::mutex> lock(model_mutex_);
  return current_model_name_;
}

}  // namespace server
}  // namespace mlxr
