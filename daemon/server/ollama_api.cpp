// Copyright © 2025 MLXR Development
// Ollama-compatible API endpoints implementation

#include "ollama_api.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include "../../core/runtime/engine.h"
#include "../registry/model_registry.h"
#include "sse_stream.h"

using json = nlohmann::json;

namespace mlxr {
namespace server {

// ==============================================================================
// OllamaAPIHandler Implementation
// ==============================================================================

OllamaAPIHandler::OllamaAPIHandler()
    : model_(nullptr), tokenizer_(nullptr), engine_(nullptr) {}

OllamaAPIHandler::~OllamaAPIHandler() = default;

void OllamaAPIHandler::set_model(std::shared_ptr<LlamaModel> model) {
  model_ = model;
}

void OllamaAPIHandler::set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) {
  tokenizer_ = tokenizer;
}

void OllamaAPIHandler::set_engine(std::shared_ptr<Engine> engine) {
  engine_ = engine;
}

void OllamaAPIHandler::set_registry(
    std::shared_ptr<registry::ModelRegistry> registry) {
  registry_ = registry;
}

// ==============================================================================
// Endpoint Handlers
// ==============================================================================

std::string OllamaAPIHandler::handle_generate(const std::string& json_request,
                                              StreamCallback stream_callback) {
  auto request_opt = parse_generate_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid generate request format");
  }

  auto& request = *request_opt;

  // Extract model name (remove :latest suffix if present)
  std::string model_name = request.model;
  size_t colon_pos = model_name.find(':');
  if (colon_pos != std::string::npos) {
    model_name = model_name.substr(0, colon_pos);
  }

  // Check if model exists in registry
  if (registry_) {
    auto model_info = registry_->get_model_by_identifier(model_name);
    if (!model_info) {
      return create_error_response("Model not found: " + request.model);
    }
  }

  // If streaming is requested and callback is provided
  if (request.stream.value_or(false) && stream_callback) {
    stream_generate(request, stream_callback);
    return "";  // Streaming handled via callback
  }

  // Non-streaming response
  OllamaGenerateResponse response;
  response.model = request.model;
  response.created_at = current_timestamp_iso8601();
  response.done = false;

  // If no engine, return explicit error
  if (!engine_) {
    return create_error_response("Inference engine not available");
  }

  try{
    // Build prompt with optional system message
    std::string full_prompt = request.prompt;
    if (request.system) {
      full_prompt = "System: " + *request.system + "\n\n" + full_prompt;
    }

    // Count prompt tokens (approximate)
    int prompt_token_count = full_prompt.length() / 4;  // Rough estimate

    // Generate response using engine
    auto start_time = std::chrono::steady_clock::now();
    std::string generated_text = engine_->generate(full_prompt);
    auto end_time = std::chrono::steady_clock::now();

    // Calculate timing
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count();

    // Count generated tokens (approximate)
    int generated_token_count = generated_text.length() / 4;  // Rough estimate

    response.response = generated_text;
    response.done = true;
    response.prompt_eval_count = prompt_token_count;
    response.eval_count = generated_token_count;
    response.total_duration = duration_ns;
    response.eval_duration = duration_ns;  // Simplification

  } catch (const std::exception& e) {
    return create_error_response(std::string("Inference failed: ") + e.what());
  }

  return serialize_generate_response(response);
}

std::string OllamaAPIHandler::handle_chat(const std::string& json_request,
                                          StreamCallback stream_callback) {
  auto request_opt = parse_chat_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid chat request format");
  }

  auto& request = *request_opt;

  // If streaming is requested and callback is provided
  if (request.stream.value_or(false) && stream_callback) {
    stream_chat(request, stream_callback);
    return "";  // Streaming handled via callback
  }

  // Extract model name (remove :latest suffix if present)
  std::string model_name = request.model;
  size_t colon_pos = model_name.find(':');
  if (colon_pos != std::string::npos) {
    model_name = model_name.substr(0, colon_pos);
  }

  // Check if model exists in registry
  if (registry_) {
    auto model_info = registry_->get_model_by_identifier(model_name);
    if (!model_info) {
      return create_error_response("Model not found: " + request.model);
    }
  }

  // Non-streaming response
  OllamaChatResponse response;
  response.model = request.model;
  response.created_at = current_timestamp_iso8601();
  response.message.role = "assistant";

  // If no engine, return explicit error
  if (!engine_) {
    return create_error_response("Inference engine not available");
  }

  try {
    // Build chat prompt from messages
    std::string chat_prompt;
    for (const auto& msg : request.messages) {
      if (msg.role == "system") {
        chat_prompt += "System: " + msg.content + "\n\n";
      } else if (msg.role == "user") {
        chat_prompt += "User: " + msg.content + "\n\n";
      } else if (msg.role == "assistant") {
        chat_prompt += "Assistant: " + msg.content + "\n\n";
      }
    }
    chat_prompt += "Assistant: ";

    // Count prompt tokens (approximate)
    int prompt_token_count = chat_prompt.length() / 4;

    // Generate response using engine - REAL INFERENCE
    auto start_time = std::chrono::steady_clock::now();
    std::string generated_text = engine_->generate(chat_prompt);
    auto end_time = std::chrono::steady_clock::now();

    // Calculate timing
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count();

    // Count generated tokens (approximate)
    int generated_token_count = generated_text.length() / 4;

    response.message.content = generated_text;
    response.done = true;
    response.prompt_eval_count = prompt_token_count;
    response.eval_count = generated_token_count;
    response.total_duration = duration_ns;
    response.eval_duration = duration_ns;

  } catch (const std::exception& e) {
    return create_error_response(std::string("Inference failed: ") + e.what());
  }

  return serialize_chat_response(response);
}

std::string OllamaAPIHandler::handle_embeddings(
    const std::string& json_request) {
  auto request_opt = parse_embeddings_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid embeddings request format");
  }

  auto& request = *request_opt;

  // TODO: In production, would use embedding model
  OllamaEmbeddingsResponse response;
  // Mock 768-dim embedding
  response.embedding.resize(768);
  for (size_t i = 0; i < 768; i++) {
    response.embedding[i] = static_cast<float>(i) / 768.0f;
  }

  return serialize_embeddings_response(response);
}

std::string OllamaAPIHandler::handle_pull(const std::string& json_request,
                                          StreamCallback stream_callback) {
  auto request_opt = parse_pull_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid pull request format");
  }

  auto& request = *request_opt;

  // If streaming is requested and callback is provided
  if (request.stream.value_or(true) && stream_callback) {
    stream_pull(request, stream_callback);
    return "";  // Streaming handled via callback
  }

  // Non-streaming not typically used for pull
  OllamaPullResponse response;
  response.status = "success";
  return serialize_pull_response(response);
}

std::string OllamaAPIHandler::handle_create(const std::string& json_request,
                                            StreamCallback stream_callback) {
  auto request_opt = parse_create_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid create request format");
  }

  auto& request = *request_opt;

  // If streaming is requested and callback is provided
  if (request.stream.value_or(true) && stream_callback) {
    stream_create(request, stream_callback);
    return "";  // Streaming handled via callback
  }

  // Non-streaming response
  OllamaCreateResponse response;
  response.status = "success";
  return serialize_create_response(response);
}

std::string OllamaAPIHandler::handle_tags() {
  OllamaTagsResponse response;

  // Query registry for all models
  if (registry_) {
    auto registered_models = registry_->list_models();

    for (const auto& model : registered_models) {
      OllamaModelInfo model_info;

      // Convert model name to Ollama format (name:latest)
      model_info.name = model.model_id + ":latest";
      model_info.modified_at = current_timestamp_iso8601();
      model_info.size = model.file_size;

      // Generate digest from file path (simple hash substitute)
      std::hash<std::string> hasher;
      size_t hash = hasher(model.file_path);
      std::stringstream ss;
      ss << "sha256:" << std::hex << hash;
      model_info.digest = ss.str();

      // Create details from model info
      OllamaModelInfo::Details details;

      // Format
      if (model.format == registry::ModelFormat::GGUF) {
        details.format = "gguf";
      } else if (model.format == registry::ModelFormat::SAFETENSORS) {
        details.format = "safetensors";
      } else {
        details.format = "unknown";
      }

      // Family/Architecture
      if (model.architecture == registry::ModelArchitecture::LLAMA) {
        details.family = "llama";
        details.families = {"llama"};
      } else if (model.architecture == registry::ModelArchitecture::MISTRAL) {
        details.family = "mistral";
        details.families = {"mistral"};
      } else {
        details.family = "unknown";
        details.families = {"unknown"};
      }

      // Parameter size (estimate from param_count)
      double billion_params = model.param_count / 1e9;
      std::stringstream param_ss;
      param_ss << std::fixed << std::setprecision(1) << billion_params << "B";
      details.parameter_size = param_ss.str();

      // Quantization level
      if (model.quant_type == registry::QuantizationType::NONE) {
        details.quantization_level = "F16";
      } else if (model.quant_type == registry::QuantizationType::Q4_K) {
        details.quantization_level = "Q4_K";
      } else if (model.quant_type == registry::QuantizationType::Q8_K) {
        details.quantization_level = "Q8_K";
      } else {
        details.quantization_level = "Q4_K_M";  // Default
      }

      model_info.details = details;
      response.models.push_back(model_info);
    }
  }

  // Return the response (empty list if no models found)
  return serialize_tags_response(response);
}

std::string OllamaAPIHandler::handle_ps() {
  // TODO: In production, would query running model registry
  OllamaProcessResponse response;

  OllamaRunningModel running_model;
  running_model.name = "llama3:latest";
  running_model.model = "llama3:latest";
  running_model.size = 3826793677;
  running_model.digest = "sha256:mock-digest-123";
  running_model.size_vram = 2147483648;  // 2GB in VRAM

  OllamaRunningModel::Details details;
  details.format = "gguf";
  details.family = "llama";
  details.families = {"llama"};
  details.parameter_size = "7B";
  details.quantization_level = "Q4_K_M";
  running_model.details = details;

  response.models.push_back(running_model);

  return serialize_ps_response(response);
}

std::string OllamaAPIHandler::handle_show(const std::string& json_request) {
  auto request_opt = parse_show_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid show request format");
  }

  auto& request = *request_opt;

  // TODO: In production, would query model registry
  OllamaShowResponse response;
  response.modelfile = "FROM llama3\\nPARAMETER temperature 0.7";
  response.parameters = "temperature 0.7\\ntop_p 0.9";
  response.template_ = "{{ .System }}\\n{{ .Prompt }}";

  OllamaShowResponse::Details details;
  details.format = "gguf";
  details.family = "llama";
  details.families = {"llama"};
  details.parameter_size = "7B";
  details.quantization_level = "Q4_K_M";
  response.details = details;

  return serialize_show_response(response);
}

std::string OllamaAPIHandler::handle_copy(const std::string& json_request) {
  auto request_opt = parse_copy_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid copy request format");
  }

  auto& request = *request_opt;

  // TODO: In production, would copy model in registry
  return "{}";  // Empty JSON for success
}

std::string OllamaAPIHandler::handle_delete(const std::string& json_request) {
  auto request_opt = parse_delete_request(json_request);
  if (!request_opt) {
    return create_error_response("Invalid delete request format");
  }

  auto& request = *request_opt;

  // TODO: In production, would delete model from registry
  return "{}";  // Empty JSON for success
}

// ==============================================================================
// Request Parsing (Production implementations with nlohmann::json)
// ==============================================================================

std::optional<OllamaGenerateRequest> OllamaAPIHandler::parse_generate_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaGenerateRequest request;

    request.model = j.value("model", "");
    request.prompt = j.value("prompt", "");

    if (j.contains("system")) request.system = j["system"];
    if (j.contains("template")) request.template_ = j["template"];
    if (j.contains("context")) request.context = j["context"];
    if (j.contains("stream")) request.stream = j["stream"];
    if (j.contains("raw")) request.raw = j["raw"];
    if (j.contains("format")) request.format = j["format"];
    if (j.contains("num_predict")) request.num_predict = j["num_predict"];
    if (j.contains("temperature")) request.temperature = j["temperature"];
    if (j.contains("top_p")) request.top_p = j["top_p"];
    if (j.contains("top_k")) request.top_k = j["top_k"];
    if (j.contains("repeat_penalty"))
      request.repeat_penalty = j["repeat_penalty"];
    if (j.contains("seed")) request.seed = j["seed"];
    if (j.contains("stop"))
      request.stop = j["stop"].get<std::vector<std::string>>();

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaChatRequest> OllamaAPIHandler::parse_chat_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaChatRequest request;

    request.model = j.value("model", "");

    if (j.contains("messages") && j["messages"].is_array()) {
      for (const auto& msg_json : j["messages"]) {
        OllamaChatMessage msg;
        msg.role = msg_json.value("role", "");
        msg.content = msg_json.value("content", "");
        if (msg_json.contains("images")) {
          msg.images = msg_json["images"].get<std::vector<std::string>>();
        }
        request.messages.push_back(msg);
      }
    }

    if (j.contains("stream")) request.stream = j["stream"];
    if (j.contains("format")) request.format = j["format"];
    if (j.contains("num_predict")) request.num_predict = j["num_predict"];
    if (j.contains("temperature")) request.temperature = j["temperature"];
    if (j.contains("top_p")) request.top_p = j["top_p"];
    if (j.contains("top_k")) request.top_k = j["top_k"];
    if (j.contains("repeat_penalty"))
      request.repeat_penalty = j["repeat_penalty"];
    if (j.contains("seed")) request.seed = j["seed"];
    if (j.contains("stop"))
      request.stop = j["stop"].get<std::vector<std::string>>();

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaEmbeddingsRequest>
OllamaAPIHandler::parse_embeddings_request(const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaEmbeddingsRequest request;

    request.model = j.value("model", "");
    request.prompt = j.value("prompt", "");

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaPullRequest> OllamaAPIHandler::parse_pull_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaPullRequest request;

    request.name = j.value("name", "");
    if (j.contains("insecure")) request.insecure = j["insecure"];
    if (j.contains("stream")) request.stream = j["stream"];

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaCreateRequest> OllamaAPIHandler::parse_create_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaCreateRequest request;

    request.name = j.value("name", "");
    if (j.contains("modelfile")) request.modelfile = j["modelfile"];
    if (j.contains("path")) request.path = j["path"];
    if (j.contains("stream")) request.stream = j["stream"];

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaShowRequest> OllamaAPIHandler::parse_show_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaShowRequest request;

    request.name = j.value("name", "");

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaCopyRequest> OllamaAPIHandler::parse_copy_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaCopyRequest request;

    request.source = j.value("source", "");
    request.destination = j.value("destination", "");

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

std::optional<OllamaDeleteRequest> OllamaAPIHandler::parse_delete_request(
    const std::string& json_str) {
  try {
    auto j = json::parse(json_str);
    OllamaDeleteRequest request;

    request.name = j.value("name", "");

    return request;
  } catch (const json::exception& e) {
    return std::nullopt;
  }
}

// ==============================================================================
// Response Serialization (Using nlohmann::json)
// ==============================================================================

std::string OllamaAPIHandler::serialize_generate_response(
    const OllamaGenerateResponse& response) {
  json j;
  j["model"] = response.model;
  j["created_at"] = response.created_at;
  j["response"] = response.response;
  j["done"] = response.done;

  if (response.context) j["context"] = *response.context;
  if (response.total_duration) j["total_duration"] = *response.total_duration;
  if (response.load_duration) j["load_duration"] = *response.load_duration;
  if (response.prompt_eval_count)
    j["prompt_eval_count"] = *response.prompt_eval_count;
  if (response.prompt_eval_duration)
    j["prompt_eval_duration"] = *response.prompt_eval_duration;
  if (response.eval_count) j["eval_count"] = *response.eval_count;
  if (response.eval_duration) j["eval_duration"] = *response.eval_duration;

  return j.dump();
}

std::string OllamaAPIHandler::serialize_chat_response(
    const OllamaChatResponse& response) {
  json j;
  j["model"] = response.model;
  j["created_at"] = response.created_at;
  j["message"]["role"] = response.message.role;
  j["message"]["content"] = response.message.content;
  j["done"] = response.done;

  if (response.total_duration) j["total_duration"] = *response.total_duration;
  if (response.load_duration) j["load_duration"] = *response.load_duration;
  if (response.prompt_eval_count)
    j["prompt_eval_count"] = *response.prompt_eval_count;
  if (response.prompt_eval_duration)
    j["prompt_eval_duration"] = *response.prompt_eval_duration;
  if (response.eval_count) j["eval_count"] = *response.eval_count;
  if (response.eval_duration) j["eval_duration"] = *response.eval_duration;

  return j.dump();
}

std::string OllamaAPIHandler::serialize_embeddings_response(
    const OllamaEmbeddingsResponse& response) {
  json j;
  j["embedding"] = response.embedding;
  return j.dump();
}

std::string OllamaAPIHandler::serialize_pull_response(
    const OllamaPullResponse& response) {
  json j;
  j["status"] = response.status;
  if (response.digest) j["digest"] = *response.digest;
  if (response.total) j["total"] = *response.total;
  if (response.completed) j["completed"] = *response.completed;
  return j.dump();
}

std::string OllamaAPIHandler::serialize_create_response(
    const OllamaCreateResponse& response) {
  json j;
  j["status"] = response.status;
  return j.dump();
}

std::string OllamaAPIHandler::serialize_tags_response(
    const OllamaTagsResponse& response) {
  json j;
  j["models"] = json::array();

  for (const auto& model : response.models) {
    json model_json;
    model_json["name"] = model.name;
    model_json["modified_at"] = model.modified_at;
    model_json["size"] = model.size;
    model_json["digest"] = model.digest;

    if (model.details) {
      model_json["details"]["format"] = model.details->format;
      model_json["details"]["family"] = model.details->family;
      model_json["details"]["families"] = model.details->families;
      model_json["details"]["parameter_size"] = model.details->parameter_size;
      model_json["details"]["quantization_level"] =
          model.details->quantization_level;
    }

    j["models"].push_back(model_json);
  }

  return j.dump();
}

std::string OllamaAPIHandler::serialize_ps_response(
    const OllamaProcessResponse& response) {
  json j;
  j["models"] = json::array();

  for (const auto& model : response.models) {
    json model_json;
    model_json["name"] = model.name;
    model_json["model"] = model.model;
    model_json["size"] = model.size;
    model_json["digest"] = model.digest;

    if (model.details) {
      model_json["details"]["format"] = model.details->format;
      model_json["details"]["family"] = model.details->family;
      model_json["details"]["families"] = model.details->families;
      model_json["details"]["parameter_size"] = model.details->parameter_size;
      model_json["details"]["quantization_level"] =
          model.details->quantization_level;
    }

    if (model.expires_at) model_json["expires_at"] = *model.expires_at;
    if (model.size_vram) model_json["size_vram"] = *model.size_vram;

    j["models"].push_back(model_json);
  }

  return j.dump();
}

std::string OllamaAPIHandler::serialize_show_response(
    const OllamaShowResponse& response) {
  json j;
  j["modelfile"] = response.modelfile;
  j["parameters"] = response.parameters;
  j["template"] = response.template_;

  if (response.details) {
    j["details"]["format"] = response.details->format;
    j["details"]["family"] = response.details->family;
    j["details"]["families"] = response.details->families;
    j["details"]["parameter_size"] = response.details->parameter_size;
    j["details"]["quantization_level"] = response.details->quantization_level;
  }

  return j.dump();
}

// ==============================================================================
// Streaming Support
// ==============================================================================

void OllamaAPIHandler::stream_generate(const OllamaGenerateRequest& request,
                                       StreamCallback callback) {
  // TODO: In production, would use inference engine for streaming
  // Mock streaming behavior
  for (int i = 0; i < 10; i++) {
    OllamaGenerateResponse chunk;
    chunk.model = request.model;
    chunk.created_at = current_timestamp_iso8601();
    chunk.response = "token" + std::to_string(i) + " ";
    chunk.done = (i == 9);

    std::string json = serialize_generate_response(chunk);
    if (!callback(json + "\n")) {
      break;  // Client disconnected
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void OllamaAPIHandler::stream_chat(const OllamaChatRequest& request,
                                   StreamCallback callback) {
  // TODO: In production, would use inference engine for streaming
  for (int i = 0; i < 10; i++) {
    OllamaChatResponse chunk;
    chunk.model = request.model;
    chunk.created_at = current_timestamp_iso8601();
    chunk.message.role = "assistant";
    chunk.message.content = "token" + std::to_string(i) + " ";
    chunk.done = (i == 9);

    std::string json = serialize_chat_response(chunk);
    if (!callback(json + "\n")) {
      break;  // Client disconnected
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void OllamaAPIHandler::stream_pull(const OllamaPullRequest& request,
                                   StreamCallback callback) {
  // Simulate pull progress
  const std::vector<std::string> statuses = {
      "pulling manifest", "verifying sha256 digest",
      "pulling layers",   "downloading",
      "verifying",        "success"};

  int64_t total = 1000000000;  // 1GB
  for (size_t i = 0; i < statuses.size(); i++) {
    OllamaPullResponse chunk;
    chunk.status = statuses[i];

    if (statuses[i] == "downloading") {
      chunk.total = total;
      chunk.completed = (i * total) / statuses.size();
    }

    std::string json = serialize_pull_response(chunk);
    if (!callback(json + "\n")) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void OllamaAPIHandler::stream_create(const OllamaCreateRequest& request,
                                     StreamCallback callback) {
  // Simulate create progress
  const std::vector<std::string> statuses = {
      "parsing modelfile", "loading base model", "creating model", "success"};

  for (const auto& status : statuses) {
    OllamaCreateResponse chunk;
    chunk.status = status;

    std::string json = serialize_create_response(chunk);
    if (!callback(json + "\n")) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// ==============================================================================
// Utility Methods
// ==============================================================================

std::string OllamaAPIHandler::current_timestamp_iso8601() {
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::gmtime(&now_c);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");

  // Add milliseconds
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";

  return oss.str();
}

std::string OllamaAPIHandler::create_error_response(const std::string& error) {
  json j;
  j["error"] = error;
  return j.dump();
}

}  // namespace server
}  // namespace mlxr
