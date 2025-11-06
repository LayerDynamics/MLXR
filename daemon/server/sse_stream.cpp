// Copyright © 2025 MLXR Development
// Server-Sent Events (SSE) streaming implementation

#include "sse_stream.h"

#include <chrono>
#include <iomanip>
#include <map>
#include <sstream>
#include <thread>

namespace mlxr {
namespace server {

// ==============================================================================
// SSEEvent Implementation
// ==============================================================================

std::string SSEEvent::format() const {
  std::ostringstream oss;

  // Event type (optional)
  if (!event.empty()) {
    oss << "event: " << event << "\n";
  }

  // Data (required) - can be multiple lines
  std::istringstream data_stream(data);
  std::string line;
  while (std::getline(data_stream, line)) {
    oss << "data: " << line << "\n";
  }

  // ID (optional)
  if (!id.empty()) {
    oss << "id: " << id << "\n";
  }

  // Retry (optional)
  if (retry >= 0) {
    oss << "retry: " << retry << "\n";
  }

  // End of event (double newline)
  oss << "\n";

  return oss.str();
}

// ==============================================================================
// SSEStream Implementation
// ==============================================================================

SSEStream::SSEStream(SSECallback callback)
    : callback_(callback), closed_(false), event_count_(0) {}

SSEStream::~SSEStream() { close(); }

bool SSEStream::send(const SSEEvent& event) {
  if (closed_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (closed_) {
    return false;
  }

  std::string formatted = event.format();
  bool success = callback_(formatted);

  if (success) {
    event_count_++;
  } else {
    closed_ = true;
  }

  return success;
}

bool SSEStream::send_data(const std::string& data) {
  SSEEvent event;
  event.data = data;
  return send(event);
}

bool SSEStream::send_event(const std::string& event_type,
                           const std::string& data) {
  SSEEvent event;
  event.event = event_type;
  event.data = data;
  return send(event);
}

bool SSEStream::send_comment(const std::string& comment) {
  if (closed_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (closed_) {
    return false;
  }

  std::string formatted = ": " + comment + "\n\n";
  bool success = callback_(formatted);

  if (!success) {
    closed_ = true;
  }

  return success;
}

void SSEStream::send_done() { send_data("[DONE]"); }

void SSEStream::close() { closed_ = true; }

// ==============================================================================
// SSEResponseBuilder Implementation
// ==============================================================================

std::map<std::string, std::string> SSEResponseBuilder::create_headers() {
  std::map<std::string, std::string> headers;
  headers["Content-Type"] = "text/event-stream";
  headers["Cache-Control"] = "no-cache";
  headers["Connection"] = "keep-alive";
  headers["X-Accel-Buffering"] = "no";  // Disable buffering for nginx
  return headers;
}

std::string SSEResponseBuilder::create_initial_response() {
  return ": SSE stream established\n\n";
}

std::string SSEResponseBuilder::format_data(const std::string& data) {
  SSEEvent event;
  event.data = data;
  return event.format();
}

std::string SSEResponseBuilder::format_event(const std::string& event_type,
                                             const std::string& data) {
  SSEEvent event;
  event.event = event_type;
  event.data = data;
  return event.format();
}

std::string SSEResponseBuilder::format_comment(const std::string& comment) {
  return ": " + comment + "\n\n";
}

std::string SSEResponseBuilder::create_done_marker() {
  return format_data("[DONE]");
}

// ==============================================================================
// StreamingGenerator Implementation
// ==============================================================================

StreamingGenerator::StreamingGenerator(std::shared_ptr<SSEStream> sse_stream)
    : sse_stream_(sse_stream),
      generating_(false),
      stop_requested_(false),
      tokens_generated_(0) {}

StreamingGenerator::~StreamingGenerator() { stop(); }

bool StreamingGenerator::generate(const std::vector<int>& prompt_tokens,
                                  int max_tokens, float temperature) {
  if (generating_) {
    return false;
  }

  generating_ = true;
  stop_requested_ = false;
  tokens_generated_ = 0;

  // Placeholder: In production, would integrate with actual inference engine
  // For now, simulate token generation

  // Mock generation loop
  for (int i = 0; i < max_tokens && !stop_requested_; i++) {
    // Simulate token generation delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Mock token
    int token_id = 100 + i;
    std::string token_text = "token" + std::to_string(i) + " ";

    on_token_generated(token_id, token_text);

    if (!sse_stream_->is_open()) {
      break;
    }
  }

  generating_ = false;
  return true;
}

void StreamingGenerator::stop() { stop_requested_ = true; }

void StreamingGenerator::on_token_generated(int token_id,
                                            const std::string& token_text) {
  // This would be called by the inference engine for each token
  tokens_generated_++;

  // Token callback would be provided by the formatter
  // For now, this is just a placeholder
  (void)token_id;    // Unused in placeholder
  (void)token_text;  // Unused in placeholder
}

// ==============================================================================
// ChatCompletionStreamFormatter Implementation
// ==============================================================================

ChatCompletionStreamFormatter::ChatCompletionStreamFormatter(
    const std::string& request_id, const std::string& model)
    : request_id_(request_id),
      model_(model),
      created_(get_current_timestamp()) {}

std::string ChatCompletionStreamFormatter::format_role(
    const std::string& role) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"chat.completion.chunk\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"delta\":{\"role\":\"" << escape_json(role) << "\"},";
  oss << "\"finish_reason\":null";
  oss << "}]";
  oss << "}";
  return oss.str();
}

std::string ChatCompletionStreamFormatter::format_content(
    const std::string& content) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"chat.completion.chunk\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"delta\":{\"content\":\"" << escape_json(content) << "\"},";
  oss << "\"finish_reason\":null";
  oss << "}]";
  oss << "}";
  return oss.str();
}

std::string ChatCompletionStreamFormatter::format_finish(
    const std::string& finish_reason) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"chat.completion.chunk\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"delta\":{},";
  oss << "\"finish_reason\":\"" << escape_json(finish_reason) << "\"";
  oss << "}]";
  oss << "}";
  return oss.str();
}

std::string ChatCompletionStreamFormatter::format_function_call(
    const std::string& function_name, const std::string& arguments) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"chat.completion.chunk\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"delta\":{";
  oss << "\"function_call\":{";
  oss << "\"name\":\"" << escape_json(function_name) << "\",";
  oss << "\"arguments\":\"" << escape_json(arguments) << "\"";
  oss << "}";
  oss << "},";
  oss << "\"finish_reason\":null";
  oss << "}]";
  oss << "}";
  return oss.str();
}

// ==============================================================================
// CompletionStreamFormatter Implementation
// ==============================================================================

CompletionStreamFormatter::CompletionStreamFormatter(
    const std::string& request_id, const std::string& model)
    : request_id_(request_id),
      model_(model),
      created_(get_current_timestamp()) {}

std::string CompletionStreamFormatter::format_text(const std::string& text) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"text_completion\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"text\":\"" << escape_json(text) << "\",";
  oss << "\"finish_reason\":null";
  oss << "}]";
  oss << "}";
  return oss.str();
}

std::string CompletionStreamFormatter::format_finish(
    const std::string& finish_reason) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\":\"" << escape_json(request_id_) << "\",";
  oss << "\"object\":\"text_completion\",";
  oss << "\"created\":" << created_ << ",";
  oss << "\"model\":\"" << escape_json(model_) << "\",";
  oss << "\"choices\":[{";
  oss << "\"index\":0,";
  oss << "\"text\":\"\",";
  oss << "\"finish_reason\":\"" << escape_json(finish_reason) << "\"";
  oss << "}]";
  oss << "}";
  return oss.str();
}

// ==============================================================================
// Utility Functions
// ==============================================================================

int64_t get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

std::string escape_json(const std::string& str) {
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
              << static_cast<int>(static_cast<unsigned char>(c));
          result += oss.str();
        } else {
          result += c;
        }
    }
  }

  return result;
}

}  // namespace server
}  // namespace mlxr
