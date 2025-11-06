// Copyright © 2025 MLXR Development
// Server-Sent Events (SSE) streaming support

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace mlxr {
namespace server {

// ==============================================================================
// SSE Event Structure
// ==============================================================================

/**
 * @brief Represents a single Server-Sent Event
 */
struct SSEEvent {
  std::string event;  // Event type (optional, defaults to "message")
  std::string data;   // Event data
  std::string id;     // Event ID (optional)
  int retry = -1;     // Retry time in milliseconds (optional)

  /**
   * @brief Format event as SSE protocol string
   * @return Formatted SSE event string
   */
  std::string format() const;
};

// ==============================================================================
// SSE Stream Handler
// ==============================================================================

/**
 * @brief Callback type for sending SSE events
 *
 * Returns true if event was sent successfully, false if connection closed
 */
using SSECallback = std::function<bool(const std::string& event)>;

/**
 * @brief Manages Server-Sent Events streaming
 *
 * Provides a thread-safe interface for streaming events to clients
 * following the SSE protocol specification.
 */
class SSEStream {
 public:
  /**
   * @brief Construct SSE stream with callback
   * @param callback Function to call for each event
   */
  explicit SSEStream(SSECallback callback);

  /**
   * @brief Destructor - closes stream and cleans up
   */
  ~SSEStream();

  // Delete copy operations
  SSEStream(const SSEStream&) = delete;
  SSEStream& operator=(const SSEStream&) = delete;

  /**
   * @brief Send an event to the client
   * @param event Event to send
   * @return True if sent successfully, false if stream closed
   */
  bool send(const SSEEvent& event);

  /**
   * @brief Send a simple data-only event
   * @param data Event data
   * @return True if sent successfully
   */
  bool send_data(const std::string& data);

  /**
   * @brief Send an event with type and data
   * @param event_type Event type
   * @param data Event data
   * @return True if sent successfully
   */
  bool send_event(const std::string& event_type, const std::string& data);

  /**
   * @brief Send a comment (used for keep-alive)
   * @param comment Comment text
   * @return True if sent successfully
   */
  bool send_comment(const std::string& comment);

  /**
   * @brief Send the [DONE] marker to indicate stream completion
   */
  void send_done();

  /**
   * @brief Close the stream
   */
  void close();

  /**
   * @brief Check if stream is still open
   * @return True if stream is open
   */
  bool is_open() const { return !closed_; }

  /**
   * @brief Get number of events sent
   * @return Event count
   */
  size_t event_count() const { return event_count_; }

 private:
  SSECallback callback_;
  std::atomic<bool> closed_;
  std::atomic<size_t> event_count_;
  mutable std::mutex mutex_;
};

// ==============================================================================
// SSE Response Builder
// ==============================================================================

/**
 * @brief Helper class for building SSE HTTP responses
 */
class SSEResponseBuilder {
 public:
  /**
   * @brief Create SSE response headers
   * @return Map of HTTP headers for SSE
   */
  static std::map<std::string, std::string> create_headers();

  /**
   * @brief Create initial SSE response
   * @return Initial HTTP response with SSE headers
   */
  static std::string create_initial_response();

  /**
   * @brief Format data as SSE event
   * @param data Event data
   * @return Formatted SSE string
   */
  static std::string format_data(const std::string& data);

  /**
   * @brief Format event with type and data
   * @param event_type Event type
   * @param data Event data
   * @return Formatted SSE string
   */
  static std::string format_event(const std::string& event_type,
                                  const std::string& data);

  /**
   * @brief Format comment
   * @param comment Comment text
   * @return Formatted SSE comment
   */
  static std::string format_comment(const std::string& comment);

  /**
   * @brief Create [DONE] marker
   * @return SSE done marker string
   */
  static std::string create_done_marker();
};

// ==============================================================================
// Streaming Token Generator
// ==============================================================================

/**
 * @brief Token callback for streaming generation
 *
 * Called for each generated token during streaming inference
 */
using TokenCallback =
    std::function<void(int token_id, const std::string& token_text)>;

/**
 * @brief Manages streaming token generation
 *
 * Wraps inference engine and provides token-by-token streaming
 * with proper SSE formatting.
 */
class StreamingGenerator {
 public:
  /**
   * @brief Create streaming generator
   * @param sse_stream SSE stream to send events to
   */
  explicit StreamingGenerator(std::shared_ptr<SSEStream> sse_stream);

  /**
   * @brief Destructor
   */
  ~StreamingGenerator();

  /**
   * @brief Start streaming generation
   * @param prompt_tokens Input token IDs
   * @param max_tokens Maximum tokens to generate
   * @param temperature Sampling temperature
   * @return True if generation completed successfully
   */
  bool generate(const std::vector<int>& prompt_tokens, int max_tokens,
                float temperature = 0.7f);

  /**
   * @brief Stop generation early
   */
  void stop();

  /**
   * @brief Check if generation is active
   * @return True if currently generating
   */
  bool is_generating() const { return generating_; }

  /**
   * @brief Get number of tokens generated so far
   * @return Token count
   */
  size_t tokens_generated() const { return tokens_generated_; }

 private:
  std::shared_ptr<SSEStream> sse_stream_;
  std::atomic<bool> generating_;
  std::atomic<bool> stop_requested_;
  std::atomic<size_t> tokens_generated_;

  // Token callback for inference engine
  void on_token_generated(int token_id, const std::string& token_text);
};

// ==============================================================================
// OpenAI Streaming Formatters
// ==============================================================================

/**
 * @brief Format chat completion chunk for streaming
 *
 * Creates OpenAI-compatible streaming chunk with delta content
 */
class ChatCompletionStreamFormatter {
 public:
  /**
   * @brief Create formatter for chat completion streaming
   * @param request_id Request ID
   * @param model Model name
   */
  ChatCompletionStreamFormatter(const std::string& request_id,
                                const std::string& model);

  /**
   * @brief Format initial role chunk
   * @param role Role (e.g., "assistant")
   * @return Formatted JSON chunk
   */
  std::string format_role(const std::string& role);

  /**
   * @brief Format content delta chunk
   * @param content Content delta
   * @return Formatted JSON chunk
   */
  std::string format_content(const std::string& content);

  /**
   * @brief Format final chunk with finish reason
   * @param finish_reason Finish reason (e.g., "stop", "length")
   * @return Formatted JSON chunk
   */
  std::string format_finish(const std::string& finish_reason);

  /**
   * @brief Format function call chunk
   * @param function_name Function name
   * @param arguments Function arguments (JSON)
   * @return Formatted JSON chunk
   */
  std::string format_function_call(const std::string& function_name,
                                   const std::string& arguments);

 private:
  std::string request_id_;
  std::string model_;
  int64_t created_;
};

/**
 * @brief Format completion chunk for streaming
 *
 * Creates OpenAI-compatible streaming chunk for text completion
 */
class CompletionStreamFormatter {
 public:
  /**
   * @brief Create formatter for completion streaming
   * @param request_id Request ID
   * @param model Model name
   */
  CompletionStreamFormatter(const std::string& request_id,
                            const std::string& model);

  /**
   * @brief Format text delta chunk
   * @param text Text delta
   * @return Formatted JSON chunk
   */
  std::string format_text(const std::string& text);

  /**
   * @brief Format final chunk with finish reason
   * @param finish_reason Finish reason
   * @return Formatted JSON chunk
   */
  std::string format_finish(const std::string& finish_reason);

 private:
  std::string request_id_;
  std::string model_;
  int64_t created_;
};

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * @brief Get current timestamp in seconds since epoch
 * @return Unix timestamp
 */
int64_t get_current_timestamp();

/**
 * @brief Escape JSON string
 * @param str String to escape
 * @return Escaped string
 */
std::string escape_json(const std::string& str);

}  // namespace server
}  // namespace mlxr
