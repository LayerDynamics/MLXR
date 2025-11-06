// Copyright © 2025 MLXR Development
// Request structure for scheduler

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace mlxr {
namespace scheduler {

// Request state through the system
enum class RequestState {
  WAITING,     // Waiting in queue
  PREFILLING,  // Processing prompt
  DECODING,    // Generating tokens
  PAUSED,      // Temporarily paused (e.g., for KV eviction)
  COMPLETED,   // Generation finished
  CANCELLED,   // Request cancelled
  FAILED       // Request failed with error
};

// Finish reason for completed requests
enum class FinishReason {
  NONE,       // Still generating
  STOP,       // Hit stop token
  LENGTH,     // Reached max_tokens
  EOS,        // End-of-sequence token
  CANCELLED,  // User cancelled
  ERROR       // Internal error
};

// Sampling parameters
struct SamplingParams {
  float temperature = 0.7f;         // More focused sampling (was 1.0f)
  float top_p = 0.9f;               // Nucleus sampling enabled (was 1.0f)
  int top_k = 40;                   // Top-k filtering enabled (was 0)
  float repetition_penalty = 1.1f;  // Slight repetition penalty (was 1.0f)
  int max_tokens = 512;
  std::vector<int> stop_token_ids;
  bool logprobs = false;
  int top_logprobs = 0;
};

// Generation request
class Request {
 public:
  // Request metadata
  std::string request_id;
  std::string prompt;
  std::vector<int> prompt_token_ids;
  SamplingParams sampling_params;

  // State tracking
  RequestState state;
  FinishReason finish_reason;
  std::string error_message;

  // Generation progress
  std::vector<int> generated_token_ids;
  int num_prompt_tokens;
  int num_generated_tokens;
  int max_tokens;

  // KV cache assignment
  std::vector<int> kv_block_ids;  // Physical block IDs assigned to this request
  int kv_num_blocks_needed;

  // Timing
  std::chrono::steady_clock::time_point arrival_time;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_token_time;
  std::chrono::steady_clock::time_point finish_time;

  // Priority (higher = more important)
  int priority;

  // Stream callback (called when new token is generated)
  using TokenCallback = std::function<void(int token_id, bool finished)>;
  TokenCallback token_callback;

  // Constructor
  Request(const std::string& id, const std::string& prompt_text,
          const std::vector<int>& tokens, const SamplingParams& params)
      : request_id(id),
        prompt(prompt_text),
        prompt_token_ids(tokens),
        sampling_params(params),
        state(RequestState::WAITING),
        finish_reason(FinishReason::NONE),
        num_prompt_tokens(tokens.size()),
        num_generated_tokens(0),
        max_tokens(params.max_tokens),
        kv_num_blocks_needed(0),
        arrival_time(std::chrono::steady_clock::now()),
        priority(0) {}

  // Getters
  int total_tokens() const { return num_prompt_tokens + num_generated_tokens; }

  bool is_prefill_phase() const {
    return state == RequestState::PREFILLING ||
           (state == RequestState::WAITING && num_generated_tokens == 0);
  }

  bool is_decode_phase() const {
    return state == RequestState::DECODING && num_generated_tokens > 0;
  }

  bool is_finished() const {
    return state == RequestState::COMPLETED ||
           state == RequestState::CANCELLED || state == RequestState::FAILED;
  }

  bool should_stop() const {
    if (num_generated_tokens >= max_tokens) {
      return true;
    }

    if (generated_token_ids.empty()) {
      return false;
    }

    int last_token = generated_token_ids.back();
    for (int stop_token : sampling_params.stop_token_ids) {
      if (last_token == stop_token) {
        return true;
      }
    }

    return false;
  }

  // State transitions
  void mark_prefilling() {
    state = RequestState::PREFILLING;
    start_time = std::chrono::steady_clock::now();
  }

  void mark_decoding() {
    state = RequestState::DECODING;
    if (start_time == std::chrono::steady_clock::time_point{}) {
      start_time = std::chrono::steady_clock::now();
    }
  }

  void mark_completed(FinishReason reason) {
    state = RequestState::COMPLETED;
    finish_reason = reason;
    finish_time = std::chrono::steady_clock::now();
  }

  void mark_failed(const std::string& error) {
    state = RequestState::FAILED;
    finish_reason = FinishReason::ERROR;
    error_message = error;
    finish_time = std::chrono::steady_clock::now();
  }

  void add_generated_token(int token_id) {
    generated_token_ids.push_back(token_id);
    num_generated_tokens++;
    last_token_time = std::chrono::steady_clock::now();

    if (token_callback) {
      token_callback(token_id, should_stop());
    }
  }

  // Timing metrics
  double elapsed_ms() const {
    if (start_time == std::chrono::steady_clock::time_point{}) {
      return 0.0;
    }

    auto end = is_finished() ? finish_time : std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_time).count();
  }

  double queue_time_ms() const {
    if (start_time == std::chrono::steady_clock::time_point{}) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - arrival_time)
          .count();
    }
    return std::chrono::duration<double, std::milli>(start_time - arrival_time)
        .count();
  }

  double tokens_per_second() const {
    double elapsed = elapsed_ms();
    if (elapsed < 1.0) {
      return 0.0;
    }
    return (num_generated_tokens * 1000.0) / elapsed;
  }
};

using RequestPtr = std::shared_ptr<Request>;

}  // namespace scheduler
}  // namespace mlxr
