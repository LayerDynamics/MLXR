// Copyright © 2025 MLXR Development
// Continuous batching scheduler with prefill/decode queues

#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "request.h"

namespace mlxr {
namespace scheduler {

// Batch of requests for execution
struct Batch {
  std::vector<RequestPtr> prefill_requests;  // Requests in prefill phase
  std::vector<RequestPtr> decode_requests;   // Requests in decode phase

  bool empty() const {
    return prefill_requests.empty() && decode_requests.empty();
  }

  size_t size() const {
    return prefill_requests.size() + decode_requests.size();
  }

  int total_tokens() const {
    int count = 0;
    for (const auto& req : prefill_requests) {
      count += req->num_prompt_tokens;
    }
    count += decode_requests.size();  // Each decode request generates 1 token
    return count;
  }
};

// Scheduler configuration
struct SchedulerConfig {
  // Token budget constraints
  int max_batch_tokens = 8192;    // Maximum tokens per batch
  int max_batch_size = 128;       // Maximum requests per batch
  int max_prefill_tokens = 4096;  // Maximum prefill tokens per batch

  // KV cache management
  int total_kv_blocks = 1024;  // Total KV cache blocks available
  int kv_block_size = 16;      // Tokens per KV block

  // Chunking for long prompts
  int max_prefill_chunk_size = 2048;  // Chunk long prefills
  bool enable_chunked_prefill = true;

  // Priority and fairness
  bool enable_priority_scheduling = true;
  float decode_preference = 2.0f;  // Prefer decode over prefill (lower latency)

  // Speculative decoding
  bool enable_speculative_decoding = false;
  int speculation_length = 4;

  // Preemption policy
  bool enable_preemption = true;
  int min_decode_steps_before_preempt = 10;  // Minimum tokens before preemption
};

// Scheduler statistics
struct SchedulerStats {
  // Queue depths
  size_t waiting_requests = 0;
  size_t prefilling_requests = 0;
  size_t decoding_requests = 0;
  size_t paused_requests = 0;

  // KV cache utilization
  int used_kv_blocks = 0;
  int available_kv_blocks = 0;
  float kv_utilization = 0.0f;

  // Throughput
  double tokens_per_second = 0.0;
  double requests_per_second = 0.0;

  // Latency
  double avg_queue_time_ms = 0.0;
  double avg_prefill_time_ms = 0.0;
  double avg_decode_latency_ms = 0.0;  // Time per token

  // Totals
  uint64_t total_requests_completed = 0;
  uint64_t total_tokens_generated = 0;
};

// Main scheduler class
class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config);
  ~Scheduler();

  /**
   * Submit a new request to the scheduler
   * @param request Request to add
   * @return true if accepted, false if rejected (e.g., queue full)
   */
  bool submit_request(RequestPtr request);

  /**
   * Cancel a pending or running request
   * @param request_id Request ID to cancel
   * @return true if found and cancelled
   */
  bool cancel_request(const std::string& request_id);

  /**
   * Get the next batch of requests to execute
   * This implements continuous batching logic
   * @return Batch to execute (may be empty if no work)
   */
  Batch get_next_batch();

  /**
   * Mark requests in the batch as completed and free their resources
   * Called after executing a batch
   * @param batch The batch that was just executed
   */
  void complete_batch(const Batch& batch);

  /**
   * Allocate KV cache blocks for a request
   * @param request Request needing KV blocks
   * @return true if allocation succeeded
   */
  bool allocate_kv_blocks(RequestPtr request);

  /**
   * Free KV cache blocks for a request
   * @param request Request to free blocks for
   */
  void free_kv_blocks(RequestPtr request);

  /**
   * Try to preempt lower-priority requests to free KV blocks
   * @param blocks_needed Number of blocks needed
   * @return true if enough blocks were freed
   */
  bool try_preempt(int blocks_needed);

  /**
   * Get current scheduler statistics
   */
  SchedulerStats get_stats() const;

  /**
   * Get request by ID
   */
  RequestPtr get_request(const std::string& request_id) const;

  /**
   * Shutdown scheduler (stop accepting new requests)
   */
  void shutdown();

  /**
   * Check if scheduler is running
   */
  bool is_running() const { return running_; }

 private:
  // Configuration
  SchedulerConfig config_;

  // Request storage
  mutable std::mutex mutex_;
  std::unordered_map<std::string, RequestPtr> all_requests_;

  // Queue structures
  std::deque<RequestPtr> waiting_queue_;      // Waiting to start
  std::vector<RequestPtr> prefilling_queue_;  // Currently prefilling
  std::vector<RequestPtr> decoding_queue_;    // Currently decoding
  std::vector<RequestPtr> paused_queue_;      // Temporarily paused

  // KV cache tracking
  std::vector<bool> kv_block_free_;  // Free list for KV blocks
  int num_free_kv_blocks_;

  // Statistics (mutable to allow updating in const methods)
  mutable SchedulerStats stats_;
  mutable std::chrono::steady_clock::time_point last_stats_update_;

  // State
  std::atomic<bool> running_;

  // Helper methods
  bool can_add_to_batch(const Batch& batch, const RequestPtr& request) const;
  int calculate_kv_blocks_needed(int num_tokens) const;
  void move_to_prefilling(RequestPtr request);
  void move_to_decoding(RequestPtr request);
  void move_to_waiting(RequestPtr request);
  void update_stats() const;
  std::vector<RequestPtr> select_preemption_candidates(int blocks_needed);
};

}  // namespace scheduler
}  // namespace mlxr
