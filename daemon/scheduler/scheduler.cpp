// Copyright © 2025 MLXR Development
// Scheduler implementation

#include "scheduler.h"

#include <algorithm>
#include <cmath>

namespace mlxr {
namespace scheduler {

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config),
      num_free_kv_blocks_(config.total_kv_blocks),
      running_(true) {
  // Initialize KV block free list
  kv_block_free_.resize(config_.total_kv_blocks, true);

  // Initialize stats
  stats_.available_kv_blocks = config_.total_kv_blocks;
  last_stats_update_ = std::chrono::steady_clock::now();
}

Scheduler::~Scheduler() { shutdown(); }

bool Scheduler::submit_request(RequestPtr request) {
  if (!running_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Check if request ID already exists
  if (all_requests_.find(request->request_id) != all_requests_.end()) {
    return false;
  }

  // Add to waiting queue
  request->state = RequestState::WAITING;
  waiting_queue_.push_back(request);
  all_requests_[request->request_id] = request;

  return true;
}

bool Scheduler::cancel_request(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = all_requests_.find(request_id);
  if (it == all_requests_.end()) {
    return false;
  }

  RequestPtr request = it->second;

  // Check if already finished/cancelled
  if (request->is_finished()) {
    return false;
  }

  // Mark as cancelled
  request->mark_completed(FinishReason::CANCELLED);

  // Free KV blocks if allocated
  free_kv_blocks(request);

  // Remove from queues
  auto remove_from_queue = [&request](std::vector<RequestPtr>& queue) {
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [&request](const RequestPtr& r) {
                                 return r->request_id == request->request_id;
                               }),
                queue.end());
  };

  auto remove_from_deque = [&request](std::deque<RequestPtr>& queue) {
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [&request](const RequestPtr& r) {
                                 return r->request_id == request->request_id;
                               }),
                queue.end());
  };

  remove_from_deque(waiting_queue_);
  remove_from_queue(prefilling_queue_);
  remove_from_queue(decoding_queue_);
  remove_from_queue(paused_queue_);

  return true;
}

Batch Scheduler::get_next_batch() {
  std::lock_guard<std::mutex> lock(mutex_);

  Batch batch;

  // Budget tracking
  int batch_tokens = 0;
  int batch_size = 0;
  int prefill_tokens = 0;

  // Priority 1: Add decoding requests (low latency, predictable compute)
  for (auto it = decoding_queue_.begin();
       it != decoding_queue_.end() && batch_size < config_.max_batch_size;) {
    RequestPtr request = *it;

    // Check if request is done
    if (request->should_stop()) {
      request->mark_completed(request->num_generated_tokens >=
                                      request->max_tokens
                                  ? FinishReason::LENGTH
                                  : FinishReason::STOP);
      free_kv_blocks(request);
      it = decoding_queue_.erase(it);
      stats_.total_requests_completed++;
      continue;
    }

    // Check token budget (each decode generates 1 token)
    if (batch_tokens + 1 <= config_.max_batch_tokens) {
      batch.decode_requests.push_back(request);
      batch_tokens += 1;
      batch_size++;
      ++it;
    } else {
      break;
    }
  }

  // Priority 2: Add prefilling requests (if budget allows)
  for (auto it = prefilling_queue_.begin();
       it != prefilling_queue_.end() && batch_size < config_.max_batch_size;) {
    RequestPtr request = *it;
    int request_tokens = request->num_prompt_tokens;

    // Check if we can fit this request
    if (prefill_tokens + request_tokens <= config_.max_prefill_tokens &&
        batch_tokens + request_tokens <= config_.max_batch_tokens) {
      batch.prefill_requests.push_back(request);
      batch_tokens += request_tokens;
      prefill_tokens += request_tokens;
      batch_size++;

      // Move to decoding queue after prefill
      move_to_decoding(request);
      it = prefilling_queue_.erase(it);
    } else {
      ++it;
    }
  }

  // Priority 3: Admit new requests from waiting queue
  while (!waiting_queue_.empty() && batch_size < config_.max_batch_size) {
    RequestPtr request = waiting_queue_.front();
    int request_tokens = request->num_prompt_tokens;

    // Check token budget
    if (prefill_tokens + request_tokens > config_.max_prefill_tokens ||
        batch_tokens + request_tokens > config_.max_batch_tokens) {
      break;
    }

    // Try to allocate KV blocks
    if (!allocate_kv_blocks(request)) {
      // Try preemption if enabled
      if (config_.enable_preemption) {
        int blocks_needed = calculate_kv_blocks_needed(
            request->num_prompt_tokens + request->max_tokens);

        if (try_preempt(blocks_needed)) {
          if (!allocate_kv_blocks(request)) {
            break;  // Still can't allocate
          }
        } else {
          break;  // Preemption failed
        }
      } else {
        break;  // Can't allocate and preemption disabled
      }
    }

    // Add to batch
    waiting_queue_.pop_front();
    request->mark_prefilling();
    batch.prefill_requests.push_back(request);
    batch_tokens += request_tokens;
    prefill_tokens += request_tokens;
    batch_size++;

    // Move to decoding after this batch
    decoding_queue_.push_back(request);
  }

  update_stats();
  return batch;
}

void Scheduler::complete_batch(const Batch& batch) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Update token counts
  stats_.total_tokens_generated += batch.decode_requests.size();

  // Prefill requests are already in decoding_queue_, nothing to do
}

bool Scheduler::allocate_kv_blocks(RequestPtr request) {
  // Calculate blocks needed for full sequence
  int max_seq_len = request->num_prompt_tokens + request->max_tokens;
  int blocks_needed = calculate_kv_blocks_needed(max_seq_len);

  if (blocks_needed > num_free_kv_blocks_) {
    return false;
  }

  // Allocate blocks
  request->kv_block_ids.clear();
  request->kv_block_ids.reserve(blocks_needed);

  for (int i = 0;
       i < static_cast<int>(kv_block_free_.size()) &&
       static_cast<int>(request->kv_block_ids.size()) < blocks_needed;
       i++) {
    if (kv_block_free_[i]) {
      kv_block_free_[i] = false;
      request->kv_block_ids.push_back(i);
    }
  }

  if (static_cast<int>(request->kv_block_ids.size()) < blocks_needed) {
    // Allocation failed, free what we allocated
    free_kv_blocks(request);
    return false;
  }

  num_free_kv_blocks_ -= blocks_needed;
  request->kv_num_blocks_needed = blocks_needed;

  return true;
}

void Scheduler::free_kv_blocks(RequestPtr request) {
  for (int block_id : request->kv_block_ids) {
    if (block_id >= 0 && block_id < static_cast<int>(kv_block_free_.size())) {
      kv_block_free_[block_id] = true;
      num_free_kv_blocks_++;
    }
  }
  request->kv_block_ids.clear();
}

bool Scheduler::try_preempt(int blocks_needed) {
  if (!config_.enable_preemption) {
    return false;
  }

  // Get candidates for preemption (lowest priority decoding requests)
  auto candidates = select_preemption_candidates(blocks_needed);

  int blocks_freed = 0;
  for (auto& candidate : candidates) {
    free_kv_blocks(candidate);
    blocks_freed += candidate->kv_num_blocks_needed;

    // Move to paused queue
    candidate->state = RequestState::PAUSED;
    paused_queue_.push_back(candidate);

    // Remove from decoding queue
    decoding_queue_.erase(
        std::remove_if(decoding_queue_.begin(), decoding_queue_.end(),
                       [&candidate](const RequestPtr& r) {
                         return r->request_id == candidate->request_id;
                       }),
        decoding_queue_.end());

    if (blocks_freed >= blocks_needed) {
      return true;
    }
  }

  return blocks_freed >= blocks_needed;
}

std::vector<RequestPtr> Scheduler::select_preemption_candidates(
    int blocks_needed) {
  std::vector<RequestPtr> candidates;

  // Sort decoding requests by priority (ascending) and tokens generated
  // (descending)
  std::vector<RequestPtr> sortable_requests = decoding_queue_;

  std::sort(sortable_requests.begin(), sortable_requests.end(),
            [this](const RequestPtr& a, const RequestPtr& b) {
              // Skip requests that haven't generated enough tokens
              bool a_eligible = a->num_generated_tokens >=
                                config_.min_decode_steps_before_preempt;
              bool b_eligible = b->num_generated_tokens >=
                                config_.min_decode_steps_before_preempt;

              if (a_eligible != b_eligible) {
                return b_eligible;  // Eligible requests go first (to end of
                                    // candidates)
              }

              if (a->priority != b->priority) {
                return a->priority < b->priority;  // Lower priority first
              }

              return a->num_generated_tokens >
                     b->num_generated_tokens;  // More progress first
            });

  // Select candidates until we have enough blocks
  int blocks_accumulated = 0;
  for (auto& request : sortable_requests) {
    if (request->num_generated_tokens <
        config_.min_decode_steps_before_preempt) {
      continue;
    }

    candidates.push_back(request);
    blocks_accumulated += request->kv_num_blocks_needed;

    if (blocks_accumulated >= blocks_needed) {
      break;
    }
  }

  return candidates;
}

SchedulerStats Scheduler::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  update_stats();
  return stats_;
}

RequestPtr Scheduler::get_request(const std::string& request_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = all_requests_.find(request_id);
  if (it != all_requests_.end()) {
    return it->second;
  }
  return nullptr;
}

void Scheduler::shutdown() {
  running_ = false;

  std::lock_guard<std::mutex> lock(mutex_);

  // Cancel all pending requests
  for (auto& [id, request] : all_requests_) {
    if (!request->is_finished()) {
      request->mark_completed(FinishReason::CANCELLED);
      free_kv_blocks(request);
    }
  }

  // Clear queues
  waiting_queue_.clear();
  prefilling_queue_.clear();
  decoding_queue_.clear();
  paused_queue_.clear();
}

// Private helper methods

bool Scheduler::can_add_to_batch(const Batch& batch,
                                 const RequestPtr& request) const {
  int current_tokens = batch.total_tokens();
  int request_tokens =
      request->is_prefill_phase() ? request->num_prompt_tokens : 1;

  return current_tokens + request_tokens <= config_.max_batch_tokens &&
         batch.size() < static_cast<size_t>(config_.max_batch_size);
}

int Scheduler::calculate_kv_blocks_needed(int num_tokens) const {
  return (num_tokens + config_.kv_block_size - 1) / config_.kv_block_size;
}

void Scheduler::move_to_prefilling(RequestPtr request) {
  request->mark_prefilling();
  prefilling_queue_.push_back(request);
}

void Scheduler::move_to_decoding(RequestPtr request) {
  request->mark_decoding();
  // Already in decoding_queue_ from get_next_batch
}

void Scheduler::move_to_waiting(RequestPtr request) {
  request->state = RequestState::WAITING;
  waiting_queue_.push_back(request);
}

void Scheduler::update_stats() const {
  stats_.waiting_requests = waiting_queue_.size();
  stats_.prefilling_requests = prefilling_queue_.size();
  stats_.decoding_requests = decoding_queue_.size();
  stats_.paused_requests = paused_queue_.size();

  stats_.used_kv_blocks = config_.total_kv_blocks - num_free_kv_blocks_;
  stats_.available_kv_blocks = num_free_kv_blocks_;
  stats_.kv_utilization =
      static_cast<float>(stats_.used_kv_blocks) / config_.total_kv_blocks;

  // Calculate throughput
  auto now = std::chrono::steady_clock::now();
  double elapsed_s =
      std::chrono::duration<double>(now - last_stats_update_).count();

  if (elapsed_s > 0.1) {  // Update every 100ms
    // Average latencies
    double total_queue_time = 0.0;
    int queue_count = 0;

    for (const auto& request : prefilling_queue_) {
      total_queue_time += request->queue_time_ms();
      queue_count++;
    }
    for (const auto& request : decoding_queue_) {
      total_queue_time += request->queue_time_ms();
      queue_count++;
    }

    if (queue_count > 0) {
      stats_.avg_queue_time_ms = total_queue_time / queue_count;
    }

    last_stats_update_ = now;
  }
}

}  // namespace scheduler
}  // namespace mlxr
