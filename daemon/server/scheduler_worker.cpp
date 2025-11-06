// Copyright © 2025 MLXR Development
// Scheduler worker implementation

#include "scheduler_worker.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "../scheduler/request.h"

namespace mlxr {
namespace server {

SchedulerWorker::SchedulerWorker(
    std::shared_ptr<scheduler::Scheduler> scheduler,
    std::shared_ptr<runtime::Engine> engine)
    : scheduler_(scheduler),
      engine_(engine),
      running_(false),
      should_stop_(false) {}

SchedulerWorker::~SchedulerWorker() { stop(); }

void SchedulerWorker::start() {
  if (running_) {
    return;
  }

  should_stop_ = false;
  running_ = true;

  worker_thread_ = std::thread(&SchedulerWorker::run_loop, this);
}

void SchedulerWorker::stop() {
  if (!running_) {
    return;
  }

  should_stop_ = true;

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }

  running_ = false;
}

void SchedulerWorker::run_loop() {
  std::cout << "[SchedulerWorker] Worker thread started" << std::endl;

  while (!should_stop_) {
    // Get next batch from scheduler
    scheduler::Batch batch = scheduler_->get_next_batch();

    if (batch.empty()) {
      // No work available, sleep briefly
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // Execute the batch
    execute_batch(batch);

    // Notify scheduler that batch is complete
    scheduler_->complete_batch(batch);
  }

  std::cout << "[SchedulerWorker] Worker thread stopped" << std::endl;
}

void SchedulerWorker::execute_batch(const scheduler::Batch& batch) {
  // Execute prefill requests first (process entire prompt)
  for (const auto& request : batch.prefill_requests) {
    if (should_stop_) {
      break;
    }

    try {
      execute_prefill(request);
    } catch (const std::exception& e) {
      std::cerr << "[SchedulerWorker] Prefill failed for request "
                << request->request_id << ": " << e.what() << std::endl;
      request->mark_failed(e.what());
    }
  }

  // Execute decode requests (generate one token each)
  for (const auto& request : batch.decode_requests) {
    if (should_stop_) {
      break;
    }

    try {
      execute_decode(request);
    } catch (const std::exception& e) {
      std::cerr << "[SchedulerWorker] Decode failed for request "
                << request->request_id << ": " << e.what() << std::endl;
      request->mark_failed(e.what());
    }
  }
}

void SchedulerWorker::execute_prefill(scheduler::RequestPtr request) {
  request->mark_prefilling();

  try {
    // If no engine is available, skip inference (for testing)
    if (!engine_) {
      request->mark_completed(scheduler::FinishReason::STOP);
      return;
    }

    // Get or create cache for this request
    runtime::InferenceCache* cache = nullptr;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cache = &cache_map_[request->request_id];
    }

    // Single forward pass for prefill - processes all prompt tokens
    auto logits = engine_->forward_prefill(request->prompt_token_ids, cache);

    // Configure sampler with request parameters
    runtime::SamplerConfig sampler_config;
    sampler_config.temperature = request->sampling_params.temperature;
    sampler_config.top_p = request->sampling_params.top_p;
    sampler_config.top_k = request->sampling_params.top_k;
    sampler_config.repetition_penalty =
        request->sampling_params.repetition_penalty;

    runtime::Sampler sampler(sampler_config);

    // Sample ONE token from logits
    int next_token = sampler.sample(logits, request->prompt_token_ids);

    // Add to request (this calls the token_callback)
    request->add_generated_token(next_token);

    // Transition to decoding phase
    request->mark_decoding();

    // Check if finished after first token
    if (request->should_stop()) {
      scheduler::FinishReason reason = scheduler::FinishReason::STOP;

      if (request->num_generated_tokens >= request->max_tokens) {
        reason = scheduler::FinishReason::LENGTH;
      }

      request->mark_completed(reason);

      // Clean up cache
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cache_map_.erase(request->request_id);
    }

  } catch (const std::exception& e) {
    // Clean up cache on error
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_map_.erase(request->request_id);
    throw;  // Re-throw for caller to handle
  }
}

void SchedulerWorker::execute_decode(scheduler::RequestPtr request) {
  try {
    // If no engine is available, skip inference (for testing)
    if (!engine_) {
      request->mark_completed(scheduler::FinishReason::STOP);
      return;
    }

    // Get existing cache for this request
    runtime::InferenceCache* cache = nullptr;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto it = cache_map_.find(request->request_id);
      if (it == cache_map_.end()) {
        throw std::runtime_error("No cache found for request " +
                                 request->request_id);
      }
      cache = &it->second;
    }

    // Get the last generated token
    if (request->generated_token_ids.empty()) {
      throw std::runtime_error("No tokens generated yet for decode phase");
    }
    int last_token = request->generated_token_ids.back();

    // Single forward pass for decode - processes ONE token with existing cache
    auto logits = engine_->forward_decode(last_token, cache);

    // Configure sampler
    runtime::SamplerConfig sampler_config;
    sampler_config.temperature = request->sampling_params.temperature;
    sampler_config.top_p = request->sampling_params.top_p;
    sampler_config.top_k = request->sampling_params.top_k;
    sampler_config.repetition_penalty =
        request->sampling_params.repetition_penalty;

    runtime::Sampler sampler(sampler_config);

    // Build context for repetition penalty (prompt + generated so far)
    std::vector<int> context = request->prompt_token_ids;
    context.insert(context.end(), request->generated_token_ids.begin(),
                   request->generated_token_ids.end());

    // Sample ONE token from logits
    int next_token = sampler.sample(logits, context);

    // Add to request (this calls the token_callback)
    request->add_generated_token(next_token);

    // Check if finished
    if (request->should_stop()) {
      scheduler::FinishReason reason = scheduler::FinishReason::STOP;

      if (request->num_generated_tokens >= request->max_tokens) {
        reason = scheduler::FinishReason::LENGTH;
      }

      request->mark_completed(reason);

      // Clean up cache
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cache_map_.erase(request->request_id);
    }

  } catch (const std::exception& e) {
    // Clean up cache on error
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_map_.erase(request->request_id);
    throw;  // Re-throw for caller to handle
  }
}

}  // namespace server
}  // namespace mlxr
