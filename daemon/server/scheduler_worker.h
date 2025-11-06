// Copyright © 2025 MLXR Development
// Scheduler worker thread - executes batches from the scheduler

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "../../core/runtime/engine.h"
#include "../../core/runtime/sampler.h"
#include "../scheduler/scheduler.h"

namespace mlxr {
namespace server {

/**
 * @brief Worker thread that executes inference batches from the scheduler
 *
 * Continuously polls the scheduler for ready batches, executes prefill/decode
 * operations using the engine, and notifies requests via token callbacks.
 */
class SchedulerWorker {
 public:
  /**
   * @brief Construct worker with scheduler and engine
   * @param scheduler Request scheduler to poll for batches
   * @param engine Inference engine for executing requests
   */
  SchedulerWorker(std::shared_ptr<scheduler::Scheduler> scheduler,
                  std::shared_ptr<runtime::Engine> engine);

  ~SchedulerWorker();

  // Delete copy operations
  SchedulerWorker(const SchedulerWorker&) = delete;
  SchedulerWorker& operator=(const SchedulerWorker&) = delete;

  /**
   * @brief Start the worker thread
   * Begins polling scheduler and executing batches
   */
  void start();

  /**
   * @brief Stop the worker thread
   * Gracefully shuts down after completing current batch
   */
  void stop();

  /**
   * @brief Check if worker is running
   */
  bool is_running() const { return running_; }

 private:
  /**
   * @brief Main worker loop
   * Polls scheduler for batches and executes them
   */
  void run_loop();

  /**
   * @brief Execute a single batch
   * Processes prefill and decode requests
   * @param batch Batch to execute
   */
  void execute_batch(const scheduler::Batch& batch);

  /**
   * @brief Execute prefill phase for a request
   * Process the entire prompt to fill KV cache
   * @param request Request in prefill phase
   */
  void execute_prefill(scheduler::RequestPtr request);

  /**
   * @brief Execute decode phase for a request
   * Generate one token for the request
   * @param request Request in decode phase
   */
  void execute_decode(scheduler::RequestPtr request);

  // Dependencies
  std::shared_ptr<scheduler::Scheduler> scheduler_;
  std::shared_ptr<runtime::Engine> engine_;

  // Worker thread
  std::thread worker_thread_;
  std::atomic<bool> running_;
  std::atomic<bool> should_stop_;

  // KV cache management - one cache per active request
  std::unordered_map<std::string, runtime::InferenceCache> cache_map_;
  std::mutex cache_mutex_;  // Protect cache map access
};

}  // namespace server
}  // namespace mlxr
