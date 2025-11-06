/**
 * @file scheduler_worker_test.cpp
 * @brief Unit tests for the scheduler worker
 */

#include "server/scheduler_worker.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "scheduler/scheduler.h"

using namespace mlxr;
using namespace mlxr::scheduler;
using namespace mlxr::server;

class SchedulerWorkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create scheduler config
    SchedulerConfig config;
    config.max_batch_tokens = 2048;
    config.max_batch_size = 32;
    config.kv_block_size = 16;
    config.total_kv_blocks = 1024;

    scheduler = std::make_shared<Scheduler>(config);
  }

  void TearDown() override {
    // Stop worker if running
    if (worker && worker->is_running()) {
      worker->stop();
    }
  }

  // Helper to create a simple request
  RequestPtr create_request(const std::string& id, int num_tokens = 5,
                            int max_gen = 10) {
    SamplingParams params;
    params.max_tokens = max_gen;

    std::vector<int> tokens(num_tokens, 1);

    return std::make_shared<Request>(id, "test prompt", tokens, params);
  }

  std::shared_ptr<Scheduler> scheduler;
  std::unique_ptr<SchedulerWorker> worker;
};

// ============================================================================
// Basic Worker Tests
// ============================================================================

TEST_F(SchedulerWorkerTest, Construction) {
  EXPECT_NO_THROW(
      { worker = std::make_unique<SchedulerWorker>(scheduler, nullptr); });
}

TEST_F(SchedulerWorkerTest, StartStop) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);

  // Worker should not be running initially
  EXPECT_FALSE(worker->is_running());

  // Start worker
  worker->start();
  EXPECT_TRUE(worker->is_running());

  // Stop worker
  worker->stop();
  EXPECT_FALSE(worker->is_running());
}

TEST_F(SchedulerWorkerTest, MultipleStartStop) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);

  // Start and stop multiple times
  for (int i = 0; i < 3; ++i) {
    worker->start();
    EXPECT_TRUE(worker->is_running());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    worker->stop();
    EXPECT_FALSE(worker->is_running());
  }
}

TEST_F(SchedulerWorkerTest, WorkerThreadRunning) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);

  worker->start();

  // Worker thread should be running
  EXPECT_TRUE(worker->is_running());

  // Let it run for a bit
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Should still be running
  EXPECT_TRUE(worker->is_running());

  worker->stop();
}

// ============================================================================
// Request Processing Tests (Mock Mode - No Engine)
// ============================================================================

TEST_F(SchedulerWorkerTest, ProcessRequestsNoEngine) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);
  worker->start();

  // Add a request
  auto request = create_request("test_no_engine", 5, 3);
  scheduler->submit_request(request);

  // Wait for worker to process
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Without engine, worker should handle gracefully (no crash)
  EXPECT_TRUE(worker->is_running());

  worker->stop();
}

TEST_F(SchedulerWorkerTest, MultipleRequestsNoEngine) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);
  worker->start();

  // Add multiple requests
  for (int i = 0; i < 5; ++i) {
    auto request = create_request("multi_test_" + std::to_string(i), 3, 2);
    scheduler->submit_request(request);
  }

  // Wait for worker to process
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Worker should handle multiple requests gracefully
  EXPECT_TRUE(worker->is_running());

  worker->stop();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(SchedulerWorkerTest, StopWhileProcessing) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);
  worker->start();

  // Add requests
  for (int i = 0; i < 10; ++i) {
    auto request = create_request("stop_test_" + std::to_string(i), 5, 10);
    scheduler->submit_request(request);
  }

  // Stop immediately while processing
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  worker->stop();

  // Should stop cleanly without hanging
  EXPECT_FALSE(worker->is_running());
}

// ============================================================================
// Lifecycle Tests
// ============================================================================

TEST_F(SchedulerWorkerTest, RepeatedStartStopCycle) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);

  for (int cycle = 0; cycle < 5; ++cycle) {
    // Start
    worker->start();
    EXPECT_TRUE(worker->is_running());

    // Add a request
    auto request = create_request("cycle_" + std::to_string(cycle), 3, 1);
    scheduler->submit_request(request);

    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Stop
    worker->stop();
    EXPECT_FALSE(worker->is_running());

    // Small pause between cycles
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

TEST_F(SchedulerWorkerTest, ShutdownSchedulerWhileWorkerRunning) {
  worker = std::make_unique<SchedulerWorker>(scheduler, nullptr);
  worker->start();

  // Let worker run
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Shutdown scheduler
  scheduler->shutdown();

  // Worker should continue to run (it polls the scheduler)
  // But scheduler won't accept new requests
  EXPECT_TRUE(worker->is_running());

  worker->stop();
}
