/**
 * @file scheduler_test.cpp
 * @brief Unit tests for the request scheduler
 */

#include "scheduler/scheduler.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "scheduler/request.h"

using namespace mlxr::scheduler;

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default config for testing
    config.max_batch_tokens = 2048;
    config.max_batch_size = 32;
    config.kv_block_size = 16;
    config.total_kv_blocks = 1024;
  }

  void TearDown() override {
    // Cleanup
  }

  SchedulerConfig config;

  // Helper to create a simple request
  RequestPtr create_request(const std::string& id, int num_tokens,
                            int max_gen = 10) {
    SamplingParams params;
    params.max_tokens = max_gen;

    std::vector<int> tokens(num_tokens, 1);  // Fill with dummy token IDs

    return std::make_shared<Request>(id, "test prompt", tokens, params);
  }
};

// ============================================================================
// Basic Scheduler Tests
// ============================================================================

TEST_F(SchedulerTest, Construction) {
  EXPECT_NO_THROW({ Scheduler scheduler(config); });
}

TEST_F(SchedulerTest, SubmitRequest) {
  Scheduler scheduler(config);

  auto request = create_request("test_request_1", 5);

  bool accepted = scheduler.submit_request(request);

  // Request should be accepted
  EXPECT_TRUE(accepted);

  // Request should be in WAITING state initially
  EXPECT_EQ(request->state, RequestState::WAITING);
}

TEST_F(SchedulerTest, SubmitMultipleRequests) {
  Scheduler scheduler(config);

  for (int i = 0; i < 5; ++i) {
    auto request = create_request("request_" + std::to_string(i), 10);
    bool accepted = scheduler.submit_request(request);
    EXPECT_TRUE(accepted);
  }

  // All requests should be added successfully
}

TEST_F(SchedulerTest, GetNextBatch) {
  Scheduler scheduler(config);

  // Add a small request
  auto request = create_request("batch_test", 100);
  scheduler.submit_request(request);

  // Get next batch
  auto batch = scheduler.get_next_batch();

  // Should return the request for prefill
  EXPECT_FALSE(batch.empty());
  EXPECT_FALSE(batch.prefill_requests.empty());
  EXPECT_EQ(batch.prefill_requests[0]->request_id, "batch_test");
}

TEST_F(SchedulerTest, CancelRequest) {
  Scheduler scheduler(config);

  auto request = create_request("cancel_test", 10);
  scheduler.submit_request(request);

  // Cancel the request
  bool cancelled = scheduler.cancel_request("cancel_test");
  EXPECT_TRUE(cancelled);

  // Trying to cancel again should return false
  bool cancelled_again = scheduler.cancel_request("cancel_test");
  EXPECT_FALSE(cancelled_again);
}

TEST_F(SchedulerTest, GetStats) {
  Scheduler scheduler(config);

  // Add some requests
  for (int i = 0; i < 3; ++i) {
    auto request = create_request("stats_test_" + std::to_string(i), 10);
    scheduler.submit_request(request);
  }

  // Get stats
  auto stats = scheduler.get_stats();

  // Should have requests in waiting queue
  EXPECT_GT(stats.waiting_requests, 0);
}

TEST_F(SchedulerTest, GetRequestById) {
  Scheduler scheduler(config);

  auto request = create_request("find_me", 10);
  scheduler.submit_request(request);

  // Find the request
  auto found = scheduler.get_request("find_me");
  EXPECT_NE(found, nullptr);
  EXPECT_EQ(found->request_id, "find_me");

  // Non-existent request
  auto not_found = scheduler.get_request("doesnt_exist");
  EXPECT_EQ(not_found, nullptr);
}

TEST_F(SchedulerTest, ShutdownScheduler) {
  Scheduler scheduler(config);

  EXPECT_TRUE(scheduler.is_running());

  scheduler.shutdown();

  EXPECT_FALSE(scheduler.is_running());
}

// ============================================================================
// KV Cache Block Management Tests
// ============================================================================

TEST_F(SchedulerTest, AllocateKVBlocks) {
  Scheduler scheduler(config);

  auto request = create_request("kv_test", 50);  // Need 50 / 16 = 4 blocks

  // Allocate blocks
  bool allocated = scheduler.allocate_kv_blocks(request);
  EXPECT_TRUE(allocated);

  // Request should have blocks assigned
  EXPECT_GT(request->kv_block_ids.size(), 0);
}

TEST_F(SchedulerTest, FreeKVBlocks) {
  Scheduler scheduler(config);

  auto request = create_request("free_test", 50);

  // Allocate and then free
  scheduler.allocate_kv_blocks(request);
  EXPECT_GT(request->kv_block_ids.size(), 0);

  scheduler.free_kv_blocks(request);
  // After freeing, blocks list should be cleared (implementation-dependent)
}

TEST_F(SchedulerTest, KVBlockExhaustion) {
  SchedulerConfig limited_config;
  limited_config.max_batch_tokens = 2048;
  limited_config.max_batch_size = 32;
  limited_config.kv_block_size = 16;
  limited_config.total_kv_blocks = 10;  // Only 10 blocks

  Scheduler scheduler(limited_config);

  // Try to allocate more blocks than available
  std::vector<RequestPtr> requests;
  int successful_allocations = 0;

  for (int i = 0; i < 20; ++i) {
    auto request = create_request("block_test_" + std::to_string(i), 16);
    if (scheduler.allocate_kv_blocks(request)) {
      successful_allocations++;
      requests.push_back(request);
    }
  }

  // Should not allocate more than available blocks
  EXPECT_LE(successful_allocations, 10);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(SchedulerTest, ConcurrentSubmitRequests) {
  Scheduler scheduler(config);

  const int num_threads = 4;
  const int requests_per_thread = 25;
  std::atomic<int> total_accepted{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        [&scheduler, t, requests_per_thread, &total_accepted]() {
          for (int i = 0; i < requests_per_thread; ++i) {
            std::string id =
                "thread_" + std::to_string(t) + "_req_" + std::to_string(i);
            auto request = std::make_shared<Request>(
                id, "test", std::vector<int>{1, 2, 3}, SamplingParams());

            if (scheduler.submit_request(request)) {
              total_accepted++;
            }

            // Small delay to increase interleaving
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // All requests should be accepted
  EXPECT_EQ(total_accepted, num_threads * requests_per_thread);
}
