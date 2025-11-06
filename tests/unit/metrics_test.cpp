// Copyright © 2025 MLXR Development
// Metrics system unit tests

#include "telemetry/metrics.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

using namespace mlxr::telemetry;

namespace {

// ==============================================================================
// Counter Tests
// ==============================================================================

TEST(CounterTest, InitialValue) {
  Counter counter;
  EXPECT_EQ(counter.value(), 0);
}

TEST(CounterTest, Increment) {
  Counter counter;
  counter.increment();
  EXPECT_EQ(counter.value(), 1);

  counter.increment(5);
  EXPECT_EQ(counter.value(), 6);
}

TEST(CounterTest, Reset) {
  Counter counter;
  counter.increment(10);
  EXPECT_EQ(counter.value(), 10);

  counter.reset();
  EXPECT_EQ(counter.value(), 0);
}

TEST(CounterTest, ThreadSafety) {
  Counter counter;
  const int num_threads = 10;
  const int increments_per_thread = 1000;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([&counter, increments_per_thread]() {
      for (int j = 0; j < increments_per_thread; j++) {
        counter.increment();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(counter.value(), num_threads * increments_per_thread);
}

// ==============================================================================
// Gauge Tests
// ==============================================================================

TEST(GaugeTest, InitialValue) {
  Gauge gauge;
  EXPECT_EQ(gauge.value(), 0);
}

TEST(GaugeTest, Set) {
  Gauge gauge;
  gauge.set(42);
  EXPECT_EQ(gauge.value(), 42);

  gauge.set(100);
  EXPECT_EQ(gauge.value(), 100);
}

TEST(GaugeTest, Increment) {
  Gauge gauge;
  gauge.set(10);
  gauge.increment(5);
  EXPECT_EQ(gauge.value(), 15);
}

TEST(GaugeTest, Decrement) {
  Gauge gauge;
  gauge.set(10);
  gauge.decrement(3);
  EXPECT_EQ(gauge.value(), 7);
}

TEST(GaugeTest, ThreadSafety) {
  Gauge gauge;
  gauge.set(0);

  const int num_threads = 10;
  const int operations_per_thread = 1000;

  std::vector<std::thread> threads;

  // Half increment, half decrement
  for (int i = 0; i < num_threads / 2; i++) {
    threads.emplace_back([&gauge, operations_per_thread]() {
      for (int j = 0; j < operations_per_thread; j++) {
        gauge.increment();
      }
    });
  }

  for (int i = 0; i < num_threads / 2; i++) {
    threads.emplace_back([&gauge, operations_per_thread]() {
      for (int j = 0; j < operations_per_thread; j++) {
        gauge.decrement();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(gauge.value(), 0);
}

// ==============================================================================
// Histogram Tests
// ==============================================================================

TEST(HistogramTest, InitialStats) {
  Histogram histogram;
  auto stats = histogram.get_stats();

  EXPECT_EQ(stats.count, 0);
  EXPECT_EQ(stats.sum, 0.0);
  EXPECT_EQ(stats.min, 0.0);
  EXPECT_EQ(stats.max, 0.0);
  EXPECT_EQ(stats.mean, 0.0);
}

TEST(HistogramTest, Observe) {
  Histogram histogram;
  histogram.observe(10.0);
  histogram.observe(20.0);
  histogram.observe(30.0);

  auto stats = histogram.get_stats();
  EXPECT_EQ(stats.count, 3);
  EXPECT_DOUBLE_EQ(stats.sum, 60.0);
  EXPECT_DOUBLE_EQ(stats.min, 10.0);
  EXPECT_DOUBLE_EQ(stats.max, 30.0);
  EXPECT_DOUBLE_EQ(stats.mean, 20.0);
}

TEST(HistogramTest, Percentiles) {
  Histogram histogram;

  // Add 100 values from 1 to 100
  for (int i = 1; i <= 100; i++) {
    histogram.observe(static_cast<double>(i));
  }

  auto stats = histogram.get_stats();
  EXPECT_EQ(stats.count, 100);

  // Check percentiles (approximate due to ceiling calculation)
  EXPECT_NEAR(stats.p50, 50.0, 1.0);
  EXPECT_NEAR(stats.p95, 95.0, 1.0);
  EXPECT_NEAR(stats.p99, 99.0, 1.0);
}

TEST(HistogramTest, Reset) {
  Histogram histogram;
  histogram.observe(10.0);
  histogram.observe(20.0);

  histogram.reset();

  auto stats = histogram.get_stats();
  EXPECT_EQ(stats.count, 0);
  EXPECT_EQ(stats.sum, 0.0);
}

// ==============================================================================
// Timer Tests
// ==============================================================================

TEST(TimerTest, MeasuresDuration) {
  Histogram histogram;

  {
    Timer timer(histogram);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto stats = histogram.get_stats();
  EXPECT_EQ(stats.count, 1);
  EXPECT_GT(stats.sum, 9.0);   // Should be at least 9ms
  EXPECT_LT(stats.sum, 50.0);  // Should be less than 50ms
}

// ==============================================================================
// MetricsRegistry Tests
// ==============================================================================

TEST(MetricsRegistryTest, RegisterCounter) {
  auto& registry = MetricsRegistry::instance();

  auto* counter = registry.register_counter("test_counter", "Test counter");
  ASSERT_NE(counter, nullptr);
  EXPECT_EQ(counter->value(), 0);

  counter->increment(5);
  EXPECT_EQ(counter->value(), 5);

  // Getting same counter should return same instance
  auto* same_counter = registry.get_counter("test_counter");
  EXPECT_EQ(same_counter, counter);
  EXPECT_EQ(same_counter->value(), 5);
}

TEST(MetricsRegistryTest, RegisterGauge) {
  auto& registry = MetricsRegistry::instance();

  auto* gauge = registry.register_gauge("test_gauge", "Test gauge");
  ASSERT_NE(gauge, nullptr);

  gauge->set(42);
  EXPECT_EQ(gauge->value(), 42);

  auto* same_gauge = registry.get_gauge("test_gauge");
  EXPECT_EQ(same_gauge, gauge);
}

TEST(MetricsRegistryTest, RegisterHistogram) {
  auto& registry = MetricsRegistry::instance();

  auto* histogram =
      registry.register_histogram("test_histogram", "Test histogram");
  ASSERT_NE(histogram, nullptr);

  histogram->observe(10.0);
  auto stats = histogram->get_stats();
  EXPECT_EQ(stats.count, 1);

  auto* same_histogram = registry.get_histogram("test_histogram");
  EXPECT_EQ(same_histogram, histogram);
}

TEST(MetricsRegistryTest, ExportPrometheus) {
  auto& registry = MetricsRegistry::instance();

  auto* counter = registry.register_counter("prom_counter");
  counter->increment(5);

  auto* gauge = registry.register_gauge("prom_gauge");
  gauge->set(42);

  std::string prom = registry.export_prometheus();

  EXPECT_TRUE(prom.find("prom_counter") != std::string::npos);
  EXPECT_TRUE(prom.find("prom_gauge") != std::string::npos);
  EXPECT_TRUE(prom.find("5") != std::string::npos);
  EXPECT_TRUE(prom.find("42") != std::string::npos);
}

TEST(MetricsRegistryTest, ExportJSON) {
  auto& registry = MetricsRegistry::instance();

  auto* counter = registry.register_counter("json_counter");
  counter->increment(10);

  std::string json = registry.export_json();

  EXPECT_TRUE(json.find("json_counter") != std::string::npos);
  EXPECT_TRUE(json.find("10") != std::string::npos);
  EXPECT_TRUE(json.find("counters") != std::string::npos);
}

TEST(MetricsRegistryTest, ResetAll) {
  auto& registry = MetricsRegistry::instance();

  auto* counter = registry.register_counter("reset_counter");
  counter->increment(100);

  auto* histogram = registry.register_histogram("reset_histogram");
  histogram->observe(50.0);

  registry.reset_all();

  EXPECT_EQ(counter->value(), 0);
  auto stats = histogram->get_stats();
  EXPECT_EQ(stats.count, 0);
}

// ==============================================================================
// StandardMetrics Tests
// ==============================================================================

TEST(StandardMetricsTest, Initialize) {
  StandardMetrics::initialize();

  EXPECT_NE(StandardMetrics::requests_total, nullptr);
  EXPECT_NE(StandardMetrics::request_duration_ms, nullptr);
  EXPECT_NE(StandardMetrics::tokens_generated, nullptr);
  EXPECT_NE(StandardMetrics::active_requests, nullptr);
  EXPECT_NE(StandardMetrics::kv_cache_blocks_used, nullptr);
}

TEST(StandardMetricsTest, RequestMetrics) {
  StandardMetrics::initialize();

  StandardMetrics::requests_total->increment();
  EXPECT_EQ(StandardMetrics::requests_total->value(), 1);

  StandardMetrics::request_duration_ms->observe(123.4);
  auto stats = StandardMetrics::request_duration_ms->get_stats();
  EXPECT_EQ(stats.count, 1);
  EXPECT_DOUBLE_EQ(stats.sum, 123.4);
}

TEST(StandardMetricsTest, ActiveRequests) {
  StandardMetrics::initialize();

  StandardMetrics::active_requests->increment();
  EXPECT_EQ(StandardMetrics::active_requests->value(), 1);

  StandardMetrics::active_requests->increment();
  EXPECT_EQ(StandardMetrics::active_requests->value(), 2);

  StandardMetrics::active_requests->decrement();
  EXPECT_EQ(StandardMetrics::active_requests->value(), 1);
}

// ==============================================================================
// RequestTracker Tests
// ==============================================================================

TEST(RequestTrackerTest, Lifecycle) {
  StandardMetrics::initialize();

  int64_t initial_active = StandardMetrics::active_requests->value();
  int64_t initial_total = StandardMetrics::requests_total->value();

  {
    RequestTracker tracker("req-123");
    EXPECT_EQ(tracker.request_id(), "req-123");
    EXPECT_EQ(StandardMetrics::active_requests->value(), initial_active + 1);
    EXPECT_EQ(StandardMetrics::requests_total->value(), initial_total + 1);
  }

  // After tracker destruction, active should decrease
  EXPECT_EQ(StandardMetrics::active_requests->value(), initial_active);
}

TEST(RequestTrackerTest, TokenGeneration) {
  StandardMetrics::initialize();

  int64_t initial_tokens = StandardMetrics::tokens_generated->value();

  {
    RequestTracker tracker("req-456");

    tracker.add_generated_token();
    tracker.add_generated_token();
    tracker.add_generated_token();

    EXPECT_EQ(tracker.tokens_generated(), 3);
  }

  EXPECT_EQ(StandardMetrics::tokens_generated->value(), initial_tokens + 3);
}

TEST(RequestTrackerTest, FirstTokenTiming) {
  StandardMetrics::initialize();

  {
    RequestTracker tracker("req-789");

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tracker.mark_first_token();

    // TTFT should be recorded
    auto stats = StandardMetrics::time_to_first_token_ms->get_stats();
    EXPECT_GT(stats.count, 0);
  }
}

TEST(RequestTrackerTest, SuccessStatus) {
  StandardMetrics::initialize();

  int64_t initial_success = StandardMetrics::requests_success->value();
  int64_t initial_error = StandardMetrics::requests_error->value();

  {
    RequestTracker tracker("req-success");
    tracker.set_status(true);
  }

  EXPECT_EQ(StandardMetrics::requests_success->value(), initial_success + 1);
  EXPECT_EQ(StandardMetrics::requests_error->value(), initial_error);
}

TEST(RequestTrackerTest, ErrorStatus) {
  StandardMetrics::initialize();

  int64_t initial_success = StandardMetrics::requests_success->value();
  int64_t initial_error = StandardMetrics::requests_error->value();

  {
    RequestTracker tracker("req-error");
    tracker.set_status(false);
  }

  EXPECT_EQ(StandardMetrics::requests_success->value(), initial_success);
  EXPECT_EQ(StandardMetrics::requests_error->value(), initial_error + 1);
}

TEST(RequestTrackerTest, DurationTracking) {
  RequestTracker tracker("req-duration");

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  int64_t duration = tracker.duration_ms();
  EXPECT_GE(duration, 10);
  EXPECT_LT(duration, 100);
}

// ==============================================================================
// SystemMonitor Tests
// ==============================================================================

TEST(SystemMonitorTest, Singleton) {
  auto& monitor1 = SystemMonitor::instance();
  auto& monitor2 = SystemMonitor::instance();

  EXPECT_EQ(&monitor1, &monitor2);
}

TEST(SystemMonitorTest, StartStop) {
  auto& monitor = SystemMonitor::instance();

  monitor.start();
  auto stats = monitor.get_stats();
  EXPECT_GE(stats.uptime_seconds, 0);

  monitor.stop();
}

TEST(SystemMonitorTest, UptimeIncreases) {
  auto& monitor = SystemMonitor::instance();
  monitor.start();

  auto stats1 = monitor.get_stats();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  auto stats2 = monitor.get_stats();

  EXPECT_GT(stats2.uptime_seconds, stats1.uptime_seconds);

  monitor.stop();
}

}  // namespace
