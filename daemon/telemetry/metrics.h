// Copyright © 2025 MLXR Development
// Metrics collection and reporting system

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mlxr {
namespace telemetry {

// ==============================================================================
// Metric Types
// ==============================================================================

// Counter: monotonically increasing value
class Counter {
 public:
  Counter() : value_(0) {}

  void increment(int64_t delta = 1) {
    value_.fetch_add(delta, std::memory_order_relaxed);
  }

  int64_t value() const { return value_.load(std::memory_order_relaxed); }

  void reset() { value_.store(0, std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> value_;
};

// Gauge: value that can go up or down
class Gauge {
 public:
  Gauge() : value_(0) {}

  void set(int64_t value) { value_.store(value, std::memory_order_relaxed); }

  void increment(int64_t delta = 1) {
    value_.fetch_add(delta, std::memory_order_relaxed);
  }

  void decrement(int64_t delta = 1) {
    value_.fetch_sub(delta, std::memory_order_relaxed);
  }

  int64_t value() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> value_;
};

// Histogram: tracks distribution of values
class Histogram {
 public:
  Histogram();

  void observe(double value);

  struct Stats {
    int64_t count;
    double sum;
    double min;
    double max;
    double mean;
    double p50;
    double p95;
    double p99;
  };

  Stats get_stats() const;
  void reset();

 private:
  mutable std::mutex mutex_;
  std::vector<double> values_;
  int64_t count_;
  double sum_;
  double min_;
  double max_;
};

// Timer: measures duration of operations
class Timer {
 public:
  explicit Timer(Histogram& histogram)
      : histogram_(histogram), start_(std::chrono::steady_clock::now()) {}

  ~Timer() {
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count();
    histogram_.observe(static_cast<double>(duration) /
                       1000.0);  // Convert to ms
  }

 private:
  Histogram& histogram_;
  std::chrono::steady_clock::time_point start_;
};

// ==============================================================================
// Metrics Registry
// ==============================================================================

class MetricsRegistry {
 public:
  static MetricsRegistry& instance();

  // Delete copy operations
  MetricsRegistry(const MetricsRegistry&) = delete;
  MetricsRegistry& operator=(const MetricsRegistry&) = delete;

  // Register metrics
  Counter* register_counter(const std::string& name,
                            const std::string& description = "");
  Gauge* register_gauge(const std::string& name,
                        const std::string& description = "");
  Histogram* register_histogram(const std::string& name,
                                const std::string& description = "");

  // Get metrics
  Counter* get_counter(const std::string& name);
  Gauge* get_gauge(const std::string& name);
  Histogram* get_histogram(const std::string& name);

  // Export all metrics
  std::string export_prometheus() const;
  std::string export_json() const;

  // Reset all metrics
  void reset_all();

 private:
  MetricsRegistry() = default;

  mutable std::mutex mutex_;
  std::map<std::string, std::unique_ptr<Counter>> counters_;
  std::map<std::string, std::unique_ptr<Gauge>> gauges_;
  std::map<std::string, std::unique_ptr<Histogram>> histograms_;
  std::map<std::string, std::string> descriptions_;
};

// ==============================================================================
// Standard Metrics
// ==============================================================================

class StandardMetrics {
 public:
  static void initialize();

  // Request metrics
  static Counter* requests_total;
  static Counter* requests_success;
  static Counter* requests_error;
  static Histogram* request_duration_ms;

  // Token metrics
  static Counter* tokens_generated;
  static Histogram* tokens_per_second;
  static Histogram* time_to_first_token_ms;

  // Model metrics
  static Gauge* active_requests;
  static Gauge* models_loaded;
  static Gauge* memory_used_bytes;
  static Gauge* gpu_memory_used_bytes;

  // KV cache metrics
  static Gauge* kv_cache_blocks_used;
  static Gauge* kv_cache_blocks_total;
  static Counter* kv_cache_evictions;
  static Histogram* kv_cache_hit_rate;

  // Scheduler metrics
  static Gauge* prefill_queue_size;
  static Gauge* decode_queue_size;
  static Histogram* batch_size;
  static Histogram* scheduler_latency_ms;

  // Speculative decoding metrics
  static Counter* speculative_tokens_proposed;
  static Counter* speculative_tokens_accepted;
  static Histogram* speculative_acceptance_rate;

  // System metrics
  static Gauge* cpu_usage_percent;
  static Gauge* gpu_usage_percent;
  static Gauge* uptime_seconds;
};

// ==============================================================================
// Request Tracker
// ==============================================================================

class RequestTracker {
 public:
  explicit RequestTracker(const std::string& request_id);
  ~RequestTracker();

  void set_model(const std::string& model);
  void set_prompt_tokens(int count);
  void mark_first_token();
  void add_generated_token();
  void set_status(bool success);

  std::string request_id() const { return request_id_; }
  int64_t duration_ms() const;
  int tokens_generated() const { return tokens_generated_; }

 private:
  std::string request_id_;
  std::string model_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point first_token_time_;
  int prompt_tokens_;
  int tokens_generated_;
  bool first_token_marked_;
  bool success_;
};

// ==============================================================================
// System Monitor
// ==============================================================================

class SystemMonitor {
 public:
  static SystemMonitor& instance();

  // Delete copy operations
  SystemMonitor(const SystemMonitor&) = delete;
  SystemMonitor& operator=(const SystemMonitor&) = delete;

  // Start/stop monitoring
  void start();
  void stop();

  // Get current stats
  struct SystemStats {
    double cpu_usage_percent;
    double gpu_usage_percent;
    int64_t memory_used_bytes;
    int64_t gpu_memory_used_bytes;
    int64_t uptime_seconds;
  };

  SystemStats get_stats() const;

 private:
  SystemMonitor();
  ~SystemMonitor();

  void monitor_loop();

  std::atomic<bool> running_;
  mutable std::mutex mutex_;
  SystemStats current_stats_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace telemetry
}  // namespace mlxr
