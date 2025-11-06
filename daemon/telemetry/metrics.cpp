// Copyright © 2025 MLXR Development
// Metrics collection and reporting implementation

#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>

using json = nlohmann::json;

namespace mlxr {
namespace telemetry {

// ==============================================================================
// Histogram Implementation
// ==============================================================================

Histogram::Histogram()
    : count_(0),
      sum_(0.0),
      min_(std::numeric_limits<double>::max()),
      max_(std::numeric_limits<double>::min()) {
  values_.reserve(10000);  // Pre-allocate
}

void Histogram::observe(double value) {
  std::lock_guard<std::mutex> lock(mutex_);
  values_.push_back(value);
  count_++;
  sum_ += value;
  min_ = std::min(min_, value);
  max_ = std::max(max_, value);
}

Histogram::Stats Histogram::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  Stats stats;
  stats.count = count_;
  stats.sum = sum_;
  stats.min = (count_ > 0) ? min_ : 0.0;
  stats.max = (count_ > 0) ? max_ : 0.0;
  stats.mean = (count_ > 0) ? sum_ / count_ : 0.0;

  if (count_ > 0 && !values_.empty()) {
    // Calculate percentiles
    std::vector<double> sorted = values_;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) -> double {
      size_t idx = static_cast<size_t>(std::ceil(p * sorted.size())) - 1;
      idx = std::min(idx, sorted.size() - 1);
      return sorted[idx];
    };

    stats.p50 = percentile(0.50);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);
  } else {
    stats.p50 = 0.0;
    stats.p95 = 0.0;
    stats.p99 = 0.0;
  }

  return stats;
}

void Histogram::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  values_.clear();
  count_ = 0;
  sum_ = 0.0;
  min_ = std::numeric_limits<double>::max();
  max_ = std::numeric_limits<double>::min();
}

// ==============================================================================
// MetricsRegistry Implementation
// ==============================================================================

MetricsRegistry& MetricsRegistry::instance() {
  static MetricsRegistry instance;
  return instance;
}

Counter* MetricsRegistry::register_counter(const std::string& name,
                                           const std::string& description) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = counters_.find(name);
  if (it != counters_.end()) {
    return it->second.get();
  }

  auto counter = std::make_unique<Counter>();
  auto* ptr = counter.get();
  counters_[name] = std::move(counter);
  if (!description.empty()) {
    descriptions_[name] = description;
  }
  return ptr;
}

Gauge* MetricsRegistry::register_gauge(const std::string& name,
                                       const std::string& description) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = gauges_.find(name);
  if (it != gauges_.end()) {
    return it->second.get();
  }

  auto gauge = std::make_unique<Gauge>();
  auto* ptr = gauge.get();
  gauges_[name] = std::move(gauge);
  if (!description.empty()) {
    descriptions_[name] = description;
  }
  return ptr;
}

Histogram* MetricsRegistry::register_histogram(const std::string& name,
                                               const std::string& description) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = histograms_.find(name);
  if (it != histograms_.end()) {
    return it->second.get();
  }

  auto histogram = std::make_unique<Histogram>();
  auto* ptr = histogram.get();
  histograms_[name] = std::move(histogram);
  if (!description.empty()) {
    descriptions_[name] = description;
  }
  return ptr;
}

Counter* MetricsRegistry::get_counter(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = counters_.find(name);
  return (it != counters_.end()) ? it->second.get() : nullptr;
}

Gauge* MetricsRegistry::get_gauge(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = gauges_.find(name);
  return (it != gauges_.end()) ? it->second.get() : nullptr;
}

Histogram* MetricsRegistry::get_histogram(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = histograms_.find(name);
  return (it != histograms_.end()) ? it->second.get() : nullptr;
}

std::string MetricsRegistry::export_prometheus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream oss;

  // Export counters
  for (const auto& [name, counter] : counters_) {
    auto desc_it = descriptions_.find(name);
    if (desc_it != descriptions_.end()) {
      oss << "# HELP " << name << " " << desc_it->second << "\n";
    }
    oss << "# TYPE " << name << " counter\n";
    oss << name << " " << counter->value() << "\n\n";
  }

  // Export gauges
  for (const auto& [name, gauge] : gauges_) {
    auto desc_it = descriptions_.find(name);
    if (desc_it != descriptions_.end()) {
      oss << "# HELP " << name << " " << desc_it->second << "\n";
    }
    oss << "# TYPE " << name << " gauge\n";
    oss << name << " " << gauge->value() << "\n\n";
  }

  // Export histograms
  for (const auto& [name, histogram] : histograms_) {
    auto stats = histogram->get_stats();
    auto desc_it = descriptions_.find(name);
    if (desc_it != descriptions_.end()) {
      oss << "# HELP " << name << " " << desc_it->second << "\n";
    }
    oss << "# TYPE " << name << " summary\n";
    oss << name << "_count " << stats.count << "\n";
    oss << name << "_sum " << stats.sum << "\n";
    oss << name << "{quantile=\"0.5\"} " << stats.p50 << "\n";
    oss << name << "{quantile=\"0.95\"} " << stats.p95 << "\n";
    oss << name << "{quantile=\"0.99\"} " << stats.p99 << "\n\n";
  }

  return oss.str();
}

std::string MetricsRegistry::export_json() const {
  std::lock_guard<std::mutex> lock(mutex_);
  json j;

  // Export counters
  json counters_json = json::object();
  for (const auto& [name, counter] : counters_) {
    counters_json[name] = counter->value();
  }
  j["counters"] = counters_json;

  // Export gauges
  json gauges_json = json::object();
  for (const auto& [name, gauge] : gauges_) {
    gauges_json[name] = gauge->value();
  }
  j["gauges"] = gauges_json;

  // Export histograms
  json histograms_json = json::object();
  for (const auto& [name, histogram] : histograms_) {
    auto stats = histogram->get_stats();
    histograms_json[name] = {{"count", stats.count}, {"sum", stats.sum},
                             {"min", stats.min},     {"max", stats.max},
                             {"mean", stats.mean},   {"p50", stats.p50},
                             {"p95", stats.p95},     {"p99", stats.p99}};
  }
  j["histograms"] = histograms_json;

  return j.dump(2);
}

void MetricsRegistry::reset_all() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& [name, counter] : counters_) {
    counter->reset();
  }

  for (auto& [name, histogram] : histograms_) {
    histogram->reset();
  }

  // Gauges are not reset as they represent current state
}

// ==============================================================================
// StandardMetrics Implementation
// ==============================================================================

Counter* StandardMetrics::requests_total = nullptr;
Counter* StandardMetrics::requests_success = nullptr;
Counter* StandardMetrics::requests_error = nullptr;
Histogram* StandardMetrics::request_duration_ms = nullptr;

Counter* StandardMetrics::tokens_generated = nullptr;
Histogram* StandardMetrics::tokens_per_second = nullptr;
Histogram* StandardMetrics::time_to_first_token_ms = nullptr;

Gauge* StandardMetrics::active_requests = nullptr;
Gauge* StandardMetrics::models_loaded = nullptr;
Gauge* StandardMetrics::memory_used_bytes = nullptr;
Gauge* StandardMetrics::gpu_memory_used_bytes = nullptr;

Gauge* StandardMetrics::kv_cache_blocks_used = nullptr;
Gauge* StandardMetrics::kv_cache_blocks_total = nullptr;
Counter* StandardMetrics::kv_cache_evictions = nullptr;
Histogram* StandardMetrics::kv_cache_hit_rate = nullptr;

Gauge* StandardMetrics::prefill_queue_size = nullptr;
Gauge* StandardMetrics::decode_queue_size = nullptr;
Histogram* StandardMetrics::batch_size = nullptr;
Histogram* StandardMetrics::scheduler_latency_ms = nullptr;

Counter* StandardMetrics::speculative_tokens_proposed = nullptr;
Counter* StandardMetrics::speculative_tokens_accepted = nullptr;
Histogram* StandardMetrics::speculative_acceptance_rate = nullptr;

Gauge* StandardMetrics::cpu_usage_percent = nullptr;
Gauge* StandardMetrics::gpu_usage_percent = nullptr;
Gauge* StandardMetrics::uptime_seconds = nullptr;

void StandardMetrics::initialize() {
  auto& registry = MetricsRegistry::instance();

  // Request metrics
  requests_total = registry.register_counter(
      "mlxr_requests_total", "Total number of requests received");
  requests_success = registry.register_counter("mlxr_requests_success",
                                               "Number of successful requests");
  requests_error = registry.register_counter("mlxr_requests_error",
                                             "Number of failed requests");
  request_duration_ms = registry.register_histogram(
      "mlxr_request_duration_ms", "Request duration in milliseconds");

  // Token metrics
  tokens_generated = registry.register_counter(
      "mlxr_tokens_generated_total", "Total number of tokens generated");
  tokens_per_second = registry.register_histogram("mlxr_tokens_per_second",
                                                  "Token generation rate");
  time_to_first_token_ms = registry.register_histogram(
      "mlxr_time_to_first_token_ms", "Time to first token in milliseconds");

  // Model metrics
  active_requests = registry.register_gauge("mlxr_active_requests",
                                            "Number of active requests");
  models_loaded = registry.register_gauge("mlxr_models_loaded",
                                          "Number of models currently loaded");
  memory_used_bytes =
      registry.register_gauge("mlxr_memory_used_bytes", "Memory used in bytes");
  gpu_memory_used_bytes = registry.register_gauge("mlxr_gpu_memory_used_bytes",
                                                  "GPU memory used in bytes");

  // KV cache metrics
  kv_cache_blocks_used = registry.register_gauge(
      "mlxr_kv_cache_blocks_used", "Number of KV cache blocks in use");
  kv_cache_blocks_total = registry.register_gauge(
      "mlxr_kv_cache_blocks_total", "Total number of KV cache blocks");
  kv_cache_evictions = registry.register_counter(
      "mlxr_kv_cache_evictions_total", "Number of KV cache evictions");
  kv_cache_hit_rate = registry.register_histogram("mlxr_kv_cache_hit_rate",
                                                  "KV cache hit rate");

  // Scheduler metrics
  prefill_queue_size = registry.register_gauge(
      "mlxr_prefill_queue_size", "Number of requests in prefill queue");
  decode_queue_size = registry.register_gauge(
      "mlxr_decode_queue_size", "Number of requests in decode queue");
  batch_size = registry.register_histogram("mlxr_batch_size", "Batch size");
  scheduler_latency_ms = registry.register_histogram(
      "mlxr_scheduler_latency_ms", "Scheduler latency in milliseconds");

  // Speculative decoding metrics
  speculative_tokens_proposed =
      registry.register_counter("mlxr_speculative_tokens_proposed_total",
                                "Number of speculative tokens proposed");
  speculative_tokens_accepted =
      registry.register_counter("mlxr_speculative_tokens_accepted_total",
                                "Number of speculative tokens accepted");
  speculative_acceptance_rate = registry.register_histogram(
      "mlxr_speculative_acceptance_rate", "Speculative token acceptance rate");

  // System metrics
  cpu_usage_percent =
      registry.register_gauge("mlxr_cpu_usage_percent", "CPU usage percentage");
  gpu_usage_percent =
      registry.register_gauge("mlxr_gpu_usage_percent", "GPU usage percentage");
  uptime_seconds =
      registry.register_gauge("mlxr_uptime_seconds", "Uptime in seconds");
}

// ==============================================================================
// RequestTracker Implementation
// ==============================================================================

RequestTracker::RequestTracker(const std::string& request_id)
    : request_id_(request_id),
      start_time_(std::chrono::steady_clock::now()),
      prompt_tokens_(0),
      tokens_generated_(0),
      first_token_marked_(false),
      success_(false) {
  if (StandardMetrics::active_requests) {
    StandardMetrics::active_requests->increment();
  }
  if (StandardMetrics::requests_total) {
    StandardMetrics::requests_total->increment();
  }
}

RequestTracker::~RequestTracker() {
  if (StandardMetrics::active_requests) {
    StandardMetrics::active_requests->decrement();
  }

  // Record final metrics
  auto duration = duration_ms();
  if (StandardMetrics::request_duration_ms) {
    StandardMetrics::request_duration_ms->observe(
        static_cast<double>(duration));
  }

  if (success_) {
    if (StandardMetrics::requests_success) {
      StandardMetrics::requests_success->increment();
    }

    // Calculate tokens per second
    if (tokens_generated_ > 0 && duration > 0) {
      double tps = (tokens_generated_ * 1000.0) / duration;
      if (StandardMetrics::tokens_per_second) {
        StandardMetrics::tokens_per_second->observe(tps);
      }
    }
  } else {
    if (StandardMetrics::requests_error) {
      StandardMetrics::requests_error->increment();
    }
  }
}

void RequestTracker::set_model(const std::string& model) { model_ = model; }

void RequestTracker::set_prompt_tokens(int count) { prompt_tokens_ = count; }

void RequestTracker::mark_first_token() {
  if (!first_token_marked_) {
    first_token_marked_ = true;
    first_token_time_ = std::chrono::steady_clock::now();

    auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(
                    first_token_time_ - start_time_)
                    .count();

    if (StandardMetrics::time_to_first_token_ms) {
      StandardMetrics::time_to_first_token_ms->observe(
          static_cast<double>(ttft));
    }
  }
}

void RequestTracker::add_generated_token() {
  tokens_generated_++;
  if (StandardMetrics::tokens_generated) {
    StandardMetrics::tokens_generated->increment();
  }

  // Mark first token if not already done
  if (!first_token_marked_) {
    mark_first_token();
  }
}

void RequestTracker::set_status(bool success) { success_ = success; }

int64_t RequestTracker::duration_ms() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                               start_time_)
      .count();
}

// ==============================================================================
// SystemMonitor Implementation
// ==============================================================================

SystemMonitor& SystemMonitor::instance() {
  static SystemMonitor instance;
  return instance;
}

SystemMonitor::SystemMonitor()
    : running_(false), start_time_(std::chrono::steady_clock::now()) {
  current_stats_ = {};
}

SystemMonitor::~SystemMonitor() { stop(); }

void SystemMonitor::start() {
  if (running_.exchange(true)) {
    return;  // Already running
  }

  // TODO: Start monitoring thread
  // For now, just initialize to zero
  std::lock_guard<std::mutex> lock(mutex_);
  current_stats_ = {};
}

void SystemMonitor::stop() {
  running_.store(false);
  // TODO: Stop monitoring thread
}

SystemMonitor::SystemStats SystemMonitor::get_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Update uptime
  auto now = std::chrono::steady_clock::now();
  auto uptime =
      std::chrono::duration_cast<std::chrono::seconds>(now - start_time_)
          .count();

  SystemStats stats = current_stats_;
  stats.uptime_seconds = uptime;

  return stats;
}

void SystemMonitor::monitor_loop() {
  // TODO: Implement system monitoring
  // This would periodically sample CPU/GPU usage and memory
  // For now, placeholder implementation
}

}  // namespace telemetry
}  // namespace mlxr
