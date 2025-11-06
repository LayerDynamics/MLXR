/**
 * Metrics and telemetry types
 * Matches daemon/telemetry/metrics.{h,cpp}
 */

export interface Counter {
  name: string
  value: number
  labels?: Record<string, string>
}

export interface Gauge {
  name: string
  value: number
  labels?: Record<string, string>
}

export interface HistogramStats {
  count: number
  sum: number
  min: number
  max: number
  mean: number
  p50: number
  p95: number
  p99: number
}

export interface Histogram {
  name: string
  stats: HistogramStats
  labels?: Record<string, string>
}

export interface MetricsSnapshot {
  timestamp: number

  // Request metrics
  requests_total: number
  requests_success: number
  requests_error: number
  request_duration_ms: HistogramStats

  // Token metrics
  tokens_generated: number
  tokens_per_second: HistogramStats
  time_to_first_token_ms: HistogramStats

  // Model metrics
  active_requests: number
  models_loaded: number
  memory_used_bytes: number
  gpu_memory_used_bytes: number

  // KV cache metrics
  kv_cache_blocks_used: number
  kv_cache_blocks_total: number
  kv_cache_evictions: number
  kv_cache_hit_rate: HistogramStats

  // Scheduler metrics
  prefill_queue_size: number
  decode_queue_size: number
  batch_size: HistogramStats
  scheduler_latency_ms: HistogramStats

  // Speculative decoding
  speculative_tokens_proposed: number
  speculative_tokens_accepted: number
  speculative_acceptance_rate: HistogramStats

  // System metrics
  cpu_usage_percent: number
  gpu_usage_percent: number
  uptime_seconds: number
}

export interface PrometheusMetric {
  name: string
  type: 'counter' | 'gauge' | 'histogram'
  help: string
  value: number | HistogramStats
  labels?: Record<string, string>
}

export interface PrometheusFormat {
  metrics: PrometheusMetric[]
  timestamp: number
}

// Time series data for charts
export interface TimeSeriesPoint {
  timestamp: number
  value: number
}

export interface TimeSeries {
  name: string
  data: TimeSeriesPoint[]
  unit?: string
}

// Latency distribution for histograms
export interface LatencyBucket {
  le: number // Less than or equal to (upper bound in ms)
  count: number
  percentage: number
}

export interface LatencyDistribution {
  buckets: LatencyBucket[]
  total_count: number
  p50: number
  p95: number
  p99: number
  mean: number
}

// KV cache heatmap data
export interface KVBlockInfo {
  block_id: number
  request_id: string | null
  sequence_id: number
  token_count: number
  last_access_time: number
  location: 'gpu' | 'cpu'
}

export interface KVCacheHeatmap {
  total_blocks: number
  blocks: KVBlockInfo[]
  gpu_utilization: number
  cpu_utilization: number
}

// Kernel timing breakdown
export interface KernelTiming {
  name: string
  gpu_time_ms: number
  cpu_time_ms: number
  call_count: number
  avg_time_ms: number
}

export interface KernelTimings {
  kernels: KernelTiming[]
  total_gpu_time_ms: number
  total_cpu_time_ms: number
}

// Request tracking data
export interface RequestMetrics {
  request_id: string
  model: string
  prompt_tokens: number
  generated_tokens: number
  first_token_latency_ms: number
  tokens_per_second: number
  total_duration_ms: number
  created_at: number
  completed_at: number
  status: 'success' | 'error' | 'cancelled'
}

// Aggregated statistics
export interface ModelStatistics {
  model_id: string
  total_requests: number
  total_tokens_generated: number
  avg_tokens_per_second: number
  avg_latency_ms: number
  error_rate: number
  last_used: number
}

// Export format options
export type MetricsFormat = 'json' | 'prometheus'

export interface MetricsExportOptions {
  format: MetricsFormat
  time_range?: {
    start: number
    end: number
  }
  include_labels?: boolean
  include_histograms?: boolean
}
