/**
 * useMetrics Hook
 *
 * Real-time metrics polling with TanStack Query
 * - Scheduler stats (requests, tokens/s, latency)
 * - Registry stats (model counts, disk usage)
 * - Metrics snapshot (JSON format for dashboard)
 * - Auto-refresh with configurable intervals
 */

import { useQuery } from '@tanstack/react-query'
import type { SchedulerStats, RegistryStats } from '@/types/backend'
import type { MetricsSnapshot } from '@/types/metrics'
import { backend } from '@/lib/api'
import { queryKeys } from '@/lib/queryClient'

export interface UseMetricsOptions {
  /**
   * Auto-refresh interval in milliseconds
   * Set to false to disable auto-refresh
   * Default: 1000ms (1 second)
   */
  refetchInterval?: number | false

  /**
   * Enable scheduler stats polling
   * Default: true
   */
  enableSchedulerStats?: boolean

  /**
   * Enable registry stats polling
   * Default: false (only fetch on mount)
   */
  enableRegistryStats?: boolean

  /**
   * Enable metrics snapshot polling
   * Default: true
   */
  enableMetricsSnapshot?: boolean

  /**
   * Error handler
   */
  onError?: (error: Error) => void
}

export interface UseMetricsReturn {
  // Scheduler stats
  schedulerStats: SchedulerStats | undefined
  isLoadingSchedulerStats: boolean
  schedulerStatsError: Error | null
  refetchSchedulerStats: () => void

  // Registry stats
  registryStats: RegistryStats | undefined
  isLoadingRegistryStats: boolean
  registryStatsError: Error | null
  refetchRegistryStats: () => void

  // Metrics snapshot
  metricsSnapshot: MetricsSnapshot | undefined
  isLoadingMetricsSnapshot: boolean
  metricsSnapshotError: Error | null
  refetchMetricsSnapshot: () => void

  // Combined loading state
  isLoading: boolean
  hasError: boolean
}

/**
 * Hook for real-time metrics polling
 */
export function useMetrics(options: UseMetricsOptions = {}): UseMetricsReturn {
  const {
    refetchInterval = 1000, // Default: 1 second
    enableSchedulerStats = true,
    enableRegistryStats = false,
    enableMetricsSnapshot = true,
  } = options

  // Query for scheduler stats
  const {
    data: schedulerStats,
    isLoading: isLoadingSchedulerStats,
    error: schedulerStatsError,
    refetch: refetchSchedulerStats,
  } = useQuery<SchedulerStats, Error>({
    queryKey: queryKeys.stats.scheduler(),
    queryFn: backend.getSchedulerStats,
    enabled: enableSchedulerStats,
    refetchInterval: enableSchedulerStats ? refetchInterval : false,
    staleTime: 0, // Always consider stale for real-time data
    gcTime: 30 * 1000, // Keep in cache for 30 seconds
  })

  // Query for registry stats
  const {
    data: registryStats,
    isLoading: isLoadingRegistryStats,
    error: registryStatsError,
    refetch: refetchRegistryStats,
  } = useQuery<RegistryStats, Error>({
    queryKey: queryKeys.stats.registry(),
    queryFn: backend.getRegistryStats,
    enabled: enableRegistryStats,
    refetchInterval: enableRegistryStats ? refetchInterval : false,
    staleTime: 5000, // Registry stats don't change as frequently
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  })

  // Query for metrics snapshot
  const {
    data: metricsSnapshot,
    isLoading: isLoadingMetricsSnapshot,
    error: metricsSnapshotError,
    refetch: refetchMetricsSnapshot,
  } = useQuery<MetricsSnapshot, Error>({
    queryKey: queryKeys.metrics.snapshot(),
    queryFn: backend.getMetricsSnapshot,
    enabled: enableMetricsSnapshot,
    refetchInterval: enableMetricsSnapshot ? refetchInterval : false,
    staleTime: 0, // Always consider stale for real-time data
    gcTime: 30 * 1000, // Keep in cache for 30 seconds
  })

  // Combined loading and error states
  const isLoading =
    (enableSchedulerStats && isLoadingSchedulerStats) ||
    (enableRegistryStats && isLoadingRegistryStats) ||
    (enableMetricsSnapshot && isLoadingMetricsSnapshot)

  const hasError =
    (enableSchedulerStats && !!schedulerStatsError) ||
    (enableRegistryStats && !!registryStatsError) ||
    (enableMetricsSnapshot && !!metricsSnapshotError)

  return {
    // Scheduler stats
    schedulerStats,
    isLoadingSchedulerStats,
    schedulerStatsError: schedulerStatsError as Error | null,
    refetchSchedulerStats,

    // Registry stats
    registryStats,
    isLoadingRegistryStats,
    registryStatsError: registryStatsError as Error | null,
    refetchRegistryStats,

    // Metrics snapshot
    metricsSnapshot,
    isLoadingMetricsSnapshot,
    metricsSnapshotError: metricsSnapshotError as Error | null,
    refetchMetricsSnapshot,

    // Combined states
    isLoading,
    hasError,
  }
}

/**
 * Hook for scheduler stats only
 * Convenience hook for components that only need scheduler metrics
 */
export function useSchedulerStats(refetchInterval: number | false = 1000) {
  const {
    data: schedulerStats,
    isLoading,
    error,
    refetch,
  } = useQuery<SchedulerStats, Error>({
    queryKey: queryKeys.stats.scheduler(),
    queryFn: backend.getSchedulerStats,
    refetchInterval,
    staleTime: 0,
    gcTime: 30 * 1000,
  })

  return {
    schedulerStats,
    isLoading,
    error: error as Error | null,
    refetch,
  }
}

/**
 * Hook for registry stats only
 * Convenience hook for components that only need registry metrics
 */
export function useRegistryStats(refetchInterval: number | false = false) {
  const {
    data: registryStats,
    isLoading,
    error,
    refetch,
  } = useQuery<RegistryStats, Error>({
    queryKey: queryKeys.stats.registry(),
    queryFn: backend.getRegistryStats,
    refetchInterval,
    staleTime: 5000,
    gcTime: 5 * 60 * 1000,
  })

  return {
    registryStats,
    isLoading,
    error: error as Error | null,
    refetch,
  }
}

/**
 * Hook for metrics snapshot only
 * Convenience hook for dashboard components
 */
export function useMetricsSnapshot(refetchInterval: number | false = 1000) {
  const {
    data: metricsSnapshot,
    isLoading,
    error,
    refetch,
  } = useQuery<MetricsSnapshot, Error>({
    queryKey: queryKeys.metrics.snapshot(),
    queryFn: backend.getMetricsSnapshot,
    refetchInterval,
    staleTime: 0,
    gcTime: 30 * 1000,
  })

  return {
    metricsSnapshot,
    isLoading,
    error: error as Error | null,
    refetch,
  }
}

/**
 * Hook for derived metrics calculations
 * Provides computed metrics from raw stats
 */
export function useDerivedMetrics(refetchInterval: number | false = 1000) {
  const { schedulerStats, isLoading, hasError } = useMetrics({
    refetchInterval,
    enableSchedulerStats: true,
    enableRegistryStats: false,
    enableMetricsSnapshot: false,
  })

  // Calculate derived metrics
  const totalRequests =
    (schedulerStats?.waiting_requests || 0) +
    (schedulerStats?.prefilling_requests || 0) +
    (schedulerStats?.decoding_requests || 0) +
    (schedulerStats?.paused_requests || 0)

  const kvUtilization = schedulerStats?.kv_utilization || 0
  const kvUsedBlocks = schedulerStats?.used_kv_blocks || 0
  const kvAvailableBlocks = schedulerStats?.available_kv_blocks || 0
  const kvTotalBlocks = kvUsedBlocks + kvAvailableBlocks

  const tokensPerSecond = schedulerStats?.tokens_per_second || 0
  const requestsPerSecond = schedulerStats?.requests_per_second || 0

  const avgQueueTime = schedulerStats?.avg_queue_time_ms || 0
  const avgPrefillTime = schedulerStats?.avg_prefill_time_ms || 0
  const avgDecodeLatency = schedulerStats?.avg_decode_latency_ms || 0

  const totalTokensGenerated = schedulerStats?.total_tokens_generated || 0
  const totalRequestsCompleted = schedulerStats?.total_requests_completed || 0

  return {
    // Current state
    totalRequests,
    waitingRequests: schedulerStats?.waiting_requests || 0,
    prefillingRequests: schedulerStats?.prefilling_requests || 0,
    decodingRequests: schedulerStats?.decoding_requests || 0,
    pausedRequests: schedulerStats?.paused_requests || 0,

    // KV cache
    kvUtilization,
    kvUsedBlocks,
    kvAvailableBlocks,
    kvTotalBlocks,

    // Throughput
    tokensPerSecond,
    requestsPerSecond,

    // Latency
    avgQueueTime,
    avgPrefillTime,
    avgDecodeLatency,

    // Totals
    totalTokensGenerated,
    totalRequestsCompleted,

    // Loading state
    isLoading,
    hasError,
  }
}
