/**
 * LiveMetrics Component
 *
 * Real-time metrics dashboard showing:
 * - Current tokens/second
 * - Active requests
 * - GPU/CPU usage
 * - KV cache utilization
 * - Auto-refreshing display
 */

import { useMetrics } from '@/hooks/useMetrics'
import { StatsCard } from './StatsCard'
import { Zap, MessageSquare, Database, Cpu } from 'lucide-react'

export interface LiveMetricsProps {
  className?: string
}

export function LiveMetrics({ className }: LiveMetricsProps) {
  const { metricsSnapshot, isLoading } = useMetrics()

  if (isLoading || !metricsSnapshot) {
    return (
      <div className={className}>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-[140px] animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      </div>
    )
  }

  const stats = [
    {
      label: 'Tokens/Second',
      value: metricsSnapshot.tokens_per_second?.mean?.toFixed(1) || '0',
      icon: Zap,
      trend: 'up' as const,
      trendValue: `p95: ${metricsSnapshot.tokens_per_second?.p95?.toFixed(1) || '0'}`,
    },
    {
      label: 'Active Requests',
      value: metricsSnapshot.active_requests || 0,
      icon: MessageSquare,
      trend: 'neutral' as const,
    },
    {
      label: 'KV Cache Used',
      value: `${((metricsSnapshot.kv_cache_blocks_used || 0) / (metricsSnapshot.kv_cache_blocks_total || 1) * 100).toFixed(0)}%`,
      icon: Database,
      trend: metricsSnapshot.kv_cache_blocks_used > metricsSnapshot.kv_cache_blocks_total * 0.8 ? 'up' as const : 'neutral' as const,
    },
    {
      label: 'GPU Memory',
      value: `${((metricsSnapshot.gpu_memory_used_bytes || 0) / 1024 / 1024 / 1024).toFixed(1)} GB`,
      icon: Cpu,
      trendValue: `${((metricsSnapshot.memory_used_bytes || 0) / 1024 / 1024 / 1024).toFixed(1)} GB total`,
    },
  ]

  return (
    <div className={className}>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </div>
    </div>
  )
}
