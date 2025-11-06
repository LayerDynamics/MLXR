/**
 * ModelStats Component
 *
 * Display usage statistics for a model:
 * - Total requests
 * - Token counts (prompt/completion)
 * - Average latency
 * - Last used timestamp
 * - Cache hit rate
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Clock,
  Zap,
  MessageSquare,
  Activity,
  Database,
  TrendingUp,
} from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ModelStatsData {
  total_requests: number
  total_prompt_tokens: number
  total_completion_tokens: number
  avg_latency_ms: number
  avg_tokens_per_second: number
  last_used_at?: number
  cache_hit_rate?: number
  uptime_hours?: number
}

export interface ModelStatsProps {
  modelId: string
  stats: ModelStatsData
  className?: string
}

export function ModelStats({ stats, className }: ModelStatsProps) {
  const formatNumber = (num: number): string => {
    if (num >= 1_000_000) {
      return `${(num / 1_000_000).toFixed(1)}M`
    } else if (num >= 1_000) {
      return `${(num / 1_000).toFixed(1)}K`
    }
    return num.toString()
  }

  const formatTimestamp = (timestamp?: number): string => {
    if (!timestamp) return 'Never'

    const date = new Date(timestamp * 1000)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const hours = Math.floor(diff / (1000 * 60 * 60))

    if (hours < 1) {
      const minutes = Math.floor(diff / (1000 * 60))
      return minutes < 1 ? 'Just now' : `${minutes}m ago`
    } else if (hours < 24) {
      return `${hours}h ago`
    } else {
      const days = Math.floor(hours / 24)
      return `${days}d ago`
    }
  }

  const formatUptime = (hours?: number): string => {
    if (!hours) return 'N/A'
    if (hours < 1) return `${Math.floor(hours * 60)}m`
    if (hours < 24) return `${Math.floor(hours)}h`
    return `${Math.floor(hours / 24)}d`
  }

  const statItems = [
    {
      icon: MessageSquare,
      label: 'Total Requests',
      value: formatNumber(stats.total_requests),
      color: 'text-blue-500',
    },
    {
      icon: Zap,
      label: 'Tokens/Second',
      value: stats.avg_tokens_per_second.toFixed(1),
      color: 'text-yellow-500',
    },
    {
      icon: Clock,
      label: 'Avg Latency',
      value: `${stats.avg_latency_ms.toFixed(0)}ms`,
      color: 'text-purple-500',
    },
    {
      icon: Activity,
      label: 'Prompt Tokens',
      value: formatNumber(stats.total_prompt_tokens),
      color: 'text-green-500',
    },
    {
      icon: TrendingUp,
      label: 'Completion Tokens',
      value: formatNumber(stats.total_completion_tokens),
      color: 'text-orange-500',
    },
    {
      icon: Database,
      label: 'Cache Hit Rate',
      value: stats.cache_hit_rate ? `${(stats.cache_hit_rate * 100).toFixed(1)}%` : 'N/A',
      color: 'text-cyan-500',
    },
  ]

  return (
    <Card className={cn('', className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">
            Usage Statistics
          </CardTitle>
          {stats.last_used_at && (
            <Badge variant="outline" className="text-xs">
              Last used {formatTimestamp(stats.last_used_at)}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {statItems.map((item, index) => (
            <div key={index} className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <item.icon className={cn('h-3.5 w-3.5', item.color)} />
                {item.label}
              </div>
              <div className="text-lg font-semibold">{item.value}</div>
            </div>
          ))}
        </div>

        {stats.uptime_hours !== undefined && (
          <div className="mt-4 pt-4 border-t">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Uptime</span>
              <span className="font-medium">{formatUptime(stats.uptime_hours)}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
