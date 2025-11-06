/**
 * StatusBar Component
 *
 * Bottom status bar showing:
 * - Daemon status indicator
 * - Current model
 * - Tokens/second (live)
 * - P95 latency (live)
 * - KV usage percentage
 * - Connection status
 */

import { useEffect, useState } from 'react'
import { Activity, Database, Wifi, WifiOff, Circle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useDaemon } from '@/hooks/useDaemon'
import { useMetrics } from '@/hooks/useMetrics'
import { useAppStore } from '@/lib/store'

export function StatusBar() {
  const { status: daemonStatus } = useDaemon()
  const { metricsSnapshot } = useMetrics()
  const activeModel = useAppStore((state) => state.activeModelId)
  const [isOnline, setIsOnline] = useState(navigator.onLine)

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  const daemonRunning = daemonStatus?.status === 'running'
  const tokensPerSecond = metricsSnapshot?.tokens_per_second?.mean ?? 0
  const p95Latency = metricsSnapshot?.request_duration_ms?.p95 ?? 0
  const kvUsage = metricsSnapshot
    ? (metricsSnapshot.kv_cache_blocks_used / metricsSnapshot.kv_cache_blocks_total) * 100
    : 0

  return (
    <footer className="flex h-8 items-center justify-between border-t border-border bg-card px-4 text-xs">
      {/* Left Section: Daemon Status */}
      <div className="flex items-center gap-4">
        {/* Daemon Status */}
        <div className="flex items-center gap-2">
          <Circle
            className={cn(
              'h-2 w-2 fill-current',
              daemonRunning ? 'text-green-500' : 'text-red-500'
            )}
          />
          <span className="text-muted-foreground">
            {daemonRunning ? 'Daemon Running' : 'Daemon Stopped'}
          </span>
        </div>

        {/* Current Model */}
        {activeModel && (
          <div className="flex items-center gap-2">
            <Database className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">{activeModel}</span>
          </div>
        )}
      </div>

      {/* Right Section: Metrics and Connection */}
      <div className="flex items-center gap-4">
        {/* Tokens/Second */}
        {daemonRunning && tokensPerSecond > 0 && (
          <div className="flex items-center gap-2">
            <Activity className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">
              {tokensPerSecond.toFixed(1)} tok/s
            </span>
          </div>
        )}

        {/* P95 Latency */}
        {daemonRunning && p95Latency > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">
              P95: {p95Latency.toFixed(0)}ms
            </span>
          </div>
        )}

        {/* KV Cache Usage */}
        {daemonRunning && kvUsage > 0 && (
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'text-muted-foreground',
                kvUsage > 90 && 'text-yellow-500',
                kvUsage > 95 && 'text-red-500'
              )}
            >
              KV: {kvUsage.toFixed(0)}%
            </span>
          </div>
        )}

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          {isOnline ? (
            <>
              <Wifi className="h-3 w-3 text-green-500" />
              <span className="text-muted-foreground">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-3 w-3 text-red-500" />
              <span className="text-muted-foreground">Offline</span>
            </>
          )}
        </div>
      </div>
    </footer>
  )
}
