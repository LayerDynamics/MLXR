/**
 * TrayPopover Component
 *
 * Quick status view for system tray with:
 * - Quick status view
 * - Current model
 * - Tokens/s, latency
 * - Context used (%)
 * - Quick actions: start/stop daemon, switch model
 * - Open main window button
 */

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useMetrics } from '@/hooks/useMetrics'
import { useDaemon } from '@/hooks/useDaemon'
import { useAppStore } from '@/lib/store'
import { Activity, ExternalLink, Power, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface TrayPopoverProps {
  onOpenMainWindow?: () => void
  className?: string
}

export function TrayPopover({ onOpenMainWindow, className }: TrayPopoverProps) {
  const { metricsSnapshot } = useMetrics()
  const { status, start, stop, restart } = useDaemon()
  const { activeModelId } = useAppStore()

  const isRunning = status?.status === 'running'
  const tokensPerSec = metricsSnapshot?.tokens_per_second?.mean?.toFixed(1) || '0'
  const latency = metricsSnapshot?.time_to_first_token_ms?.p95?.toFixed(0) || '0'
  const kvUsage = metricsSnapshot
    ? ((metricsSnapshot.kv_cache_blocks_used / metricsSnapshot.kv_cache_blocks_total) * 100).toFixed(0)
    : '0'

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className={cn('h-8 w-8', className)}>
          <Activity className={cn('h-4 w-4', isRunning ? 'text-green-500' : 'text-muted-foreground')} />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-4">
          {/* Status Header */}
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">MLXR Status</h3>
            <Badge variant={isRunning ? 'default' : 'secondary'}>
              {isRunning ? 'Running' : 'Stopped'}
            </Badge>
          </div>

          <Separator />

          {/* Current Model */}
          <div className="space-y-1">
            <div className="text-xs text-muted-foreground">Current Model</div>
            <div className="text-sm font-medium">
              {activeModelId || 'No model selected'}
            </div>
          </div>

          {/* Metrics */}
          {isRunning && metricsSnapshot && (
            <>
              <Separator />
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">{tokensPerSec}</div>
                  <div className="text-xs text-muted-foreground">tok/s</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{latency}</div>
                  <div className="text-xs text-muted-foreground">ms p95</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{kvUsage}%</div>
                  <div className="text-xs text-muted-foreground">KV used</div>
                </div>
              </div>
            </>
          )}

          <Separator />

          {/* Quick Actions */}
          <div className="space-y-2">
            {isRunning ? (
              <>
                <Button variant="outline" size="sm" className="w-full" onClick={restart}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Restart Daemon
                </Button>
                <Button variant="outline" size="sm" className="w-full" onClick={stop}>
                  <Power className="h-4 w-4 mr-2" />
                  Stop Daemon
                </Button>
              </>
            ) : (
              <Button variant="outline" size="sm" className="w-full" onClick={start}>
                <Power className="h-4 w-4 mr-2" />
                Start Daemon
              </Button>
            )}

            {onOpenMainWindow && (
              <Button size="sm" className="w-full" onClick={onOpenMainWindow}>
                <ExternalLink className="h-4 w-4 mr-2" />
                Open Main Window
              </Button>
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
