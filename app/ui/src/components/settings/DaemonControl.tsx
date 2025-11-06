/**
 * DaemonControl Component
 *
 * Control panel for managing the daemon process:
 * - Start/stop/restart buttons
 * - Status indicator
 * - Version info
 * - Resource usage (CPU, memory)
 */

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Square, RotateCw, Loader2, Activity } from 'lucide-react'
import { useDaemon } from '@/hooks/useDaemon'

export interface DaemonControlProps {
  className?: string
}

export function DaemonControl({ className }: DaemonControlProps) {
  const { status, start, stop, restart } = useDaemon()
  const [isLoading, setIsLoading] = useState(false)

  const handleStart = async () => {
    setIsLoading(true)
    try {
      await start()
    } catch (error) {
      console.error('Failed to start daemon:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStop = async () => {
    setIsLoading(true)
    try {
      await stop()
    } catch (error) {
      console.error('Failed to stop daemon:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleRestart = async () => {
    setIsLoading(true)
    try {
      await restart()
    } catch (error) {
      console.error('Failed to restart daemon:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const isRunning = status?.status === 'running'
  const statusVariant = isRunning ? 'default' : 'secondary'
  const statusText = isRunning ? 'Running' : status?.status === 'stopped' ? 'Stopped' : 'Unknown'

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Daemon Status</CardTitle>
            <CardDescription>Manage the MLXR background service</CardDescription>
          </div>
          <Badge variant={statusVariant} className="gap-1.5">
            <Activity className="h-3 w-3" />
            {statusText}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={handleStart}
            disabled={isRunning || isLoading}
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Start
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleStop}
            disabled={!isRunning || isLoading}
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Square className="mr-2 h-4 w-4" />
            )}
            Stop
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleRestart}
            disabled={!isRunning || isLoading}
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RotateCw className="mr-2 h-4 w-4" />
            )}
            Restart
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
