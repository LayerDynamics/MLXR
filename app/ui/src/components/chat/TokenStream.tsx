/**
 * TokenStream Component
 *
 * Displays real-time streaming statistics:
 * - Tokens per second
 * - Total tokens generated
 * - Time elapsed
 * - Stop generation button
 */

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { StopCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface TokenStreamProps {
  isStreaming: boolean
  tokensPerSecond: number
  totalTokens?: number
  onStop?: () => void
  className?: string
}

export function TokenStream({
  isStreaming,
  tokensPerSecond,
  totalTokens = 0,
  onStop,
  className,
}: TokenStreamProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  // Track elapsed time while streaming
  useEffect(() => {
    if (!isStreaming) {
      setElapsedSeconds(0)
      return
    }

    const startTime = Date.now()
    const intervalId = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000
      setElapsedSeconds(elapsed)
    }, 100)

    return () => clearInterval(intervalId)
  }, [isStreaming])

  if (!isStreaming) {
    return null
  }

  return (
    <div
      className={cn(
        'flex items-center justify-between rounded-lg border border-primary/20 bg-primary/5 px-4 py-3',
        className
      )}
    >
      <div className="flex items-center gap-6 text-sm">
        {/* Streaming Indicator */}
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
          <span className="font-medium">Generating</span>
        </div>

        {/* Tokens Per Second */}
        <div className="flex items-center gap-2 text-muted-foreground">
          <span className="font-mono font-semibold text-foreground">
            {tokensPerSecond}
          </span>
          <span>tokens/s</span>
        </div>

        {/* Total Tokens */}
        {totalTokens > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <span className="font-mono font-semibold text-foreground">
              {totalTokens}
            </span>
            <span>tokens</span>
          </div>
        )}

        {/* Elapsed Time */}
        <div className="flex items-center gap-2 text-muted-foreground">
          <span className="font-mono font-semibold text-foreground">
            {elapsedSeconds.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Stop Button */}
      {onStop && (
        <Button
          variant="outline"
          size="sm"
          onClick={onStop}
          className="gap-2"
        >
          <StopCircle className="h-4 w-4" />
          Stop
        </Button>
      )}
    </div>
  )
}
