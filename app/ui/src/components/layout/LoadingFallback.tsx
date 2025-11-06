/**
 * LoadingFallback Component
 *
 * Full-page loading indicator with:
 * - Loading spinner
 * - Loading message
 * - Progress indicator (if available)
 * - Cancel button (optional)
 */

import { Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

interface LoadingFallbackProps {
  message?: string
  progress?: number
  onCancel?: () => void
}

export function LoadingFallback({
  message = 'Loading...',
  progress,
  onCancel,
}: LoadingFallbackProps) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4 p-8">
        {/* Spinner */}
        <Loader2 className="h-12 w-12 animate-spin text-primary" />

        {/* Loading Message */}
        <p className="text-lg font-medium text-foreground">{message}</p>

        {/* Progress Bar (if progress is provided) */}
        {progress !== undefined && (
          <div className="w-64 space-y-2">
            <Progress value={progress} />
            <p className="text-center text-sm text-muted-foreground">
              {Math.round(progress)}%
            </p>
          </div>
        )}

        {/* Cancel Button (if onCancel is provided) */}
        {onCancel && (
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </div>
  )
}
