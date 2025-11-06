/**
 * ErrorFallback Component
 *
 * Full-page error display with:
 * - Error message and stack
 * - Retry button
 * - Report issue button (opens GitHub)
 * - Copy error details
 */

import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'

interface ErrorFallbackProps {
  error: Error
  resetError?: () => void
}

export function ErrorFallback({ error, resetError }: ErrorFallbackProps) {
  const handleCopyError = () => {
    const errorText = `Error: ${error.message}\n\nStack Trace:\n${error.stack}`
    navigator.clipboard.writeText(errorText)
  }

  const handleReportIssue = () => {
    const issueUrl = `https://github.com/anthropics/mlxr/issues/new?title=${encodeURIComponent(
      `Error: ${error.message}`
    )}&body=${encodeURIComponent(
      `## Error\n\n\`\`\`\n${error.message}\n\`\`\`\n\n## Stack Trace\n\n\`\`\`\n${error.stack || 'No stack trace available'}\n\`\`\``
    )}`
    window.open(issueUrl, '_blank')
  }

  return (
    <div className="flex h-full w-full items-center justify-center bg-background p-4">
      <Card className="max-w-2xl">
        <CardHeader>
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-6 w-6 text-destructive" />
            <div>
              <CardTitle className="text-destructive">An error occurred</CardTitle>
              <CardDescription>
                Something went wrong while loading this page
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="mb-2 text-sm font-medium">Error Message:</p>
            <pre className="overflow-auto rounded-md bg-muted p-3 text-xs">
              {error.message}
            </pre>
          </div>
          {error.stack && (
            <div>
              <p className="mb-2 text-sm font-medium">Stack Trace:</p>
              <pre className="max-h-48 overflow-auto rounded-md bg-muted p-3 text-xs">
                {error.stack}
              </pre>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-wrap gap-2">
          {resetError && (
            <Button onClick={resetError} variant="default">
              Try Again
            </Button>
          )}
          <Button onClick={handleCopyError} variant="outline">
            Copy Error Details
          </Button>
          <Button onClick={handleReportIssue} variant="ghost">
            Report Issue on GitHub
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
