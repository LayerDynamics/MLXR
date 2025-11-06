/**
 * Error Boundary Component
 *
 * Catches React errors and displays a fallback UI
 * - Error reporting
 * - Retry functionality
 * - Customizable fallback UI
 */

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Button } from './button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './card'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: (error: Error, reset: () => void) => ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
  resetKeys?: unknown[]
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

/**
 * Error Boundary to catch and handle React errors
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
    }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error
    console.error('ErrorBoundary caught an error:', error, errorInfo)

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    // Reset error boundary if resetKeys change
    if (
      this.state.hasError &&
      this.props.resetKeys &&
      prevProps.resetKeys &&
      !this.areResetKeysEqual(prevProps.resetKeys, this.props.resetKeys)
    ) {
      this.reset()
    }
  }

  areResetKeysEqual(prevKeys: unknown[], nextKeys: unknown[]): boolean {
    if (prevKeys.length !== nextKeys.length) {
      return false
    }
    return prevKeys.every((key, index) => key === nextKeys[index])
  }

  reset = (): void => {
    this.setState({
      hasError: false,
      error: null,
    })
  }

  render(): ReactNode {
    if (this.state.hasError && this.state.error) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset)
      }

      // Default fallback UI
      return (
        <div className="flex h-full w-full items-center justify-center p-4">
          <Card className="max-w-lg">
            <CardHeader>
              <CardTitle className="text-destructive">Something went wrong</CardTitle>
              <CardDescription>
                An error occurred while rendering this component
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <p className="text-sm font-medium">Error Message:</p>
                <pre className="overflow-auto rounded-md bg-muted p-3 text-xs">
                  {this.state.error.message}
                </pre>
                {this.state.error.stack && (
                  <>
                    <p className="text-sm font-medium">Stack Trace:</p>
                    <pre className="max-h-48 overflow-auto rounded-md bg-muted p-3 text-xs">
                      {this.state.error.stack}
                    </pre>
                  </>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex gap-2">
              <Button onClick={this.reset} variant="default">
                Try Again
              </Button>
              <Button
                onClick={() => {
                  navigator.clipboard.writeText(
                    `Error: ${this.state.error!.message}\n\nStack:\n${this.state.error!.stack}`
                  )
                }}
                variant="outline"
              >
                Copy Error
              </Button>
              <Button
                onClick={() => {
                  const issueUrl = `https://github.com/anthropics/mlxr/issues/new?title=${encodeURIComponent(
                    `Error: ${this.state.error!.message}`
                  )}&body=${encodeURIComponent(
                    `## Error\n\n\`\`\`\n${this.state.error!.message}\n\`\`\`\n\n## Stack Trace\n\n\`\`\`\n${this.state.error!.stack}\n\`\`\``
                  )}`
                  window.open(issueUrl, '_blank')
                }}
                variant="ghost"
              >
                Report Issue
              </Button>
            </CardFooter>
          </Card>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Hook version of ErrorBoundary for convenience
 */
export function useErrorHandler(): (error: Error) => void {
  const [, setError] = React.useState()

  return React.useCallback((error: Error) => {
    setError(() => {
      throw error
    })
  }, [])
}
