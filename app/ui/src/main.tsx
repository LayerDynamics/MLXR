/**
 * Main Entry Point
 *
 * Bootstrap the React application
 * - Initialize React root
 * - Set up QueryClient provider
 * - Initialize theme on startup
 * - Mount App component
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { Toaster } from '@/components/ui/sonner'
import App from './App'
import { queryClient } from './lib/queryClient'
import { initializeTheme } from './lib/theme'
import './styles/index.css'

// Initialize theme before first render to prevent flash
initializeTheme()

// Error boundary for top-level errors
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Application error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen w-screen items-center justify-center bg-background">
          <div className="flex max-w-md flex-col gap-4 rounded-lg border border-destructive bg-card p-6 text-card-foreground">
            <h1 className="text-2xl font-bold text-destructive">
              Application Error
            </h1>
            <p className="text-muted-foreground">
              An unexpected error occurred. Please restart the application.
            </p>
            {this.state.error && (
              <pre className="overflow-auto rounded bg-muted p-3 text-sm">
                {this.state.error.message}
              </pre>
            )}
            <button
              className="rounded bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
              onClick={() => window.location.reload()}
            >
              Reload Application
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

// Mount the app
const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Failed to find root element')
}

const root = ReactDOM.createRoot(rootElement)

root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <App />
        <Toaster />
        {import.meta.env.DEV && (
          <ReactQueryDevtools
            initialIsOpen={false}
            buttonPosition="bottom-left"
          />
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>
)

// Enable hot module replacement in development
if (import.meta.hot) {
  import.meta.hot.accept()
}

// Log app initialization
console.log(
  `MLXR UI initialized (${import.meta.env.MODE} mode) at ${new Date().toISOString()}`
)
