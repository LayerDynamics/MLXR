/**
 * Root Application Component
 *
 * Main application shell with:
 * - Theme management
 * - Router setup with all pages
 * - Layout structure
 * - Global keyboard shortcuts
 * - Error boundaries
 */

import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from './components/ui/error-boundary'
import { Layout } from './components/layout/Layout'
import { NotificationProvider } from './components/layout/NotificationProvider'
import { LoadingFallback } from './components/layout/LoadingFallback'
import { ErrorFallback } from './components/layout/ErrorFallback'
import { useTheme } from './lib/theme'

// Lazy load pages for code splitting
import { lazy, Suspense } from 'react'

const ChatPage = lazy(() => import('./pages/ChatPage'))
const ModelsPage = lazy(() => import('./pages/ModelsPage'))
const PlaygroundPage = lazy(() => import('./pages/PlaygroundPage'))
const MetricsPage = lazy(() => import('./pages/MetricsPage'))
const LogsPage = lazy(() => import('./pages/LogsPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

function App() {
  const { theme } = useTheme()

  // Log theme changes in development
  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log(`MLXR UI - Theme: ${theme}`)
    }
  }, [theme])

  return (
    <ErrorBoundary
      fallback={(error, reset) => <ErrorFallback error={error} resetError={reset} />}
    >
      <BrowserRouter>
        <NotificationProvider>
          <Layout>
            <Suspense fallback={<LoadingFallback message="Loading page..." />}>
              <Routes>
                {/* Main routes */}
                <Route path="/" element={<ChatPage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/playground" element={<PlaygroundPage />} />
                <Route path="/metrics" element={<MetricsPage />} />
                <Route path="/logs" element={<LogsPage />} />
                <Route path="/settings" element={<SettingsPage />} />

                {/* Chat with conversation ID */}
                <Route path="/chat/:conversationId" element={<ChatPage />} />

                {/* Catch-all redirect to home */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </Layout>
        </NotificationProvider>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
