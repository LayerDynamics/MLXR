/**
 * NotificationProvider Component
 *
 * Global notification system with:
 * - Toast notification setup
 * - Global notification state
 * - Error notification on API failure
 * - Success notification on actions
 */

import { ReactNode, createContext, useContext, useCallback } from 'react'
import { toast } from 'sonner'

interface NotificationContextValue {
  success: (message: string, description?: string) => void
  error: (message: string, description?: string) => void
  info: (message: string, description?: string) => void
  warning: (message: string, description?: string) => void
  promise: <T,>(
    promise: Promise<T>,
    messages: {
      loading: string
      success: string | ((data: T) => string)
      error: string | ((error: Error) => string)
    }
  ) => Promise<T>
}

const NotificationContext = createContext<NotificationContextValue | null>(null)

interface NotificationProviderProps {
  children: ReactNode
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const success = useCallback((message: string, description?: string) => {
    toast.success(message, { description })
  }, [])

  const error = useCallback((message: string, description?: string) => {
    toast.error(message, { description })
  }, [])

  const info = useCallback((message: string, description?: string) => {
    toast.info(message, { description })
  }, [])

  const warning = useCallback((message: string, description?: string) => {
    toast.warning(message, { description })
  }, [])

  const promiseNotification = useCallback(
    <T,>(
      promise: Promise<T>,
      messages: {
        loading: string
        success: string | ((data: T) => string)
        error: string | ((error: Error) => string)
      }
    ): Promise<T> => {
      toast.promise(promise, {
        loading: messages.loading,
        success: (data) =>
          typeof messages.success === 'function'
            ? messages.success(data)
            : messages.success,
        error: (err) =>
          typeof messages.error === 'function'
            ? messages.error(err as Error)
            : messages.error,
      })
      return promise
    },
    []
  )

  const value: NotificationContextValue = {
    success,
    error,
    info,
    warning,
    promise: promiseNotification,
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

/**
 * Hook to access notification functions
 */
export function useNotifications() {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}
