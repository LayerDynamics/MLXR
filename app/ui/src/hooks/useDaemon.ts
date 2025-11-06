/**
 * useDaemon Hook
 *
 * Daemon status and control
 * - Status query with polling
 * - Start/stop/restart mutations
 * - Health check
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { backend } from '@/lib/api'
import { queryKeys } from '@/lib/queryClient'

export interface DaemonStatus {
  status: 'running' | 'stopped' | 'starting' | 'stopping' | 'error'
  pid?: number
  uptime?: number
  version?: string
}

export interface UseDaemonReturn {
  status: DaemonStatus | undefined
  isLoading: boolean
  error: Error | null
  start: () => Promise<void>
  stop: () => Promise<void>
  restart: () => Promise<void>
  refetch: () => void
}

/**
 * Hook for daemon status and control
 */
export function useDaemon(): UseDaemonReturn {
  const queryClient = useQueryClient()

  // Poll daemon status every 1 second
  const {
    data: status,
    isLoading,
    error,
    refetch,
  } = useQuery<DaemonStatus, Error>({
    queryKey: queryKeys.daemon.status(),
    queryFn: async () => {
      try {
        const health = await backend.health()
        return {
          status: 'running' as const,
          version: health.version,
        }
      } catch {
        return {
          status: 'stopped' as const,
        }
      }
    },
    refetchInterval: 1000, // Poll every second
    retry: false, // Don't retry on failure
  })

  // Start daemon mutation
  const startMutation = useMutation({
    mutationFn: async () => {
      // Call bridge to start daemon
      if (window.__HOST__) {
        await window.__HOST__.startDaemon()
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.daemon.status() })
    },
  })

  // Stop daemon mutation
  const stopMutation = useMutation({
    mutationFn: async () => {
      // Call bridge to stop daemon
      if (window.__HOST__) {
        await window.__HOST__.stopDaemon()
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.daemon.status() })
    },
  })

  // Restart helper
  const restart = async () => {
    await stopMutation.mutateAsync()
    await new Promise((resolve) => setTimeout(resolve, 1000)) // Wait 1 second
    await startMutation.mutateAsync()
  }

  return {
    status,
    isLoading,
    error: error as Error | null,
    start: async () => startMutation.mutateAsync(),
    stop: async () => stopMutation.mutateAsync(),
    restart,
    refetch,
  }
}
