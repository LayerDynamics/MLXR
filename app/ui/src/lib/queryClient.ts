/**
 * TanStack Query configuration and client setup
 * Handles server state caching, refetching, and synchronization
 */

import { QueryClient, QueryCache, MutationCache } from '@tanstack/react-query'
import { useAppStore } from './store'
import { APIError, backend } from './api'

/**
 * Global error handler for queries
 */
const handleQueryError = (error: unknown) => {
  console.error('Query error:', error)

  // Add notification via store
  const { addNotification } = useAppStore.getState()

  if (error instanceof APIError) {
    addNotification({
      type: 'error',
      title: 'API Error',
      message: error.message,
      duration: 5000,
      timestamp: Date.now(),
    })
  } else if (error instanceof Error) {
    addNotification({
      type: 'error',
      title: 'Error',
      message: error.message,
      duration: 5000,
      timestamp: Date.now(),
    })
  }
}

/**
 * Global error handler for mutations
 */
const handleMutationError = (error: unknown) => {
  console.error('Mutation error:', error)

  const { addNotification } = useAppStore.getState()

  if (error instanceof APIError) {
    addNotification({
      type: 'error',
      title: 'Operation Failed',
      message: error.message,
      duration: 5000,
      timestamp: Date.now(),
    })
  } else if (error instanceof Error) {
    addNotification({
      type: 'error',
      title: 'Operation Failed',
      message: error.message,
      duration: 5000,
      timestamp: Date.now(),
    })
  }
}

/**
 * Create and configure the QueryClient
 */
export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: handleQueryError,
  }),
  mutationCache: new MutationCache({
    onError: handleMutationError,
  }),
  defaultOptions: {
    queries: {
      // Stale time: how long data is considered fresh (5 minutes default)
      staleTime: 5 * 60 * 1000,

      // Cache time: how long inactive data stays in cache (30 minutes default)
      gcTime: 30 * 60 * 1000,

      // Retry configuration
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors (client errors)
        if (error instanceof APIError && error.status && error.status >= 400 && error.status < 500) {
          return false
        }
        // Retry up to 3 times for network/server errors
        return failureCount < 3
      },

      // Retry delay with exponential backoff
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),

      // Refetch configuration
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
      refetchOnMount: true,

      // Network mode
      networkMode: 'online',
    },
    mutations: {
      // Retry mutations once on network error
      retry: (failureCount, error) => {
        if (error instanceof APIError && error.status && error.status >= 400 && error.status < 500) {
          return false
        }
        return failureCount < 1
      },

      networkMode: 'online',
    },
  },
})

/**
 * Query keys factory for type-safe query key management
 * Follows TanStack Query best practices for hierarchical keys
 */
export const queryKeys = {
  // Models
  models: {
    all: ['models'] as const,
    lists: () => [...queryKeys.models.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) =>
      [...queryKeys.models.lists(), filters] as const,
    details: () => [...queryKeys.models.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.models.details(), id] as const,
  },

  // Conversations
  conversations: {
    all: ['conversations'] as const,
    lists: () => [...queryKeys.conversations.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) =>
      [...queryKeys.conversations.lists(), filters] as const,
    details: () => [...queryKeys.conversations.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.conversations.details(), id] as const,
  },

  // Requests (active inference requests)
  requests: {
    all: ['requests'] as const,
    lists: () => [...queryKeys.requests.all, 'list'] as const,
    list: () => [...queryKeys.requests.lists()] as const,
    details: () => [...queryKeys.requests.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.requests.details(), id] as const,
  },

  // Stats
  stats: {
    all: ['stats'] as const,
    scheduler: () => [...queryKeys.stats.all, 'scheduler'] as const,
    registry: () => [...queryKeys.stats.all, 'registry'] as const,
    kv_cache: () => [...queryKeys.stats.all, 'kv_cache'] as const,
    speculative: () => [...queryKeys.stats.all, 'speculative'] as const,
  },

  // Metrics
  metrics: {
    all: ['metrics'] as const,
    prometheus: () => [...queryKeys.metrics.all, 'prometheus'] as const,
    snapshot: () => [...queryKeys.metrics.all, 'snapshot'] as const,
    timeseries: (metric: string, range: string) =>
      [...queryKeys.metrics.all, 'timeseries', metric, range] as const,
  },

  // Daemon
  daemon: {
    all: ['daemon'] as const,
    status: () => [...queryKeys.daemon.all, 'status'] as const,
  },

  // Health
  health: ['health'] as const,

  // Config
  config: {
    all: ['config'] as const,
    server: () => [...queryKeys.config.all, 'server'] as const,
  },

  // System
  system: {
    all: ['system'] as const,
    info: () => [...queryKeys.system.all, 'info'] as const,
    version: () => [...queryKeys.system.all, 'version'] as const,
  },
}

/**
 * Mutation keys for tracking mutation state
 */
export const mutationKeys = {
  // Models
  deleteModel: (id: string) => ['deleteModel', id] as const,
  importModel: ['importModel'] as const,
  pullModel: (name: string) => ['pullModel', name] as const,

  // Conversations
  createConversation: ['createConversation'] as const,
  updateConversation: (id: string) => ['updateConversation', id] as const,
  deleteConversation: (id: string) => ['deleteConversation', id] as const,

  // Requests
  cancelRequest: (id: string) => ['cancelRequest', id] as const,

  // Chat
  sendMessage: (conversationId: string) => ['sendMessage', conversationId] as const,

  // Config
  updateConfig: ['updateConfig'] as const,

  // Daemon
  startDaemon: ['startDaemon'] as const,
  stopDaemon: ['stopDaemon'] as const,
}

/**
 * Invalidation helper - invalidate related queries after mutations
 */
export const invalidateQueries = {
  models: () => queryClient.invalidateQueries({ queryKey: queryKeys.models.all }),
  conversations: () =>
    queryClient.invalidateQueries({ queryKey: queryKeys.conversations.all }),
  requests: () => queryClient.invalidateQueries({ queryKey: queryKeys.requests.all }),
  stats: () => queryClient.invalidateQueries({ queryKey: queryKeys.stats.all }),
  metrics: () => queryClient.invalidateQueries({ queryKey: queryKeys.metrics.all }),
  all: () => queryClient.invalidateQueries(),
}

/**
 * Prefetch helper - prefetch data before navigation
 */
export const prefetchQueries = {
  models: async (filters?: Record<string, unknown>) => {
    return queryClient.prefetchQuery({
      queryKey: queryKeys.models.list(filters),
      queryFn: () => backend.listModels(filters),
      staleTime: 5 * 60 * 1000,
    })
  },

  schedulerStats: async () => {
    return queryClient.prefetchQuery({
      queryKey: queryKeys.stats.scheduler(),
      queryFn: () => backend.getSchedulerStats(),
      staleTime: 10 * 1000, // 10 seconds for frequently changing data
    })
  },

  health: async () => {
    return queryClient.prefetchQuery({
      queryKey: queryKeys.health,
      queryFn: () => backend.health(),
      staleTime: 30 * 1000,
    })
  },
}

/**
 * Cache update helpers for optimistic updates
 */
export const updateCache = {
  /**
   * Update a specific model in the cache
   */
  updateModel: (id: string, updater: (old: unknown) => unknown) => {
    queryClient.setQueryData(queryKeys.models.detail(id), updater)
    // Also invalidate the list to ensure consistency
    invalidateQueries.models()
  },

  /**
   * Update a specific conversation in the cache
   */
  updateConversation: (id: string, updater: (old: unknown) => unknown) => {
    queryClient.setQueryData(queryKeys.conversations.detail(id), updater)
    invalidateQueries.conversations()
  },

  /**
   * Add a new message to a conversation (optimistic update)
   */
  addMessageToConversation: (conversationId: string, message: unknown) => {
    queryClient.setQueryData(
      queryKeys.conversations.detail(conversationId),
      (old: unknown) => {
        if (!old || typeof old !== 'object') return old
        const conversation = old as { messages?: unknown[]; [key: string]: unknown }
        return {
          ...conversation,
          messages: [...(conversation.messages || []), message],
          updated_at: Date.now(),
        }
      }
    )
  },
}

/**
 * Subscribe to query changes for real-time updates
 */
export const subscribeToQuery = (
  queryKey: unknown[],
  callback: (data: unknown) => void
) => {
  return queryClient.getQueryCache().subscribe(event => {
    if (event.type === 'updated' && event.query.queryKey === queryKey) {
      callback(event.query.state.data)
    }
  })
}

export default queryClient
