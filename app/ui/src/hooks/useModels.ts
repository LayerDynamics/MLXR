/**
 * useModels Hook
 *
 * Model management with TanStack Query
 * - List models query
 * - Get model details query
 * - Delete model mutation
 * - Works with existing backend API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { ModelInfo, QueryOptions } from '@/types/backend'
import { backend } from '@/lib/api'
import { queryKeys } from '@/lib/queryClient'

export interface UseModelsOptions {
  /**
   * Auto-refresh interval in milliseconds
   * Set to false to disable auto-refresh
   */
  refetchInterval?: number | false

  /**
   * Query options for filtering models
   */
  queryOptions?: QueryOptions

  /**
   * Callback when a model is successfully deleted
   */
  onModelDeleted?: (modelId: string) => void

  /**
   * Error handler
   */
  onError?: (error: Error) => void
}

export interface UseModelsReturn {
  // Queries
  models: ModelInfo[]
  isLoading: boolean
  isError: boolean
  error: Error | null
  refetch: () => void

  // Model details
  getModel: (modelId: string) => ModelInfo | undefined

  // Mutations
  deleteModel: (modelId: string) => Promise<void>

  // Mutation states
  isDeletingModel: boolean
}

/**
 * Hook for model management with TanStack Query
 */
export function useModels(options: UseModelsOptions = {}): UseModelsReturn {
  const {
    refetchInterval = 5000, // Default: refresh every 5 seconds
    queryOptions,
    onModelDeleted,
    onError,
  } = options

  const queryClient = useQueryClient()

  // Query for model list
  const {
    data: models,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<ModelInfo[], Error>({
    queryKey: queryKeys.models.list(),
    queryFn: () => backend.listModels(queryOptions),
    refetchInterval,
    staleTime: 1000, // Consider stale after 1 second
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  })

  // Helper to get a specific model
  const getModel = (modelId: string): ModelInfo | undefined => {
    return models?.find(m => m.model_id === modelId)
  }

  // Delete model mutation
  const deleteModelMutation = useMutation({
    mutationFn: backend.deleteModel,
    onSuccess: (_data, modelId) => {
      // Invalidate models list to refresh
      queryClient.invalidateQueries({ queryKey: queryKeys.models.list() })
      onModelDeleted?.(modelId)
    },
    onError: (err: Error) => {
      console.error('Failed to delete model:', err)
      onError?.(err)
    },
  })

  return {
    // Queries
    models: models || [],
    isLoading,
    isError,
    error: error as Error | null,
    refetch,

    // Model details
    getModel,

    // Mutations
    deleteModel: deleteModelMutation.mutateAsync,

    // Mutation states
    isDeletingModel: deleteModelMutation.isPending,
  }
}

/**
 * Hook for a specific model's details
 */
export function useModel(modelId: string | undefined) {
  const {
    data: model,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<ModelInfo, Error>({
    queryKey: queryKeys.models.detail(modelId!),
    queryFn: () => backend.getModel(modelId!),
    enabled: !!modelId,
    staleTime: 5000,
    gcTime: 5 * 60 * 1000,
  })

  return {
    model,
    isLoading,
    isError,
    error: error as Error | null,
    refetch,
  }
}

/**
 * Hook for model search and filtering
 */
export function useModelSearch(
  searchQuery?: string,
  filters?: {
    format?: string[]
    quant_type?: string[]
    is_loaded?: boolean
  }
) {
  const { models, isLoading, isError, error } = useModels()

  const filteredModels = models.filter(model => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      const matchesName = model.name.toLowerCase().includes(query)
      const matchesModelId = model.model_id.toLowerCase().includes(query)
      const matchesArch = model.architecture.toLowerCase().includes(query)
      if (!matchesName && !matchesModelId && !matchesArch) {
        return false
      }
    }

    // Format filter
    if (filters?.format && filters.format.length > 0) {
      if (!filters.format.includes(model.format)) {
        return false
      }
    }

    // Quantization type filter
    if (filters?.quant_type && filters.quant_type.length > 0) {
      if (!filters.quant_type.includes(model.quant_type)) {
        return false
      }
    }

    // Loaded filter
    if (filters?.is_loaded !== undefined) {
      if (model.is_loaded !== filters.is_loaded) {
        return false
      }
    }

    return true
  })

  return {
    models: filteredModels,
    isLoading,
    isError,
    error,
  }
}
