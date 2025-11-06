/**
 * useConfig Hook
 *
 * Server configuration management
 * - Load config from bridge
 * - Save config via bridge
 * - YAML parsing/serialization
 * - Validation
 * - Reset to defaults
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { parse as parseYAML, stringify as stringifyYAML } from 'yaml'
import { queryKeys } from '@/lib/queryClient'
import type { ServerConfig } from '@/types/config'
import { DEFAULT_SERVER_CONFIG } from '@/types/config'

export interface UseConfigOptions {
  /**
   * Callback when config is loaded
   */
  onLoad?: (config: ServerConfig) => void

  /**
   * Callback when config is saved
   */
  onSave?: (config: ServerConfig) => void

  /**
   * Callback on error
   */
  onError?: (error: Error) => void
}

export interface UseConfigReturn {
  // Config data
  config: ServerConfig | undefined
  configYAML: string | undefined
  isLoading: boolean
  error: Error | null

  // Actions
  updateConfig: (config: ServerConfig) => Promise<void>
  updateConfigYAML: (yaml: string) => Promise<void>
  resetToDefaults: () => Promise<void>
  refetch: () => void

  // Validation
  validate: (config: ServerConfig) => ConfigValidation
  hasUnsavedChanges: boolean
}

export interface ConfigValidation {
  valid: boolean
  errors: string[]
  warnings: string[]
}

/**
 * Hook for server configuration management
 */
export function useConfig(options: UseConfigOptions = {}): UseConfigReturn {
  const { onLoad, onSave, onError } = options
  const queryClient = useQueryClient()

  // Query to load config
  const {
    data: configYAML,
    isLoading,
    error,
    refetch,
  } = useQuery<string, Error>({
    queryKey: queryKeys.config.server(),
    queryFn: async () => {
      // Use bridge to read config
      if (window.__HOST__) {
        try {
          const yaml = await window.__HOST__.readConfig()
          if (onLoad) {
            try {
              const parsed = parseYAML(yaml) as ServerConfig
              onLoad(parsed)
            } catch {
              // Ignore parse errors for onLoad
            }
          }
          return yaml
        } catch (err) {
          const error = err instanceof Error ? err : new Error('Failed to load config')
          if (onError) {
            onError(error)
          }
          throw error
        }
      }
      // Fallback for development
      return stringifyYAML(DEFAULT_SERVER_CONFIG)
    },
    staleTime: 30 * 1000, // 30 seconds
    retry: 1,
  })

  // Parse YAML to config object
  const config = configYAML
    ? (() => {
        try {
          return parseYAML(configYAML) as ServerConfig
        } catch {
          return undefined
        }
      })()
    : undefined

  // Mutation to save config
  const saveMutation = useMutation({
    mutationFn: async (yaml: string) => {
      if (window.__HOST__) {
        await window.__HOST__.writeConfig(yaml)
      }
      return yaml
    },
    onSuccess: (yaml) => {
      // Update cache
      queryClient.setQueryData(queryKeys.config.server(), yaml)

      // Call onSave callback
      if (onSave) {
        try {
          const parsed = parseYAML(yaml) as ServerConfig
          onSave(parsed)
        } catch {
          // Ignore parse errors for onSave
        }
      }
    },
    onError: (err) => {
      const error = err instanceof Error ? err : new Error('Failed to save config')
      if (onError) {
        onError(error)
      }
    },
  })

  // Update config (object)
  const updateConfig = async (newConfig: ServerConfig) => {
    const yaml = stringifyYAML(newConfig)
    await saveMutation.mutateAsync(yaml)
  }

  // Update config (YAML string)
  const updateConfigYAML = async (yaml: string) => {
    // Validate YAML syntax
    try {
      parseYAML(yaml)
    } catch (err) {
      throw new Error(
        `Invalid YAML: ${err instanceof Error ? err.message : 'Parse error'}`
      )
    }

    await saveMutation.mutateAsync(yaml)
  }

  // Reset to defaults
  const resetToDefaults = async () => {
    await updateConfig(DEFAULT_SERVER_CONFIG)
  }

  // Validate config
  const validate = (configToValidate: ServerConfig): ConfigValidation => {
    const errors: string[] = []
    const warnings: string[] = []

    // Validate scheduler settings
    if (configToValidate.scheduler.max_batch_tokens < 1) {
      errors.push('scheduler.max_batch_tokens must be at least 1')
    }
    if (configToValidate.scheduler.max_batch_size < 1) {
      errors.push('scheduler.max_batch_size must be at least 1')
    }
    if (configToValidate.scheduler.kv_block_size < 1) {
      errors.push('scheduler.kv_block_size must be at least 1')
    }

    // Validate performance settings
    if (configToValidate.performance.target_latency_ms < 1) {
      errors.push('performance.target_latency_ms must be at least 1')
    }
    if (configToValidate.performance.speculation_length < 1) {
      warnings.push('performance.speculation_length should be at least 1')
    }

    // Validate memory settings
    if (
      configToValidate.memory.gpu_memory_fraction < 0 ||
      configToValidate.memory.gpu_memory_fraction > 1
    ) {
      errors.push('memory.gpu_memory_fraction must be between 0 and 1')
    }

    // Validate paths
    if (!configToValidate.models.models_dir) {
      errors.push('models.models_dir is required')
    }
    if (!configToValidate.models.cache_dir) {
      errors.push('models.cache_dir is required')
    }

    // Validate logging
    const validLogLevels = ['debug', 'info', 'warn', 'error']
    if (!validLogLevels.includes(configToValidate.logging.level)) {
      errors.push(
        `logging.level must be one of: ${validLogLevels.join(', ')}`
      )
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    }
  }

  // Check for unsaved changes (compare with cache)
  const cachedConfig = queryClient.getQueryData<string>(
    queryKeys.config.server()
  )
  const hasUnsavedChanges = configYAML !== cachedConfig

  return {
    config,
    configYAML,
    isLoading,
    error: error as Error | null,
    updateConfig,
    updateConfigYAML,
    resetToDefaults,
    refetch,
    validate,
    hasUnsavedChanges,
  }
}

/**
 * Hook to check if config needs migration
 */
export function useConfigMigration() {
  const { config } = useConfig()

  // Check if config is from an older version
  const needsMigration = config ? checkConfigVersion(config) : false

  return { needsMigration }
}

/**
 * Check if config needs migration (placeholder for future use)
 */
function checkConfigVersion(_config: ServerConfig): boolean {
  // For now, assume all configs are up to date
  // In the future, check version field and migrate if needed
  return false
}
