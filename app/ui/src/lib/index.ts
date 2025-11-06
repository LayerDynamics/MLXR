/**
 * Main library exports for MLXR frontend
 * Re-exports all library modules for convenient import
 */

// API client
export * from './api'
export { default as api } from './api'

// Bridge
export * from './bridge'

// State management
export * from './store'

// Query client
export * from './queryClient'
export { default as queryClient } from './queryClient'

// SSE utilities
export * from './sse'

// Theme management
export * from './theme'
export { default as useTheme } from './theme'

// Utilities
export { cn, generateId, formatBytes, formatDuration, formatTokensPerSecond } from './utils'

// i18n - import default to avoid formatNumber conflict
export { default as i18n } from './i18n'
export { t, tInterpolate, tPlural, formatDate, getLocale, setLocale } from './i18n'
// Note: formatNumber is exported from i18n and conflicts with utils.formatNumber

// Keyboard shortcuts
export * from './keyboard'

// Markdown utilities
export * from './markdown'
