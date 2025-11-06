/**
 * Internationalization (i18n) utilities for MLXR
 * Currently supports English only, with structure for future expansion
 */

export type Locale = 'en'

export interface TranslationKeys {
  // Common
  common: {
    ok: string
    cancel: string
    save: string
    delete: string
    edit: string
    close: string
    back: string
    next: string
    loading: string
    error: string
    success: string
    warning: string
    info: string
    confirm: string
    search: string
    clear: string
    refresh: string
    copy: string
    copied: string
  }

  // App
  app: {
    title: string
    description: string
  }

  // Chat
  chat: {
    title: string
    placeholder: string
    newChat: string
    deleteChat: string
    clearHistory: string
    stopGenerating: string
    regenerate: string
    copyMessage: string
    tokenCount: string
    latency: string
  }

  // Models
  models: {
    title: string
    noModels: string
    loading: string
    import: string
    pull: string
    delete: string
    showDetails: string
    architecture: string
    format: string
    quantization: string
    parameters: string
    contextLength: string
    size: string
  }

  // Metrics
  metrics: {
    title: string
    latency: string
    throughput: string
    memory: string
    kvCache: string
    requests: string
    tokensPerSecond: string
    prefillLatency: string
    decodeLatency: string
    cacheHitRate: string
    activeRequests: string
    queuedRequests: string
  }

  // Settings
  settings: {
    title: string
    general: string
    performance: string
    paths: string
    updates: string
    privacy: string
    unsavedChanges: string
    saveChanges: string
    discardChanges: string
  }

  // Logs
  logs: {
    title: string
    level: string
    timestamp: string
    message: string
    clear: string
    autoScroll: string
    filter: string
  }

  // Sampling
  sampling: {
    temperature: string
    topP: string
    topK: string
    maxTokens: string
    repetitionPenalty: string
    reset: string
  }

  // Errors
  errors: {
    network: string
    timeout: string
    unknown: string
    modelNotFound: string
    daemonNotRunning: string
    invalidConfig: string
  }

  // Notifications
  notifications: {
    modelDeleted: string
    modelImported: string
    configSaved: string
    daemonStarted: string
    daemonStopped: string
    copySuccess: string
    copyError: string
  }
}

/**
 * English translations
 */
const en: TranslationKeys = {
  common: {
    ok: 'OK',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    close: 'Close',
    back: 'Back',
    next: 'Next',
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    warning: 'Warning',
    info: 'Info',
    confirm: 'Confirm',
    search: 'Search',
    clear: 'Clear',
    refresh: 'Refresh',
    copy: 'Copy',
    copied: 'Copied!',
  },

  app: {
    title: 'MLXR',
    description: 'High-performance LLM inference engine for Apple Silicon',
  },

  chat: {
    title: 'Chat',
    placeholder: 'Type a message...',
    newChat: 'New Chat',
    deleteChat: 'Delete Chat',
    clearHistory: 'Clear History',
    stopGenerating: 'Stop Generating',
    regenerate: 'Regenerate',
    copyMessage: 'Copy Message',
    tokenCount: 'Tokens',
    latency: 'Latency',
  },

  models: {
    title: 'Models',
    noModels: 'No models found',
    loading: 'Loading models...',
    import: 'Import Model',
    pull: 'Pull from Registry',
    delete: 'Delete Model',
    showDetails: 'Show Details',
    architecture: 'Architecture',
    format: 'Format',
    quantization: 'Quantization',
    parameters: 'Parameters',
    contextLength: 'Context Length',
    size: 'Size',
  },

  metrics: {
    title: 'Metrics',
    latency: 'Latency',
    throughput: 'Throughput',
    memory: 'Memory',
    kvCache: 'KV Cache',
    requests: 'Requests',
    tokensPerSecond: 'Tokens/Second',
    prefillLatency: 'Prefill Latency',
    decodeLatency: 'Decode Latency',
    cacheHitRate: 'Cache Hit Rate',
    activeRequests: 'Active Requests',
    queuedRequests: 'Queued Requests',
  },

  settings: {
    title: 'Settings',
    general: 'General',
    performance: 'Performance',
    paths: 'Paths',
    updates: 'Updates',
    privacy: 'Privacy',
    unsavedChanges: 'You have unsaved changes',
    saveChanges: 'Save Changes',
    discardChanges: 'Discard Changes',
  },

  logs: {
    title: 'Logs',
    level: 'Level',
    timestamp: 'Timestamp',
    message: 'Message',
    clear: 'Clear Logs',
    autoScroll: 'Auto-scroll',
    filter: 'Filter',
  },

  sampling: {
    temperature: 'Temperature',
    topP: 'Top P',
    topK: 'Top K',
    maxTokens: 'Max Tokens',
    repetitionPenalty: 'Repetition Penalty',
    reset: 'Reset to Defaults',
  },

  errors: {
    network: 'Network error. Please check your connection.',
    timeout: 'Request timed out. Please try again.',
    unknown: 'An unknown error occurred.',
    modelNotFound: 'Model not found.',
    daemonNotRunning: 'Daemon is not running. Please start the daemon.',
    invalidConfig: 'Invalid configuration.',
  },

  notifications: {
    modelDeleted: 'Model deleted successfully',
    modelImported: 'Model imported successfully',
    configSaved: 'Configuration saved successfully',
    daemonStarted: 'Daemon started successfully',
    daemonStopped: 'Daemon stopped successfully',
    copySuccess: 'Copied to clipboard',
    copyError: 'Failed to copy to clipboard',
  },
}

/**
 * Translations map
 */
const translations: Record<Locale, TranslationKeys> = {
  en,
}

/**
 * Current locale (hardcoded to English for now)
 */
let currentLocale: Locale = 'en'

/**
 * Get current locale
 */
export function getLocale(): Locale {
  return currentLocale
}

/**
 * Set current locale
 * @param locale - Locale to set
 */
export function setLocale(locale: Locale): void {
  if (locale in translations) {
    currentLocale = locale
  } else {
    console.warn(`Locale '${locale}' not found, falling back to 'en'`)
    currentLocale = 'en'
  }
}

/**
 * Get translations for current locale
 */
export function getTranslations(): TranslationKeys {
  return translations[currentLocale]
}

/**
 * Get translation by key path
 * @param keyPath - Dot-separated key path (e.g., 'common.ok')
 * @returns Translation string or key path if not found
 */
export function t(keyPath: string): string {
  const keys = keyPath.split('.')
  let value: unknown = translations[currentLocale]

  for (const key of keys) {
    if (value && typeof value === 'object' && key in value) {
      value = (value as Record<string, unknown>)[key]
    } else {
      console.warn(`Translation key '${keyPath}' not found`)
      return keyPath
    }
  }

  return typeof value === 'string' ? value : keyPath
}

/**
 * Get translation with interpolation
 * @param keyPath - Dot-separated key path
 * @param params - Parameters to interpolate
 * @returns Interpolated translation string
 *
 * Example:
 *   t('errors.modelNotFound', { model: 'llama-7b' })
 *   // If translation is "Model {model} not found"
 *   // Returns: "Model llama-7b not found"
 */
export function tInterpolate(
  keyPath: string,
  params: Record<string, string | number>
): string {
  let translation = t(keyPath)

  Object.entries(params).forEach(([key, value]) => {
    translation = translation.replace(new RegExp(`\\{${key}\\}`, 'g'), String(value))
  })

  return translation
}

/**
 * Get translation with pluralization
 * @param keyPath - Base key path
 * @param count - Count for pluralization
 * @returns Pluralized translation string
 *
 * Example:
 *   tPlural('items', 0) // "items"
 *   tPlural('items', 1) // "item"
 *   tPlural('items', 5) // "items"
 */
export function tPlural(keyPath: string, count: number): string {
  const singularKey = `${keyPath}.singular`
  const pluralKey = `${keyPath}.plural`

  if (count === 1) {
    const singular = t(singularKey)
    return singular !== singularKey ? singular : t(keyPath)
  }

  const plural = t(pluralKey)
  return plural !== pluralKey ? plural : t(keyPath)
}

/**
 * Format number according to locale
 * @param num - Number to format
 * @param options - Intl.NumberFormat options
 * @returns Formatted number string
 */
export function formatNumber(
  num: number,
  options?: Intl.NumberFormatOptions
): string {
  // Map locale to BCP 47 language tag
  const localeMap: Record<Locale, string> = {
    en: 'en-US',
  }

  return new Intl.NumberFormat(localeMap[currentLocale], options).format(num)
}

/**
 * Format date according to locale
 * @param date - Date to format
 * @param options - Intl.DateTimeFormat options
 * @returns Formatted date string
 */
export function formatDate(
  date: Date | number,
  options?: Intl.DateTimeFormatOptions
): string {
  const localeMap: Record<Locale, string> = {
    en: 'en-US',
  }

  return new Intl.DateTimeFormat(localeMap[currentLocale], options).format(date)
}

/**
 * Get available locales
 */
export function getAvailableLocales(): Locale[] {
  return Object.keys(translations) as Locale[]
}

/**
 * Check if a locale is available
 */
export function isLocaleAvailable(locale: string): locale is Locale {
  return locale in translations
}

export default {
  getLocale,
  setLocale,
  getTranslations,
  t,
  tInterpolate,
  tPlural,
  formatNumber,
  formatDate,
  getAvailableLocales,
  isLocaleAvailable,
}
