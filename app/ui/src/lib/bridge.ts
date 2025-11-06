/**
 * WebView bridge implementation for communication with Swift/ObjC host
 * Provides type-safe bridge with development fallback
 */

import type {
  HostBridge,
  BridgeMessage,
  BridgeResponse,
} from '@/types/bridge'
import { BridgeErrorCode } from '@/types/bridge'
import { generateId } from './utils'

// Re-export types that are used elsewhere
export type { BridgeError, SystemInfo } from '@/types/bridge'

// Check if running in WebView
const IS_WEBVIEW = typeof window !==undefined && !!window.webkit?.messageHandlers?.bridge
const IS_DEVELOPMENT = !IS_WEBVIEW && import.meta.env.DEV

// Default timeout for bridge calls (30 seconds)
const DEFAULT_TIMEOUT = 30000

/**
 * Bridge client for communicating with native host
 */
class BridgeClient {
  private pendingRequests = new Map<
    string,
    {
      resolve: (value: unknown) => void
      reject: (error: Error) => void
      timeout: ReturnType<typeof setTimeout>
    }
  >()

  constructor(private readonly timeout = DEFAULT_TIMEOUT) {
    // Listen for responses from Swift
    if (IS_WEBVIEW) {
      window.addEventListener('message', this.handleResponse.bind(this))
    }
  }

  /**
   * Call a bridge method
   */
  async call<T>(method: keyof HostBridge, ...params: unknown[]): Promise<T> {
    const id = generateId('bridge')
    const message: BridgeMessage = {
      id,
      method,
      params,
      timestamp: Date.now(),
    }

    return new Promise<T>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(id)
        reject(this.createError(BridgeErrorCode.TIMEOUT, `Bridge call timeout: ${method}`))
      }, this.timeout)

      this.pendingRequests.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timeout: timeoutId,
      })

      // Send to Swift via WKWebView message handler
      try {
        window.webkit?.messageHandlers?.bridge?.postMessage(message)
      } catch (error) {
        this.pendingRequests.delete(id)
        clearTimeout(timeoutId)
        reject(
          this.createError(
            BridgeErrorCode.NETWORK_ERROR,
            `Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`
          )
        )
      }
    })
  }

  /**
   * Handle response from Swift
   */
  private handleResponse(event: MessageEvent<BridgeResponse>) {
    const { id, result, error } = event.data
    const pending = this.pendingRequests.get(id)

    if (!pending) {
      console.warn(`Received response for unknown request: ${id}`)
      return
    }

    clearTimeout(pending.timeout)
    this.pendingRequests.delete(id)

    if (error) {
      pending.reject(
        new Error(`${error.code}: ${error.message}`)
      )
    } else {
      pending.resolve(result)
    }
  }

  /**
   * Create a bridge error
   */
  private createError(code: BridgeErrorCode, message: string): Error {
    const error = new Error(message)
    error.name = code
    return error
  }

  /**
   * Clean up pending requests
   */
  destroy() {
    this.pendingRequests.forEach(({ timeout, reject }) => {
      clearTimeout(timeout)
      reject(new Error('Bridge client destroyed'))
    })
    this.pendingRequests.clear()
  }
}

// Create bridge client instance
const bridgeClient = new BridgeClient()

/**
 * Development fallback implementation
 * Proxies requests to localhost:8080 daemon
 */
const developmentBridge: HostBridge = {
  async request(path: string, init?: RequestInit) {
    try {
      const response = await fetch(`http://localhost:8080${path}`, {
        ...init,
        headers: {
          'Content-Type': 'application/json',
          ...init?.headers,
        },
      })

      let body: unknown
      const contentType = response.headers.get('content-type')

      if (contentType?.includes('application/json')) {
        body = await response.json()
      } else if (contentType?.includes('text/')) {
        body = await response.text()
      } else {
        body = await response.blob()
      }

      return {
        status: response.status,
        body,
      }
    } catch (error) {
      console.error('Development bridge request failed:', error)
      throw new Error(
        `Failed to connect to daemon: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
    }
  },

  async openPathDialog(kind: 'models' | 'cache') {
    console.log('[Dev Bridge] openPathDialog:', kind)
    // Return mock path for development
    const basePath = '/Users/developer/Library/Application Support/MLXRunner'
    return kind === 'models' ? `${basePath}/models` : `${basePath}/cache`
  },

  async readConfig() {
    console.log('[Dev Bridge] readConfig')
    try {
      const response = await fetch('http://localhost:8080/api/config')
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      return await response.text()
    } catch (error) {
      console.error('[Dev Bridge] readConfig failed:', error)
      // Return default config for development
      return `
server:
  enable_http: true
  http_port: 8080

scheduler:
  max_batch_tokens: 8192
  max_batch_size: 128
  kv_block_size: 16

performance:
  target_latency_ms: 50
  enable_speculative: true
  speculation_length: 4
  kv_persistence: true

models:
  models_dir: ~/Library/Application Support/MLXRunner/models
  cache_dir: ~/Library/Application Support/MLXRunner/cache

logging:
  level: info

telemetry:
  enabled: false
`.trim()
    }
  },

  async writeConfig(yaml: string) {
    console.log('[Dev Bridge] writeConfig:', yaml)
    try {
      await fetch('http://localhost:8080/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'text/yaml' },
        body: yaml,
      })
    } catch (error) {
      console.error('[Dev Bridge] writeConfig failed:', error)
      throw error
    }
  },

  async startDaemon() {
    console.log('[Dev Bridge] startDaemon')
    // In development, daemon should already be running
    console.warn('[Dev] Daemon should be started manually in development mode')
  },

  async stopDaemon() {
    console.log('[Dev Bridge] stopDaemon')
    console.warn('[Dev] Daemon should be stopped manually in development mode')
  },

  async getVersion() {
    console.log('[Dev Bridge] getVersion')
    return {
      app: '0.1.0-dev',
      daemon: '0.1.0-dev',
    }
  },

  async showNotification(title: string, body: string, identifier?: string) {
    console.log('[Dev Bridge] showNotification:', { title, body, identifier })
    // In development, just log to console
    if (Notification.permission === 'granted') {
      new Notification(title, { body })
    }
  },

  async openURL(url: string) {
    console.log('[Dev Bridge] openURL:', url)
    window.open(url, '_blank')
  },

  async getSystemInfo() {
    console.log('[Dev Bridge] getSystemInfo')
    return {
      platform: 'darwin' as const,
      arch: navigator.platform,
      osVersion: navigator.userAgent,
      modelName: 'Development Machine',
      totalMemory: (navigator as typeof navigator & { deviceMemory?: number }).deviceMemory ? (navigator as typeof navigator & { deviceMemory: number }).deviceMemory * 1024 * 1024 * 1024 : 16 * 1024 * 1024 * 1024,
      freeMemory: 8 * 1024 * 1024 * 1024,
      cpuCount: navigator.hardwareConcurrency || 8,
      metalSupported: true,
    }
  },

  async checkForUpdates() {
    console.log('[Dev Bridge] checkForUpdates')
    console.warn('[Dev] Update checking not available in development mode')
  },
}

/**
 * Production bridge implementation using BridgeClient
 */
const productionBridge: HostBridge = {
  request: (path, init) => bridgeClient.call('request', path, init),
  openPathDialog: kind => bridgeClient.call('openPathDialog', kind),
  readConfig: () => bridgeClient.call('readConfig'),
  writeConfig: yaml => bridgeClient.call('writeConfig', yaml),
  startDaemon: () => bridgeClient.call('startDaemon'),
  stopDaemon: () => bridgeClient.call('stopDaemon'),
  getVersion: () => bridgeClient.call('getVersion'),
  showNotification: (title, body, identifier) =>
    bridgeClient.call('showNotification', title, body, identifier),
  openURL: url => bridgeClient.call('openURL', url),
  getSystemInfo: () => bridgeClient.call('getSystemInfo'),
  checkForUpdates: () => bridgeClient.call('checkForUpdates'),
}

/**
 * Export the appropriate bridge based on environment
 */
export const bridge: HostBridge = IS_WEBVIEW
  ? productionBridge
  : developmentBridge

/**
 * Check if bridge is available
 */
export function isBridgeAvailable(): boolean {
  return IS_WEBVIEW || IS_DEVELOPMENT
}

/**
 * Get bridge mode
 */
export function getBridgeMode(): 'webview' | 'development' | 'none' {
  if (IS_WEBVIEW) return 'webview'
  if (IS_DEVELOPMENT) return 'development'
  return 'none'
}

/**
 * Inject bridge into window for Swift interop
 */
if (typeof window !== 'undefined') {
  ;(window as typeof window & { __HOST__?: HostBridge }).__HOST__ = bridge
}

/**
 * Clean up on unload
 */
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    bridgeClient.destroy()
  })
}
