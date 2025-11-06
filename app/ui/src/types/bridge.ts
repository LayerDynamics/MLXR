/**
 * WebView bridge types for communication between React and Swift/ObjC host
 */

// Bridge interface exposed by the native host
export interface HostBridge {
  /**
   * Proxy HTTP request to daemon via UDS
   * @param path - API path (e.g., '/v1/chat/completions')
   * @param init - Fetch request options
   * @returns Response with status and body
   */
  request(
    path: string,
    init?: RequestInit
  ): Promise<{ status: number; body: unknown }>

  /**
   * Open native file dialog
   * @param kind - Type of path to select
   * @returns Selected path or null if cancelled
   */
  openPathDialog(kind: 'models' | 'cache'): Promise<string | null>

  /**
   * Read server configuration file
   * @returns YAML config as string
   */
  readConfig(): Promise<string>

  /**
   * Write server configuration file
   * @param yaml - YAML config string
   */
  writeConfig(yaml: string): Promise<void>

  /**
   * Start the daemon process
   */
  startDaemon(): Promise<void>

  /**
   * Stop the daemon process
   */
  stopDaemon(): Promise<void>

  /**
   * Get application and daemon versions
   * @returns Version information
   */
  getVersion(): Promise<{ app: string; daemon: string }>

  /**
   * Show notification (native macOS notification center)
   * @param title - Notification title
   * @param body - Notification body
   * @param identifier - Optional identifier for grouping
   */
  showNotification?(
    title: string,
    body: string,
    identifier?: string
  ): Promise<void>

  /**
   * Open URL in default browser
   * @param url - URL to open
   */
  openURL?(url: string): Promise<void>

  /**
   * Get system info
   * @returns System information
   */
  getSystemInfo?(): Promise<SystemInfo>

  /**
   * Check for updates (Sparkle)
   */
  checkForUpdates?(): Promise<void>
}

// Message structure for bridge communication
export interface BridgeMessage<T = unknown> {
  id: string
  method: keyof HostBridge
  params: T[]
  timestamp: number
}

// Response structure from bridge
export interface BridgeResponse<T = unknown> {
  id: string
  result?: T
  error?: BridgeError
}

// Bridge error structure
export interface BridgeError {
  code: BridgeErrorCode
  message: string
  details?: unknown
}

// Error codes
export enum BridgeErrorCode {
  // Connection errors
  DAEMON_NOT_RUNNING = 'DAEMON_NOT_RUNNING',
  UDS_CONNECTION_FAILED = 'UDS_CONNECTION_FAILED',
  NETWORK_ERROR = 'NETWORK_ERROR',

  // Permission errors
  UDS_PERMISSION_DENIED = 'UDS_PERMISSION_DENIED',
  FILE_PERMISSION_DENIED = 'FILE_PERMISSION_DENIED',

  // Operation errors
  TIMEOUT = 'TIMEOUT',
  INVALID_REQUEST = 'INVALID_REQUEST',
  METHOD_NOT_FOUND = 'METHOD_NOT_FOUND',

  // Configuration errors
  CONFIG_READ_ERROR = 'CONFIG_READ_ERROR',
  CONFIG_WRITE_ERROR = 'CONFIG_WRITE_ERROR',
  CONFIG_PARSE_ERROR = 'CONFIG_PARSE_ERROR',

  // Daemon errors
  DAEMON_START_FAILED = 'DAEMON_START_FAILED',
  DAEMON_STOP_FAILED = 'DAEMON_STOP_FAILED',
  DAEMON_ALREADY_RUNNING = 'DAEMON_ALREADY_RUNNING',

  // Generic error
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

// System information from host
export interface SystemInfo {
  platform: 'darwin' | 'linux' | 'win32'
  arch: string
  osVersion: string
  modelName: string
  totalMemory: number
  freeMemory: number
  cpuCount: number
  gpuName?: string
  gpuMemory?: number
  metalSupported: boolean
}

// Window interface augmentation
declare global {
  interface Window {
    __HOST__: HostBridge

    // WebKit message handlers (for sending messages to Swift)
    webkit?: {
      messageHandlers?: {
        bridge?: {
          postMessage: (message: BridgeMessage) => void
        }
      }
    }

    // Development mode flag
    __DEV__?: boolean
  }
}

// Helper type to check if bridge is available
export type BridgeAvailability = {
  available: boolean
  reason?: string
}

// Bridge connection status
export enum BridgeStatus {
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  ERROR = 'error',
}

// Bridge event types
export interface BridgeEvent {
  type: 'daemon_status_changed' | 'config_changed' | 'error'
  data: unknown
  timestamp: number
}

// Export empty object to make this a module
export {}
