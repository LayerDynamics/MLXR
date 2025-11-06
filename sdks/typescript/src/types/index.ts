/**
 * MLXR TypeScript SDK - Type Definitions
 */

export * from './openai';
export * from './ollama';

/**
 * Retry configuration for handling transient network errors
 */
export interface RetryConfig {
  /**
   * Maximum number of retry attempts (default: 3)
   */
  maxRetries?: number;

  /**
   * Initial delay in milliseconds before first retry (default: 1000)
   */
  initialDelay?: number;

  /**
   * Multiplier for exponential backoff (default: 2)
   */
  backoffMultiplier?: number;

  /**
   * Maximum delay in milliseconds between retries (default: 10000)
   */
  maxDelay?: number;

  /**
   * HTTP status codes that should trigger a retry (default: [408, 429, 500, 502, 503, 504])
   */
  retryableStatusCodes?: number[];
}

/**
 * SDK configuration options
 */
export interface MLXRConfig {
  /**
   * Base URL for HTTP connections (default: http://localhost:11434)
   */
  baseUrl?: string;

  /**
   * Unix domain socket path (default: ~/Library/Application Support/MLXRunner/run/mlxrunner.sock)
   */
  unixSocketPath?: string;

  /**
   * API key for authentication (optional)
   */
  apiKey?: string;

  /**
   * Request timeout in milliseconds (default: 30000)
   */
  timeout?: number;

  /**
   * Custom headers to include in requests
   */
  headers?: Record<string, string>;

  /**
   * Prefer Unix domain socket over HTTP (default: true on macOS)
   */
  preferUnixSocket?: boolean;

  /**
   * Retry configuration for handling transient errors (optional)
   */
  retry?: RetryConfig;
}

/**
 * Stream event types
 */
export type StreamEvent<T> = {
  data: T;
  done: boolean;
};

/**
 * Stream callback function
 */
export type StreamCallback<T> = (event: StreamEvent<T>) => void;

/**
 * Error callback function
 */
export type ErrorCallback = (error: Error) => void;
