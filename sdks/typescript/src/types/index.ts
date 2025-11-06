/**
 * MLXR TypeScript SDK - Type Definitions
 */

export * from './openai';
export * from './ollama';

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
