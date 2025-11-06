/**
 * HTTP client with Unix domain socket and SSE streaming support
 */

import * as http from 'http';
import * as https from 'https';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import type { MLXRConfig, RetryConfig } from '../types';
import { parseSSEStream } from './sse-parser';

export interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  headers?: Record<string, string>;
  body?: unknown;
  stream?: boolean;
}

export interface StreamChunk {
  data: string;
  done: boolean;
}

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Check if an error is retryable
 */
function isRetryableError(
  error: unknown,
  statusCode?: number,
  retryableStatusCodes?: number[]
): boolean {
  // Network errors are retryable
  if (error instanceof Error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (
      code === 'ECONNRESET' ||
      code === 'ETIMEDOUT' ||
      code === 'ENOTFOUND' ||
      code === 'ECONNREFUSED'
    ) {
      return true;
    }
  }

  // HTTP status codes that are retryable
  const defaultRetryableCodes = [408, 429, 500, 502, 503, 504];
  const codes = retryableStatusCodes || defaultRetryableCodes;
  if (statusCode && codes.includes(statusCode)) {
    return true;
  }

  return false;
}

/**
 * HTTP client for MLXR API requests
 */
export class HttpClient {
  private config: Required<
    Omit<MLXRConfig, 'retry'> & { retry: Required<RetryConfig> }
  >;
  private unixSocketExists: boolean = false;

  constructor(config: MLXRConfig = {}) {
    // Set defaults
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:11434',
      unixSocketPath:
        config.unixSocketPath ||
        path.join(
          os.homedir(),
          'Library/Application Support/MLXRunner/run/mlxrunner.sock'
        ),
      apiKey: config.apiKey || '',
      timeout: config.timeout || 30000,
      headers: config.headers || {},
      preferUnixSocket: config.preferUnixSocket ?? process.platform === 'darwin',
      retry: {
        maxRetries: config.retry?.maxRetries ?? 3,
        initialDelay: config.retry?.initialDelay ?? 1000,
        backoffMultiplier: config.retry?.backoffMultiplier ?? 2,
        maxDelay: config.retry?.maxDelay ?? 10000,
        retryableStatusCodes: config.retry?.retryableStatusCodes ?? [
          408, 429, 500, 502, 503, 504,
        ],
      },
    };

    // Check if Unix socket exists
    try {
      if (fs.existsSync(this.config.unixSocketPath)) {
        this.unixSocketExists = true;
      }
    } catch (err) {
      // Socket doesn't exist or not accessible
      this.unixSocketExists = false;
    }
  }

  /**
   * Make a request to the MLXR API with retry logic
   */
  async request<T>(options: RequestOptions): Promise<T> {
    if (options.stream) {
      throw new Error('Use requestStream for streaming requests');
    }

    let lastError: Error | undefined;
    let attempt = 0;
    let delay = this.config.retry.initialDelay;

    while (attempt <= this.config.retry.maxRetries) {
      try {
        return await this.executeRequest<T>(options);
      } catch (error) {
        lastError = error as Error;
        const statusCode = (error as { statusCode?: number }).statusCode;

        // Check if error is retryable
        if (
          attempt < this.config.retry.maxRetries &&
          isRetryableError(error, statusCode, this.config.retry.retryableStatusCodes)
        ) {
          await sleep(delay);
          delay = Math.min(
            delay * this.config.retry.backoffMultiplier,
            this.config.retry.maxDelay
          );
          attempt++;
        } else {
          throw error;
        }
      }
    }

    throw lastError || new Error('Request failed after retries');
  }

  /**
   * Execute a single HTTP request
   */
  private async executeRequest<T>(options: RequestOptions): Promise<T> {
    const useUnixSocket =
      this.config.preferUnixSocket && this.unixSocketExists;

    return new Promise((resolve, reject) => {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...this.config.headers,
        ...options.headers,
      };

      // Add API key if configured
      if (this.config.apiKey) {
        headers['Authorization'] = `Bearer ${this.config.apiKey}`;
      }

      let body: string | undefined;
      if (
      // Only set Content-Length for requests with bodies
        options.body &&
        ['POST', 'PUT', 'PATCH'].includes(options.method)
      ) {
        body = JSON.stringify(options.body);
        headers['Content-Length'] = Buffer.byteLength(body).toString();
      } else if (options.body) {
        body = JSON.stringify(options.body);
      }

      const requestOptions: http.RequestOptions = {
        method: options.method,
        path: options.path,
        headers,
        timeout: this.config.timeout,
      };

      let requestFn: typeof http.request | typeof https.request = http.request;

      if (useUnixSocket) {
        requestOptions.socketPath = this.config.unixSocketPath;
      } else {
        const url = new URL(this.config.baseUrl);
        requestOptions.hostname = url.hostname;
        requestOptions.port = url.port;
        // Select appropriate request function based on protocol
        requestFn = url.protocol === 'https:' ? https.request : http.request;
      }

      const req = requestFn(requestOptions, (res) => {
        let data = '';

        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              const parsed = JSON.parse(data);
              resolve(parsed as T);
            } catch (err) {
              reject(new Error(`Failed to parse response: ${err}`));
            }
          } else {
            // Try to parse error as JSON, but handle non-JSON responses
            let errorMessage = `HTTP ${res.statusCode}`;
            try {
              const error = JSON.parse(data);
              errorMessage = error.error?.message || errorMessage;
            } catch {
              // If not JSON, include the raw response for debugging
              errorMessage = data
                ? `${errorMessage}: ${data.slice(0, 200)}`
                : errorMessage;
            }
            const error = new Error(errorMessage) as Error & {
              statusCode?: number;
            };
            error.statusCode = res.statusCode;
            reject(error);
          }
        });
      });

      req.on('error', (err) => {
        reject(err);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      if (body) {
        req.write(body);
      }

      req.end();
    });
  }

  /**
   * Make a streaming request to the MLXR API
   */
  async *requestStream(
    options: RequestOptions
  ): AsyncGenerator<StreamChunk, void, unknown> {
    const useUnixSocket =
      this.config.preferUnixSocket && this.unixSocketExists;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...this.config.headers,
      ...options.headers,
    };

    // Add API key if configured
    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    let body: string | undefined;
    // Only set Content-Length for requests with bodies
    if (
      options.body &&
      ['POST', 'PUT', 'PATCH'].includes(options.method)
    ) {
      body = JSON.stringify(options.body);
      headers['Content-Length'] = Buffer.byteLength(body).toString();
    } else if (options.body) {
      body = JSON.stringify(options.body);
    }

    const requestOptions: http.RequestOptions = {
      method: options.method,
      path: options.path,
      headers,
      timeout: this.config.timeout,
    };

    let requestFn: typeof http.request | typeof https.request = http.request;

    if (useUnixSocket) {
      requestOptions.socketPath = this.config.unixSocketPath;
    } else {
      const url = new URL(this.config.baseUrl);
      requestOptions.hostname = url.hostname;
      requestOptions.port = url.port;
      // Select appropriate request function based on protocol
      requestFn = url.protocol === 'https:' ? https.request : http.request;
    }

    const generator = await new Promise<
      AsyncGenerator<StreamChunk, void, unknown>
    >((resolve, reject) => {
      const req = requestFn(requestOptions, (res) => {
        if (res.statusCode && (res.statusCode < 200 || res.statusCode >= 300)) {
          let errorData = '';
          res.on('data', (chunk) => {
            errorData += chunk;
          });
          res.on('end', () => {
            // Try to parse error as JSON, but handle non-JSON responses
            let errorMessage = `HTTP ${res.statusCode}`;
            try {
              const error = JSON.parse(errorData);
              errorMessage = error.error?.message || errorMessage;
            } catch {
              // If not JSON, include the raw response for debugging
              errorMessage = errorData
                ? `${errorMessage}: ${errorData.slice(0, 200)}`
                : errorMessage;
            }
            reject(new Error(errorMessage));
          });
          return;
        }

        // Use shared SSE parser utility instead of inline parsing
        const streamGenerator = parseSSEStream(res);
        resolve(streamGenerator);
      });

      req.on('error', (err) => {
        reject(err);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      if (body) {
        req.write(body);
      }

      req.end();
    });

    yield* generator;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<MLXRConfig>): void {
    this.config = {
      ...this.config,
      ...config,
      retry: {
        ...this.config.retry,
        ...(config.retry || {}),
      },
    };

    // Re-check Unix socket if path changed
    if (config.unixSocketPath) {
      try {
        this.unixSocketExists = fs.existsSync(this.config.unixSocketPath);
      } catch {
        this.unixSocketExists = false;
      }
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<MLXRConfig> {
    return { ...this.config };
  }

  /**
   * Check if Unix socket is available
   */
  isUnixSocketAvailable(): boolean {
    return this.unixSocketExists;
  }
}
