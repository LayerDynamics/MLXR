/**
 * HTTP client with Unix domain socket and SSE streaming support
 */

import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import type { MLXRConfig } from '../types';

export interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
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
 * HTTP client for MLXR API requests
 */
export class HttpClient {
  private config: Required<MLXRConfig>;
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
   * Make a request to the MLXR API
   */
  async request<T>(options: RequestOptions): Promise<T> {
    if (options.stream) {
      throw new Error('Use requestStream for streaming requests');
    }

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

      if (useUnixSocket) {
        requestOptions.socketPath = this.config.unixSocketPath;
      } else {
        const url = new URL(this.config.baseUrl);
        requestOptions.hostname = url.hostname;
        requestOptions.port = url.port;
        requestOptions.protocol = url.protocol;
      }

      const req = http.request(requestOptions, (res) => {
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
            try {
              const error = JSON.parse(data);
              reject(
                new Error(
                  error.error?.message || `HTTP ${res.statusCode}: ${data}`
                )
              );
            } catch {
              reject(new Error(`HTTP ${res.statusCode}: ${data}`));
            }
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
    if (options.body) {
      body = JSON.stringify(options.body);
      headers['Content-Length'] = Buffer.byteLength(body).toString();
    }

    const requestOptions: http.RequestOptions = {
      method: options.method,
      path: options.path,
      headers,
      timeout: this.config.timeout,
    };

    if (useUnixSocket) {
      requestOptions.socketPath = this.config.unixSocketPath;
    } else {
      const url = new URL(this.config.baseUrl);
      requestOptions.hostname = url.hostname;
      requestOptions.port = url.port;
      requestOptions.protocol = url.protocol;
    }

    const generator = await new Promise<AsyncGenerator<StreamChunk, void, unknown>>(
      (resolve, reject) => {
        const req = http.request(requestOptions, (res) => {
          if (res.statusCode && (res.statusCode < 200 || res.statusCode >= 300)) {
            let errorData = '';
            res.on('data', (chunk) => {
              errorData += chunk;
            });
            res.on('end', () => {
              try {
                const error = JSON.parse(errorData);
                reject(
                  new Error(
                    error.error?.message || `HTTP ${res.statusCode}: ${errorData}`
                  )
                );
              } catch {
                reject(new Error(`HTTP ${res.statusCode}: ${errorData}`));
              }
            });
            return;
          }

          async function* streamGenerator() {
            let buffer = '';

            for await (const chunk of res) {
              buffer += chunk.toString();
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(6).trim();
                  if (data === '[DONE]') {
                    yield { data: '', done: true };
                    return;
                  }
                  if (data) {
                    yield { data, done: false };
                  }
                }
              }
            }

            // Process any remaining data
            if (buffer.startsWith('data: ')) {
              const data = buffer.slice(6).trim();
              if (data && data !== '[DONE]') {
                yield { data, done: false };
              }
            }
          }

          resolve(streamGenerator());
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
      }
    );

    yield* generator;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<MLXRConfig>): void {
    this.config = { ...this.config, ...config };

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
