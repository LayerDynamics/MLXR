/**
 * Ollama-compatible API client for MLXR
 */

import { HttpClient } from '../utils/http-client';
import type {
  MLXRConfig,
  OllamaGenerateRequest,
  OllamaGenerateResponse,
  OllamaChatRequest,
  OllamaChatResponse,
  OllamaEmbeddingsRequest,
  OllamaEmbeddingsResponse,
  OllamaPullRequest,
  OllamaPullResponse,
  OllamaCreateRequest,
  OllamaCreateResponse,
  OllamaTagsResponse,
  OllamaProcessResponse,
  OllamaShowRequest,
  OllamaShowResponse,
  OllamaDeleteRequest,
  OllamaCopyRequest,
} from '../types';

/**
 * Ollama-compatible client for MLXR
 */
export class OllamaClient {
  private httpClient: HttpClient;

  constructor(config?: MLXRConfig) {
    this.httpClient = new HttpClient(config);
  }

  /**
   * Generate a response
   */
  async generate(
    request: OllamaGenerateRequest
  ): Promise<OllamaGenerateResponse> {
    return this.httpClient.request<OllamaGenerateResponse>({
      method: 'POST',
      path: '/api/generate',
      body: request,
    });
  }

  /**
   * Stream a generated response
   */
  async *streamGenerate(
    request: OllamaGenerateRequest
  ): AsyncGenerator<OllamaGenerateResponse, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/api/generate',
      body: streamRequest,
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.done) {
        break;
      }
      try {
        const parsed = JSON.parse(chunk.data);
        yield parsed as OllamaGenerateResponse;
        if (parsed.done) {
          break;
        }
      } catch (err) {
        console.error('Failed to parse chunk:', err);
      }
    }
  }

  /**
   * Generate a chat response
   */
  async chat(request: OllamaChatRequest): Promise<OllamaChatResponse> {
    return this.httpClient.request<OllamaChatResponse>({
      method: 'POST',
      path: '/api/chat',
      body: request,
    });
  }

  /**
   * Stream a chat response
   */
  async *streamChat(
    request: OllamaChatRequest
  ): AsyncGenerator<OllamaChatResponse, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/api/chat',
      body: streamRequest,
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.done) {
        break;
      }
      try {
        const parsed = JSON.parse(chunk.data);
        yield parsed as OllamaChatResponse;
        if (parsed.done) {
          break;
        }
      } catch (err) {
        console.error('Failed to parse chunk:', err);
      }
    }
  }

  /**
   * Generate embeddings
   */
  async embeddings(
    request: OllamaEmbeddingsRequest
  ): Promise<OllamaEmbeddingsResponse> {
    return this.httpClient.request<OllamaEmbeddingsResponse>({
      method: 'POST',
      path: '/api/embeddings',
      body: request,
    });
  }

  /**
   * Pull a model from the registry
   */
  async pull(request: OllamaPullRequest): Promise<void> {
    await this.httpClient.request<void>({
      method: 'POST',
      path: '/api/pull',
      body: request,
    });
  }

  /**
   * Stream pull progress
   */
  async *streamPull(
    request: OllamaPullRequest
  ): AsyncGenerator<OllamaPullResponse, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/api/pull',
      body: streamRequest,
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.done) {
        break;
      }
      try {
        const parsed = JSON.parse(chunk.data);
        yield parsed as OllamaPullResponse;
      } catch (err) {
        console.error('Failed to parse chunk:', err);
      }
    }
  }

  /**
   * Create a model from a Modelfile
   */
  async create(request: OllamaCreateRequest): Promise<void> {
    await this.httpClient.request<void>({
      method: 'POST',
      path: '/api/create',
      body: request,
    });
  }

  /**
   * Stream model creation progress
   */
  async *streamCreate(
    request: OllamaCreateRequest
  ): AsyncGenerator<OllamaCreateResponse, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/api/create',
      body: streamRequest,
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.done) {
        break;
      }
      try {
        const parsed = JSON.parse(chunk.data);
        yield parsed as OllamaCreateResponse;
      } catch (err) {
        console.error('Failed to parse chunk:', err);
      }
    }
  }

  /**
   * List local models
   */
  async tags(): Promise<OllamaTagsResponse> {
    return this.httpClient.request<OllamaTagsResponse>({
      method: 'GET',
      path: '/api/tags',
    });
  }

  /**
   * List running models
   */
  async ps(): Promise<OllamaProcessResponse> {
    return this.httpClient.request<OllamaProcessResponse>({
      method: 'GET',
      path: '/api/ps',
    });
  }

  /**
   * Show model information
   */
  async show(request: OllamaShowRequest): Promise<OllamaShowResponse> {
    return this.httpClient.request<OllamaShowResponse>({
      method: 'POST',
      path: '/api/show',
      body: request,
    });
  }

  /**
   * Delete a model
   */
  async delete(request: OllamaDeleteRequest): Promise<void> {
    await this.httpClient.request<void>({
      method: 'DELETE',
      path: '/api/delete',
      body: request,
    });
  }

  /**
   * Copy a model
   */
  async copy(request: OllamaCopyRequest): Promise<void> {
    await this.httpClient.request<void>({
      method: 'POST',
      path: '/api/copy',
      body: request,
    });
  }

  /**
   * Update client configuration
   */
  updateConfig(config: Partial<MLXRConfig>): void {
    this.httpClient.updateConfig(config);
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<MLXRConfig> {
    return this.httpClient.getConfig();
  }

  /**
   * Check if Unix socket is available
   */
  isUnixSocketAvailable(): boolean {
    return this.httpClient.isUnixSocketAvailable();
  }
}
