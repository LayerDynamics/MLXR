/**
 * Ollama-compatible API client for MLXR
 */

import { HttpClient } from '../utils/http-client';
import { parseJSONStream } from '../utils/sse-parser';
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

    // Use shared JSON stream parser
    for await (const response of parseJSONStream<OllamaGenerateResponse>(stream)) {
      yield response;
      if (response.done) {
        break;
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

    // Use shared JSON stream parser
    for await (const response of parseJSONStream<OllamaChatResponse>(stream)) {
      yield response;
      if (response.done) {
        break;
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

    // Use shared JSON stream parser
    yield* parseJSONStream<OllamaPullResponse>(stream);
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

    // Use shared JSON stream parser
    yield* parseJSONStream<OllamaCreateResponse>(stream);
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
