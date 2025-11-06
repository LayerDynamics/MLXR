/**
 * OpenAI-compatible API client for MLXR
 */

import { HttpClient } from '../utils/http-client';
import { parseJSONStream } from '../utils/sse-parser';
import type {
  MLXRConfig,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  CompletionRequest,
  CompletionResponse,
  EmbeddingRequest,
  EmbeddingResponse,
  ModelListResponse,
  ModelInfo,
} from '../types';

/**
 * OpenAI-compatible client for MLXR
 */
export class OpenAIClient {
  private httpClient: HttpClient;

  constructor(config?: MLXRConfig) {
    this.httpClient = new HttpClient(config);
  }

  /**
   * Create a chat completion
   */
  async createChatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    return this.httpClient.request<ChatCompletionResponse>({
      method: 'POST',
      path: '/v1/chat/completions',
      body: request,
    });
  }

  /**
   * Create a streaming chat completion
   */
  async *streamChatCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<ChatCompletionChunk, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/v1/chat/completions',
      body: streamRequest,
      stream: true,
    });

    // Use shared JSON stream parser
    yield* parseJSONStream<ChatCompletionChunk>(stream);
  }

  /**
   * Create a text completion
   */
  async createCompletion(
    request: CompletionRequest
  ): Promise<CompletionResponse> {
    return this.httpClient.request<CompletionResponse>({
      method: 'POST',
      path: '/v1/completions',
      body: request,
    });
  }

  /**
   * Create a streaming text completion
   */
  async *streamCompletion(
    request: CompletionRequest
  ): AsyncGenerator<CompletionResponse, void, unknown> {
    const streamRequest = { ...request, stream: true };

    const stream = await this.httpClient.requestStream({
      method: 'POST',
      path: '/v1/completions',
      body: streamRequest,
      stream: true,
    });

    // Use shared JSON stream parser
    yield* parseJSONStream<CompletionResponse>(stream);
  }

  /**
   * Create embeddings
   */
  async createEmbedding(
    request: EmbeddingRequest
  ): Promise<EmbeddingResponse> {
    return this.httpClient.request<EmbeddingResponse>({
      method: 'POST',
      path: '/v1/embeddings',
      body: request,
    });
  }

  /**
   * List available models
   */
  async listModels(): Promise<ModelListResponse> {
    return this.httpClient.request<ModelListResponse>({
      method: 'GET',
      path: '/v1/models',
    });
  }

  /**
   * Get model information
   */
  async getModel(modelId: string): Promise<ModelInfo> {
    return this.httpClient.request<ModelInfo>({
      method: 'GET',
      path: `/v1/models/${modelId}`,
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
