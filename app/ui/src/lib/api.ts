/**
 * Complete API client for MLXR backend
 * Supports both OpenAI and Ollama compatible endpoints
 * All requests go through the WebView bridge
 */

import { bridge } from './bridge'
import type {
  // OpenAI types
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  CompletionRequest,
  CompletionResponse,
  EmbeddingsRequest,
  EmbeddingsResponse,
  ModelListResponse,
  // Ollama types
  OllamaGenerateRequest,
  OllamaGenerateResponse,
  OllamaChatRequest,
  OllamaChatResponse,
  OllamaEmbeddingsRequest,
  OllamaEmbeddingsResponse,
  OllamaTagsResponse,
  OllamaPullRequest,
  OllamaPullProgress,
  OllamaCreateRequest,
  OllamaCreateProgress,
  OllamaCopyRequest,
  OllamaDeleteRequest,
  OllamaShowRequest,
  OllamaShowResponse,
  OllamaListRunningResponse,
  // Backend types
  ModelInfo,
  RequestInfo,
  SchedulerStats,
  RegistryStats,
  QueryOptions,
  // Metrics types
  MetricsSnapshot,
} from '@/types'

/**
 * API error class with status code and details
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: unknown
  ) {
    super(message)
    this.name = 'APIError'
  }
}

/**
 * Parse JSON response with error handling
 */
async function parseJSONResponse<T>(response: {
  status: number
  body: unknown
}): Promise<T> {
  if (response.status >= 400) {
    const error =
      typeof response.body === 'object' && response.body !== null
        ? (response.body as { error?: { message?: string } })
        : {}
    throw new APIError(
      error.error?.message || `HTTP ${response.status}`,
      response.status,
      response.body
    )
  }
  return response.body as T
}

/**
 * OpenAI-compatible API client
 */
export const openai = {
  /**
   * Create chat completion
   * @param request - Chat completion request
   * @returns Chat completion response
   */
  async createChatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    const response = await bridge.request('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseJSONResponse<ChatCompletionResponse>(response)
  },

  /**
   * Stream chat completion
   * @param request - Chat completion request with stream: true
   * @returns Async generator of chat completion chunks
   */
  async *streamChatCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<ChatCompletionChunk, void, unknown> {
    const response = await bridge.request('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }

    // In production, the daemon returns the stream directly
    // For now, we'll parse the chunks from the response
    const stream = response.body as ReadableStream
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6)
            if (data === '[DONE]') return
            try {
              yield JSON.parse(data) as ChatCompletionChunk
            } catch (err) {
              console.warn('Failed to parse SSE chunk:', data, err)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  /**
   * Create completion
   * @param request - Completion request
   * @returns Completion response
   */
  async createCompletion(
    request: CompletionRequest
  ): Promise<CompletionResponse> {
    const response = await bridge.request('/v1/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseJSONResponse<CompletionResponse>(response)
  },

  /**
   * Create embeddings
   * @param request - Embeddings request
   * @returns Embeddings response
   */
  async createEmbeddings(
    request: EmbeddingsRequest
  ): Promise<EmbeddingsResponse> {
    const response = await bridge.request('/v1/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseJSONResponse<EmbeddingsResponse>(response)
  },

  /**
   * List available models
   * @returns Model list response
   */
  async listModels(): Promise<ModelListResponse> {
    const response = await bridge.request('/v1/models', { method: 'GET' })
    return parseJSONResponse<ModelListResponse>(response)
  },
}

/**
 * Ollama-compatible API client
 */
export const ollama = {
  /**
   * Generate completion
   * @param request - Generate request
   * @returns Generate response
   */
  async generate(
    request: OllamaGenerateRequest
  ): Promise<OllamaGenerateResponse> {
    const response = await bridge.request('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: false }),
    })
    return parseJSONResponse<OllamaGenerateResponse>(response)
  },

  /**
   * Stream generate completion
   * @param request - Generate request
   * @returns Async generator of generate responses
   */
  async *streamGenerate(
    request: OllamaGenerateRequest
  ): AsyncGenerator<OllamaGenerateResponse, void, unknown> {
    const response = await bridge.request('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }

    const stream = response.body as ReadableStream
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed) {
            try {
              const chunk = JSON.parse(trimmed) as OllamaGenerateResponse
              yield chunk
              if (chunk.done) return
            } catch (err) {
              console.warn('Failed to parse Ollama chunk:', trimmed, err)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  /**
   * Chat completion
   * @param request - Chat request
   * @returns Chat response
   */
  async chat(request: OllamaChatRequest): Promise<OllamaChatResponse> {
    const response = await bridge.request('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: false }),
    })
    return parseJSONResponse<OllamaChatResponse>(response)
  },

  /**
   * Stream chat completion
   * @param request - Chat request
   * @returns Async generator of chat responses
   */
  async *streamChat(
    request: OllamaChatRequest
  ): AsyncGenerator<OllamaChatResponse, void, unknown> {
    const response = await bridge.request('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }

    const stream = response.body as ReadableStream
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed) {
            try {
              const chunk = JSON.parse(trimmed) as OllamaChatResponse
              yield chunk
              if (chunk.done) return
            } catch (err) {
              console.warn('Failed to parse Ollama chat chunk:', trimmed, err)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  /**
   * Create embeddings
   * @param request - Embeddings request
   * @returns Embeddings response
   */
  async embeddings(
    request: OllamaEmbeddingsRequest
  ): Promise<OllamaEmbeddingsResponse> {
    const response = await bridge.request('/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseJSONResponse<OllamaEmbeddingsResponse>(response)
  },

  /**
   * List available models (tags)
   * @returns Tags response
   */
  async tags(): Promise<OllamaTagsResponse> {
    const response = await bridge.request('/api/tags', { method: 'GET' })
    return parseJSONResponse<OllamaTagsResponse>(response)
  },

  /**
   * Pull model from registry
   * @param request - Pull request
   * @returns Async generator of pull progress
   */
  async *pull(
    request: OllamaPullRequest
  ): AsyncGenerator<OllamaPullProgress, void, unknown> {
    const response = await bridge.request('/api/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })

    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }

    const stream = response.body as ReadableStream
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed) {
            try {
              yield JSON.parse(trimmed) as OllamaPullProgress
            } catch (err) {
              console.warn('Failed to parse pull progress:', trimmed, err)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  /**
   * Create model from Modelfile
   * @param request - Create request
   * @returns Async generator of create progress
   */
  async *create(
    request: OllamaCreateRequest
  ): AsyncGenerator<OllamaCreateProgress, void, unknown> {
    const response = await bridge.request('/api/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })

    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }

    const stream = response.body as ReadableStream
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed) {
            try {
              yield JSON.parse(trimmed) as OllamaCreateProgress
            } catch (err) {
              console.warn('Failed to parse create progress:', trimmed, err)
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  /**
   * Copy model
   * @param request - Copy request
   */
  async copy(request: OllamaCopyRequest): Promise<void> {
    const response = await bridge.request('/api/copy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }
  },

  /**
   * Delete model
   * @param request - Delete request
   */
  async delete(request: OllamaDeleteRequest): Promise<void> {
    const response = await bridge.request('/api/delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }
  },

  /**
   * Show model information
   * @param request - Show request
   * @returns Show response
   */
  async show(request: OllamaShowRequest): Promise<OllamaShowResponse> {
    const response = await bridge.request('/api/show', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseJSONResponse<OllamaShowResponse>(response)
  },

  /**
   * List running models
   * @returns List running response
   */
  async ps(): Promise<OllamaListRunningResponse> {
    const response = await bridge.request('/api/ps', { method: 'GET' })
    return parseJSONResponse<OllamaListRunningResponse>(response)
  },
}

/**
 * MLXR native backend API
 */
export const backend = {
  /**
   * List models in registry
   * @param options - Query options
   * @returns Array of model info
   */
  async listModels(options?: QueryOptions): Promise<ModelInfo[]> {
    const params = new URLSearchParams()
    if (options?.architecture) params.set('architecture', options.architecture)
    if (options?.format) params.set('format', options.format)
    if (options?.quant_type) params.set('quant_type', options.quant_type)
    if (options?.is_loaded !== undefined)
      params.set('is_loaded', options.is_loaded.toString())
    if (options?.limit !== undefined)
      params.set('limit', options.limit.toString())
    if (options?.offset !== undefined)
      params.set('offset', options.offset.toString())
    if (options?.sort_by) params.set('sort_by', options.sort_by)
    if (options?.sort_order) params.set('sort_order', options.sort_order)

    const query = params.toString()
    const path = query ? `/v1/models?${query}` : '/v1/models'

    const response = await bridge.request(path, { method: 'GET' })
    const data = await parseJSONResponse<{ data: ModelInfo[] }>(response)
    return data.data
  },

  /**
   * Get model by ID
   * @param id - Model ID
   * @returns Model info
   */
  async getModel(id: string): Promise<ModelInfo> {
    const response = await bridge.request(`/v1/models/${id}`, { method: 'GET' })
    return parseJSONResponse<ModelInfo>(response)
  },

  /**
   * Delete model
   * @param id - Model ID
   */
  async deleteModel(id: string): Promise<void> {
    const response = await bridge.request(`/v1/models/${id}`, {
      method: 'DELETE',
    })
    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }
  },

  /**
   * Get scheduler stats
   * @returns Scheduler stats
   */
  async getSchedulerStats(): Promise<SchedulerStats> {
    const response = await bridge.request('/v1/stats/scheduler', {
      method: 'GET',
    })
    return parseJSONResponse<SchedulerStats>(response)
  },

  /**
   * Get registry stats
   * @returns Registry stats
   */
  async getRegistryStats(): Promise<RegistryStats> {
    const response = await bridge.request('/v1/stats/registry', {
      method: 'GET',
    })
    return parseJSONResponse<RegistryStats>(response)
  },

  /**
   * List active requests
   * @returns Array of request info
   */
  async listRequests(): Promise<RequestInfo[]> {
    const response = await bridge.request('/v1/requests', { method: 'GET' })
    return parseJSONResponse<RequestInfo[]>(response)
  },

  /**
   * Cancel request
   * @param id - Request ID
   */
  async cancelRequest(id: string): Promise<void> {
    const response = await bridge.request(`/v1/requests/${id}/cancel`, {
      method: 'POST',
    })
    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }
  },

  /**
   * Get Prometheus metrics
   * @returns Prometheus-formatted metrics
   */
  async getMetrics(): Promise<string> {
    const response = await bridge.request('/metrics', { method: 'GET' })
    if (response.status >= 400) {
      throw new APIError(`HTTP ${response.status}`, response.status, response.body)
    }
    return response.body as string
  },

  /**
   * Get metrics snapshot (JSON format)
   * @returns Metrics snapshot
   */
  async getMetricsSnapshot(): Promise<MetricsSnapshot> {
    const response = await bridge.request('/metrics/json', { method: 'GET' })
    return parseJSONResponse<MetricsSnapshot>(response)
  },

  /**
   * Health check
   * @returns Health status
   */
  async health(): Promise<{ status: string; version: string }> {
    const response = await bridge.request('/health', { method: 'GET' })
    return parseJSONResponse<{ status: string; version: string }>(response)
  },
}

/**
 * Combined API client with all endpoints
 */
export const api = {
  openai,
  ollama,
  backend,
}

export default api
