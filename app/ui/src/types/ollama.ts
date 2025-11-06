/**
 * Ollama API compatible types
 * Based on https://github.com/ollama/ollama/blob/main/docs/api.md
 */

export interface OllamaGenerateRequest {
  model: string
  prompt: string
  images?: string[] // Base64 encoded images
  format?: 'json'
  options?: OllamaOptions
  system?: string
  template?: string
  context?: number[]
  stream?: boolean
  raw?: boolean
  keep_alive?: string | number
}

export interface OllamaOptions {
  num_keep?: number
  seed?: number
  num_predict?: number
  top_k?: number
  top_p?: number
  tfs_z?: number
  typical_p?: number
  repeat_last_n?: number
  temperature?: number
  repeat_penalty?: number
  presence_penalty?: number
  frequency_penalty?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  penalize_newline?: boolean
  stop?: string[]
  numa?: boolean
  num_ctx?: number
  num_batch?: number
  num_gqa?: number
  num_gpu?: number
  main_gpu?: number
  low_vram?: boolean
  f16_kv?: boolean
  vocab_only?: boolean
  use_mmap?: boolean
  use_mlock?: boolean
  num_thread?: number
}

export interface OllamaGenerateResponse {
  model: string
  created_at: string
  response: string
  done: boolean
  context?: number[]
  total_duration?: number
  load_duration?: number
  prompt_eval_count?: number
  prompt_eval_duration?: number
  eval_count?: number
  eval_duration?: number
}

export interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
  images?: string[] // Base64 encoded images
}

export interface OllamaChatRequest {
  model: string
  messages: OllamaChatMessage[]
  format?: 'json'
  options?: OllamaOptions
  stream?: boolean
  keep_alive?: string | number
}

export interface OllamaChatResponse {
  model: string
  created_at: string
  message: OllamaChatMessage
  done: boolean
  total_duration?: number
  load_duration?: number
  prompt_eval_count?: number
  prompt_eval_duration?: number
  eval_count?: number
  eval_duration?: number
}

export interface OllamaEmbeddingsRequest {
  model: string
  prompt: string
  options?: OllamaOptions
  keep_alive?: string | number
}

export interface OllamaEmbeddingsResponse {
  embedding: number[]
}

export interface OllamaModelInfo {
  name: string
  modified_at: string
  size: number
  digest: string
  details: {
    format: string
    family: string
    families?: string[]
    parameter_size: string
    quantization_level: string
  }
}

export interface OllamaTagsResponse {
  models: OllamaModelInfo[]
}

export interface OllamaPullRequest {
  name: string
  insecure?: boolean
  stream?: boolean
}

export interface OllamaPullProgress {
  status: string
  digest?: string
  total?: number
  completed?: number
}

export interface OllamaPushRequest {
  name: string
  insecure?: boolean
  stream?: boolean
}

export interface OllamaPushProgress {
  status: string
  digest?: string
  total?: number
  completed?: number
}

export interface OllamaCreateRequest {
  name: string
  modelfile: string
  stream?: boolean
  path?: string
}

export interface OllamaCreateProgress {
  status: string
}

export interface OllamaCopyRequest {
  source: string
  destination: string
}

export interface OllamaDeleteRequest {
  name: string
}

export interface OllamaShowRequest {
  name: string
}

export interface OllamaShowResponse {
  modelfile: string
  parameters: string
  template: string
  details: {
    format: string
    family: string
    families?: string[]
    parameter_size: string
    quantization_level: string
  }
}

export interface OllamaListRunningResponse {
  models: Array<{
    name: string
    size: number
    digest: string
    details: {
      format: string
      family: string
      families?: string[]
      parameter_size: string
      quantization_level: string
    }
    expires_at: string
    size_vram: number
  }>
}

export interface OllamaErrorResponse {
  error: string
}

// Helper types for streaming
export type OllamaStreamChunk = OllamaGenerateResponse | OllamaChatResponse

export interface OllamaVersion {
  version: string
}

// Model file format
export interface OllamaModelfile {
  from: string
  parameter?: Record<string, string | number>
  template?: string
  system?: string
  adapter?: string
  license?: string
  message?: string
}
