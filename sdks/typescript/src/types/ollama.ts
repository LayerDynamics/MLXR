/**
 * Ollama-compatible API types for MLXR
 * Based on Ollama API specification
 */

/**
 * Generate request parameters
 */
export interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  system?: string;
  template?: string;
  context?: string;
  stream?: boolean;
  raw?: boolean;
  format?: 'json';
  // Model parameters
  num_predict?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  seed?: number;
  stop?: string[];
}

/**
 * Generate response (non-streaming)
 */
export interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

/**
 * Chat message
 */
export interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  images?: string[]; // Base64 encoded
}

/**
 * Chat request parameters
 */
export interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  stream?: boolean;
  format?: 'json';
  // Model parameters
  num_predict?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  seed?: number;
  stop?: string[];
}

/**
 * Chat response (non-streaming)
 */
export interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: OllamaChatMessage;
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

/**
 * Embeddings request parameters
 */
export interface OllamaEmbeddingsRequest {
  model: string;
  prompt: string;
}

/**
 * Embeddings response
 */
export interface OllamaEmbeddingsResponse {
  embedding: number[];
}

/**
 * Pull request parameters
 */
export interface OllamaPullRequest {
  name: string;
  insecure?: boolean;
  stream?: boolean;
}

/**
 * Pull response (streaming)
 */
export interface OllamaPullResponse {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
}

/**
 * Create request parameters
 */
export interface OllamaCreateRequest {
  name: string;
  modelfile?: string;
  path?: string;
  stream?: boolean;
}

/**
 * Create response (streaming)
 */
export interface OllamaCreateResponse {
  status: string;
}

/**
 * Model details
 */
export interface OllamaModelDetails {
  format: string;
  family: string;
  families: string[];
  parameter_size: string;
  quantization_level: string;
}

/**
 * Model information
 */
export interface OllamaModelInfo {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details?: OllamaModelDetails;
}

/**
 * Tags (model list) response
 */
export interface OllamaTagsResponse {
  models: OllamaModelInfo[];
}

/**
 * Running model information
 */
export interface OllamaRunningModel {
  name: string;
  model: string;
  size: number;
  digest: string;
  details?: OllamaModelDetails;
  expires_at?: string;
  size_vram?: number;
}

/**
 * Process (running models) response
 */
export interface OllamaProcessResponse {
  models: OllamaRunningModel[];
}

/**
 * Show request parameters
 */
export interface OllamaShowRequest {
  name: string;
}

/**
 * Show response (detailed model info)
 */
export interface OllamaShowResponse {
  modelfile: string;
  parameters: string;
  template: string;
  details: OllamaModelDetails;
}

/**
 * Delete request parameters
 */
export interface OllamaDeleteRequest {
  name: string;
}

/**
 * Copy request parameters
 */
export interface OllamaCopyRequest {
  source: string;
  destination: string;
}
