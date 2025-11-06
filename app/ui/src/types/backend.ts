/**
 * Backend types mirroring C++ structures from daemon/
 * These types ensure type safety between frontend and backend
 */

// Enums matching C++ enums

export enum ModelFormat {
  GGUF = 'gguf',
  SAFETENSORS = 'safetensors',
  MLX_NATIVE = 'mlx',
}

export enum ModelArchitecture {
  LLAMA = 'llama',
  MISTRAL = 'mistral',
  MIXTRAL = 'mixtral',
  GEMMA = 'gemma',
  PHI = 'phi',
  QWEN = 'qwen',
  UNKNOWN = 'unknown',
}

export enum QuantizationType {
  // GGML base quantizations
  Q4_0 = 'Q4_0',
  Q4_1 = 'Q4_1',
  Q5_0 = 'Q5_0',
  Q5_1 = 'Q5_1',
  Q8_0 = 'Q8_0',
  Q8_1 = 'Q8_1',
  // K-quants
  Q2_K = 'Q2_K',
  Q3_K = 'Q3_K',
  Q4_K = 'Q4_K',
  Q5_K = 'Q5_K',
  Q6_K = 'Q6_K',
  Q8_K = 'Q8_K',
  // IQ variants
  IQ2_XXS = 'IQ2_XXS',
  IQ2_XS = 'IQ2_XS',
  IQ3_XXS = 'IQ3_XXS',
  IQ1_S = 'IQ1_S',
  IQ4_NL = 'IQ4_NL',
  IQ3_S = 'IQ3_S',
  IQ2_S = 'IQ2_S',
  IQ4_XS = 'IQ4_XS',
  IQ1_M = 'IQ1_M',
  // Floating point
  F16 = 'F16',
  F32 = 'F32',
  NONE = 'NONE',
}

export enum RequestState {
  WAITING = 'waiting',
  PREFILLING = 'prefilling',
  DECODING = 'decoding',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
  FAILED = 'failed',
}

export enum FinishReason {
  STOP = 'stop',
  LENGTH = 'length',
  EOS = 'eos',
  CANCELLED = 'cancelled',
  ERROR = 'error',
  NONE = 'none',
}

export enum AdapterType {
  LORA = 'lora',
  QLORA = 'qlora',
  IA3 = 'ia3',
}

// Interfaces matching C++ structures

export interface ModelInfo {
  id: number
  name: string
  model_id: string
  architecture: ModelArchitecture
  file_path: string
  format: ModelFormat
  file_size: number
  sha256: string
  param_count: number
  context_length: number
  hidden_size: number
  num_layers: number
  num_heads: number
  num_kv_heads: number
  intermediate_size: number
  vocab_size: number
  quant_type: QuantizationType
  quant_details: string
  tokenizer_type: string
  tokenizer_path: string
  rope_freq_base: number
  rope_scale: number
  rope_scaling_type: string
  description: string
  license: string
  source_url: string
  tags: string[]
  is_loaded: boolean
  last_used_timestamp: number
  created_timestamp: number
  chat_template: string
}

export interface AdapterInfo {
  id: number
  base_model_id: number
  name: string
  adapter_id: string
  file_path: string
  adapter_type: AdapterType
  rank: number
  scale: number
  target_modules: string
  created_timestamp: number
}

export interface SamplingParams {
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  max_tokens: number
  stop_token_ids: number[]
  logprobs: boolean
  top_logprobs: number
}

export interface RequestInfo {
  request_id: string
  prompt: string
  prompt_token_ids: number[]
  sampling_params: SamplingParams
  state: RequestState
  finish_reason: FinishReason
  generated_token_ids: number[]
  num_prompt_tokens: number
  num_generated_tokens: number
  created_time_ms: number
  first_token_time_ms: number
  last_token_time_ms: number
  kv_block_ids: number[]
}

export interface SchedulerStats {
  waiting_requests: number
  prefilling_requests: number
  decoding_requests: number
  paused_requests: number
  used_kv_blocks: number
  available_kv_blocks: number
  kv_utilization: number
  tokens_per_second: number
  requests_per_second: number
  avg_queue_time_ms: number
  avg_prefill_time_ms: number
  avg_decode_latency_ms: number
  total_requests_completed: number
  total_tokens_generated: number
}

export interface KVCacheStats {
  total_blocks: number
  used_blocks: number
  free_blocks: number
  block_size: number
  evictions: number
  hit_rate: number
  gpu_blocks: number
  cpu_blocks: number
}

export interface SpeculativeDecodingStats {
  enabled: boolean
  draft_model: string
  tokens_proposed: number
  tokens_accepted: number
  acceptance_rate: number
  avg_speculation_length: number
}

// Query options for listing models
export interface QueryOptions {
  architecture?: ModelArchitecture
  format?: ModelFormat
  quant_type?: QuantizationType
  is_loaded?: boolean
  limit?: number
  offset?: number
  sort_by?: 'name' | 'created_timestamp' | 'last_used_timestamp' | 'param_count'
  sort_order?: 'asc' | 'desc'
}

// Model tag structure
export interface ModelTag {
  key: string
  value: string
}

// Model registry statistics
export interface RegistryStats {
  total_models: number
  total_adapters: number
  total_disk_usage: number
  models_by_format: Record<ModelFormat, number>
  models_by_architecture: Record<ModelArchitecture, number>
  models_by_quant: Record<QuantizationType, number>
}

// Note: MetricsSnapshot is defined in types/metrics.ts
