/**
 * OpenAI-compatible API types for MLXR
 * Based on OpenAI API specification
 */

/**
 * Chat message role types
 */
export type ChatRole = 'system' | 'user' | 'assistant' | 'function';

/**
 * Chat completion message
 */
export interface ChatMessage {
  role: ChatRole;
  content: string;
  name?: string;
  function_call?: string;
}

/**
 * Function definition for function calling
 */
export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>; // JSON schema
}

/**
 * Tool definition
 */
export interface ToolDefinition {
  type: 'function';
  function: FunctionDefinition;
}

/**
 * Chat completion request parameters
 */
export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  n?: number;
  user?: string;
  tools?: ToolDefinition[];
  tool_choice?: string;
  seed?: number;
}

/**
 * Usage statistics
 */
export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/**
 * Chat completion choice
 */
export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: 'stop' | 'length' | 'function_call' | 'content_filter' | null;
}

/**
 * Chat completion response
 */
export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: UsageInfo;
}

/**
 * Streaming chunk delta
 */
export interface ChatCompletionDelta {
  role?: string;
  content?: string;
  function_call?: string;
}

/**
 * Streaming chunk choice
 */
export interface ChatCompletionStreamChoice {
  index: number;
  delta: ChatCompletionDelta;
  finish_reason: string | null;
}

/**
 * Streaming chunk
 */
export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionStreamChoice[];
}

/**
 * Completion request parameters (non-chat)
 */
export interface CompletionRequest {
  model: string;
  prompt: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  n?: number;
  suffix?: string;
  seed?: number;
}

/**
 * Completion choice
 */
export interface CompletionChoice {
  index: number;
  text: string;
  finish_reason: string | null;
}

/**
 * Completion response
 */
export interface CompletionResponse {
  id: string;
  object: 'text_completion';
  created: number;
  model: string;
  choices: CompletionChoice[];
  usage: UsageInfo;
}

/**
 * Embedding request parameters
 */
export interface EmbeddingRequest {
  model: string;
  input: string | string[];
  encoding_format?: 'float' | 'base64';
  user?: string;
}

/**
 * Single embedding object
 */
export interface EmbeddingObject {
  index: number;
  embedding: number[];
  object: 'embedding';
}

/**
 * Embedding response
 */
export interface EmbeddingResponse {
  object: 'list';
  data: EmbeddingObject[];
  model: string;
  usage: UsageInfo;
}

/**
 * Model information
 */
export interface ModelInfo {
  id: string;
  object: 'model';
  created: number;
  owned_by: string;
}

/**
 * Model list response
 */
export interface ModelListResponse {
  object: 'list';
  data: ModelInfo[];
}

/**
 * Error detail
 */
export interface ErrorDetail {
  message: string;
  type: string;
  code?: string;
}

/**
 * Error response
 */
export interface ErrorResponse {
  error: ErrorDetail;
}
