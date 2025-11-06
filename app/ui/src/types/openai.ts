/**
 * OpenAI API compatible types
 * Based on https://platform.openai.com/docs/api-reference
 */

export type ChatRole = 'system' | 'user' | 'assistant' | 'function' | 'tool'

export interface ChatMessage {
  role: ChatRole
  content: string | null
  name?: string
  function_call?: FunctionCall
  tool_calls?: ToolCall[]
}

export interface FunctionCall {
  name: string
  arguments: string // JSON string
}

export interface ToolCall {
  id: string
  type: 'function'
  function: FunctionCall
}

export interface FunctionDefinition {
  name: string
  description?: string
  parameters?: Record<string, unknown> // JSON Schema
}

export interface Tool {
  type: 'function'
  function: FunctionDefinition
}

export type ResponseFormat = 'text' | 'json_object'

export interface ChatCompletionRequest {
  model: string
  messages: ChatMessage[]
  temperature?: number
  top_p?: number
  n?: number
  stream?: boolean
  stop?: string | string[]
  max_tokens?: number
  presence_penalty?: number
  frequency_penalty?: number
  logit_bias?: Record<string, number>
  user?: string
  functions?: FunctionDefinition[]
  function_call?: 'none' | 'auto' | { name: string }
  tools?: Tool[]
  tool_choice?: 'none' | 'auto' | { type: 'function'; function: { name: string } }
  response_format?: { type: ResponseFormat }
  seed?: number
}

export interface ChatCompletionChoice {
  index: number
  message: ChatMessage
  finish_reason: 'stop' | 'length' | 'function_call' | 'tool_calls' | 'content_filter' | null
  logprobs?: ChatCompletionLogprobs | null
}

export interface ChatCompletionLogprobs {
  content: TokenLogprob[] | null
}

export interface TokenLogprob {
  token: string
  logprob: number
  bytes: number[] | null
  top_logprobs: TopLogprob[]
}

export interface TopLogprob {
  token: string
  logprob: number
  bytes: number[] | null
}

export interface ChatCompletionResponse {
  id: string
  object: 'chat.completion'
  created: number
  model: string
  choices: ChatCompletionChoice[]
  usage: UsageInfo
  system_fingerprint?: string
}

export interface ChatCompletionChunkDelta {
  role?: ChatRole
  content?: string
  function_call?: Partial<FunctionCall>
  tool_calls?: Partial<ToolCall>[]
}

export interface ChatCompletionChunkChoice {
  index: number
  delta: ChatCompletionChunkDelta
  finish_reason: 'stop' | 'length' | 'function_call' | 'tool_calls' | 'content_filter' | null
  logprobs?: ChatCompletionLogprobs | null
}

export interface ChatCompletionChunk {
  id: string
  object: 'chat.completion.chunk'
  created: number
  model: string
  choices: ChatCompletionChunkChoice[]
  system_fingerprint?: string
}

export interface CompletionRequest {
  model: string
  prompt: string | string[]
  suffix?: string
  max_tokens?: number
  temperature?: number
  top_p?: number
  n?: number
  stream?: boolean
  logprobs?: number
  echo?: boolean
  stop?: string | string[]
  presence_penalty?: number
  frequency_penalty?: number
  best_of?: number
  logit_bias?: Record<string, number>
  user?: string
}

export interface CompletionChoice {
  text: string
  index: number
  logprobs: CompletionLogprobs | null
  finish_reason: 'stop' | 'length' | 'content_filter' | null
}

export interface CompletionLogprobs {
  tokens: string[]
  token_logprobs: number[]
  top_logprobs: Record<string, number>[]
  text_offset: number[]
}

export interface CompletionResponse {
  id: string
  object: 'text_completion'
  created: number
  model: string
  choices: CompletionChoice[]
  usage: UsageInfo
}

export interface EmbeddingsRequest {
  model: string
  input: string | string[]
  user?: string
  encoding_format?: 'float' | 'base64'
}

export interface Embedding {
  object: 'embedding'
  embedding: number[]
  index: number
}

export interface EmbeddingsResponse {
  object: 'list'
  data: Embedding[]
  model: string
  usage: UsageInfo
}

export interface UsageInfo {
  prompt_tokens: number
  completion_tokens?: number
  total_tokens: number
}

export interface Model {
  id: string
  object: 'model'
  created: number
  owned_by: string
}

export interface ModelListResponse {
  object: 'list'
  data: Model[]
}

export interface ErrorResponse {
  error: {
    message: string
    type: string
    param: string | null
    code: string | null
  }
}

// Vision support types
export interface ImageContent {
  type: 'image_url'
  image_url: {
    url: string // Can be URL or base64 data URI
    detail?: 'auto' | 'low' | 'high'
  }
}

export interface TextContent {
  type: 'text'
  text: string
}

export type MessageContent = string | (TextContent | ImageContent)[]

export interface ChatMessageWithVision {
  role: ChatRole
  content: MessageContent
  name?: string
}
