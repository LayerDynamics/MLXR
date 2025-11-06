/**
 * Central type exports for MLXR frontend
 * Import all types from a single location for convenience
 */

// Backend types
export type {
  ModelInfo,
  AdapterInfo,
  SamplingParams,
  RequestInfo,
  SchedulerStats,
  KVCacheStats,
  SpeculativeDecodingStats,
  QueryOptions,
  ModelTag,
  RegistryStats,
} from './backend'

export {
  ModelFormat,
  ModelArchitecture,
  QuantizationType,
  RequestState,
  FinishReason,
  AdapterType,
} from './backend'

// OpenAI types
export type {
  ChatRole,
  ChatMessage,
  FunctionCall,
  ToolCall,
  FunctionDefinition,
  Tool,
  ResponseFormat,
  ChatCompletionRequest,
  ChatCompletionChoice,
  ChatCompletionLogprobs,
  TokenLogprob,
  TopLogprob,
  ChatCompletionResponse,
  ChatCompletionChunkDelta,
  ChatCompletionChunkChoice,
  ChatCompletionChunk,
  CompletionRequest,
  CompletionChoice,
  CompletionLogprobs,
  CompletionResponse,
  EmbeddingsRequest,
  Embedding,
  EmbeddingsResponse,
  UsageInfo,
  Model,
  ModelListResponse,
  ErrorResponse,
  ImageContent,
  TextContent,
  MessageContent,
  ChatMessageWithVision,
} from './openai'

// Ollama types
export type {
  OllamaGenerateRequest,
  OllamaOptions,
  OllamaGenerateResponse,
  OllamaChatMessage,
  OllamaChatRequest,
  OllamaChatResponse,
  OllamaEmbeddingsRequest,
  OllamaEmbeddingsResponse,
  OllamaModelInfo,
  OllamaTagsResponse,
  OllamaPullRequest,
  OllamaPullProgress,
  OllamaPushRequest,
  OllamaPushProgress,
  OllamaCreateRequest,
  OllamaCreateProgress,
  OllamaCopyRequest,
  OllamaDeleteRequest,
  OllamaShowRequest,
  OllamaShowResponse,
  OllamaListRunningResponse,
  OllamaErrorResponse,
  OllamaStreamChunk,
  OllamaVersion,
  OllamaModelfile,
} from './ollama'

// Metrics types
export type {
  Counter,
  Gauge,
  HistogramStats,
  Histogram,
  MetricsSnapshot,
  PrometheusMetric,
  PrometheusFormat,
  TimeSeriesPoint,
  TimeSeries,
  LatencyBucket,
  LatencyDistribution,
  KVBlockInfo,
  KVCacheHeatmap,
  KernelTiming,
  KernelTimings,
  RequestMetrics,
  ModelStatistics,
  MetricsFormat,
  MetricsExportOptions,
} from './metrics'

// Config types
export type {
  ServerConfig,
  ModelConfig,
  PathConfig,
  SystemPaths,
  UpdateConfig,
  PrivacyConfig,
  UIPreferences,
  KeyboardShortcutsConfig,
  AppConfig,
  ConfigValidationResult,
} from './config'

export {
  DEFAULT_SERVER_CONFIG,
  DEFAULT_UI_PREFERENCES,
  DEFAULT_KEYBOARD_SHORTCUTS,
} from './config'

// Bridge types
export type {
  HostBridge,
  BridgeMessage,
  BridgeResponse,
  BridgeError,
  SystemInfo,
  BridgeAvailability,
  BridgeEvent,
} from './bridge'

export { BridgeErrorCode, BridgeStatus } from './bridge'

// Store types
export type {
  DaemonStatus,
  Conversation,
  AppState,
  ChatUIState,
  ConversationsState,
  ModelsUIState,
  MetricsUIState,
  SettingsUIState,
  LogsUIState,
  AppNotification,
  KeyboardShortcutsState,
  CombinedStore,
  StorePersistConfig,
} from './store'

export { DEFAULT_SAMPLING_PARAMS } from './store'
