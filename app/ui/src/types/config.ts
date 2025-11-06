/**
 * Server configuration types
 * Matches configs/server.yaml structure
 */

export interface ServerConfig {
  // Server settings
  server: {
    uds_path?: string
    http_port?: number
    enable_http?: boolean
    bind_address?: string
  }

  // Scheduler settings
  scheduler: {
    max_batch_tokens: number
    max_batch_size: number
    max_prefill_tokens: number
    total_kv_blocks: number
    kv_block_size: number
    max_prefill_chunk_size: number
    enable_chunked_prefill: boolean
    enable_priority_scheduling: boolean
    decode_preference: number
    enable_preemption: boolean
    min_decode_steps_before_preempt: number
  }

  // Performance settings
  performance: {
    target_latency_ms: number
    enable_speculative: boolean
    draft_model?: string
    speculation_length: number
    kv_persistence: boolean
  }

  // Model settings
  models: {
    default_model?: string
    models_dir: string
    cache_dir: string
    auto_load_default: boolean
  }

  // Memory settings
  memory: {
    gpu_memory_fraction: number
    enable_unified_memory: boolean
    pin_host_memory: boolean
  }

  // Logging settings
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error'
    log_file?: string
    enable_structured_logging: boolean
  }

  // Telemetry settings
  telemetry: {
    enabled: boolean
    collect_model_names: boolean
    collect_prompts: boolean
    metrics_port?: number
  }
}

export interface ModelConfig {
  model_id: string
  name: string
  file_path: string

  // Tokenizer configuration
  tokenizer: {
    type: 'sentencepiece' | 'huggingface' | 'tiktoken'
    path: string
    vocab_size?: number
  }

  // Context configuration
  context: {
    max_length: number
    sliding_window?: number
  }

  // RoPE scaling configuration
  rope: {
    freq_base: number
    scale: number
    scaling_type?: 'linear' | 'ntk' | 'yarn'
    ntk_alpha?: number
    yarn_alpha?: number
    yarn_beta?: number
  }

  // Quantization configuration
  quantization?: {
    enabled: boolean
    type: string
    group_size?: number
  }

  // Adapter configuration
  adapters?: Array<{
    name: string
    path: string
    type: 'lora' | 'qlora' | 'ia3'
    scale: number
  }>

  // Chat template
  chat_template?: string

  // Sampling defaults
  sampling_defaults?: {
    temperature: number
    top_p: number
    top_k: number
    repetition_penalty: number
    max_tokens: number
  }
}

export interface PathConfig {
  models_dir: string
  cache_dir: string
  logs_dir: string
  config_dir: string
  run_dir: string
}

export interface SystemPaths {
  application_support: string
  logs: string
  cache: string
  tmp: string
}

// Update settings (Sparkle configuration)
export interface UpdateConfig {
  enabled: boolean
  check_on_launch: boolean
  auto_download: boolean
  auto_install: boolean
  update_channel: 'stable' | 'beta' | 'dev'
  appcast_url: string
}

// Privacy settings
export interface PrivacyConfig {
  telemetry_enabled: boolean
  collect_crash_reports: boolean
  collect_usage_statistics: boolean
  share_model_names: boolean
  share_prompts: boolean
}

// UI preferences
export interface UIPreferences {
  theme: 'light' | 'dark' | 'system'
  compact_mode: boolean
  show_token_count: boolean
  show_latency: boolean
  enable_animations: boolean
  font_size: 'small' | 'medium' | 'large'
  code_theme: 'github-light' | 'github-dark' | 'monokai' | 'solarized'
}

// Keyboard shortcuts configuration
export interface KeyboardShortcutsConfig {
  command_palette: string
  new_conversation: string
  import_model: string
  toggle_logs: string
  toggle_sidebar: string
  send_message: string
  cancel_generation: string
  focus_input: string
}

// Complete application configuration
export interface AppConfig {
  server: ServerConfig
  paths: PathConfig
  updates: UpdateConfig
  privacy: PrivacyConfig
  ui: UIPreferences
  keyboard_shortcuts: KeyboardShortcutsConfig
}

// Configuration validation result
export interface ConfigValidationResult {
  valid: boolean
  errors: Array<{
    path: string
    message: string
  }>
  warnings: Array<{
    path: string
    message: string
  }>
}

// Default configurations
export const DEFAULT_SERVER_CONFIG: ServerConfig = {
  server: {
    uds_path: '~/Library/Application Support/MLXRunner/run/mlxrunner.sock',
    enable_http: false,
    http_port: 8080,
    bind_address: '127.0.0.1',
  },
  scheduler: {
    max_batch_tokens: 8192,
    max_batch_size: 128,
    max_prefill_tokens: 4096,
    total_kv_blocks: 1024,
    kv_block_size: 16,
    max_prefill_chunk_size: 2048,
    enable_chunked_prefill: true,
    enable_priority_scheduling: true,
    decode_preference: 2.0,
    enable_preemption: true,
    min_decode_steps_before_preempt: 10,
  },
  performance: {
    target_latency_ms: 50,
    enable_speculative: true,
    speculation_length: 4,
    kv_persistence: true,
  },
  models: {
    models_dir: '~/Library/Application Support/MLXRunner/models',
    cache_dir: '~/Library/Application Support/MLXRunner/cache',
    auto_load_default: true,
  },
  memory: {
    gpu_memory_fraction: 0.9,
    enable_unified_memory: true,
    pin_host_memory: true,
  },
  logging: {
    level: 'info',
    log_file: '~/Library/Logs/mlxrunnerd.log',
    enable_structured_logging: true,
  },
  telemetry: {
    enabled: false,
    collect_model_names: false,
    collect_prompts: false,
  },
}

export const DEFAULT_UI_PREFERENCES: UIPreferences = {
  theme: 'system',
  compact_mode: false,
  show_token_count: true,
  show_latency: true,
  enable_animations: true,
  font_size: 'medium',
  code_theme: 'github-dark',
}

export const DEFAULT_KEYBOARD_SHORTCUTS: KeyboardShortcutsConfig = {
  command_palette: 'mod+k',
  new_conversation: 'mod+n',
  import_model: 'mod+i',
  toggle_logs: 'mod+/',
  toggle_sidebar: 'mod+b',
  send_message: 'mod+enter',
  cancel_generation: 'escape',
  focus_input: 'mod+l',
}
