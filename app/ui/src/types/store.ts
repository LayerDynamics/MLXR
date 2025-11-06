/**
 * Zustand store types for client-side state management
 * Note: Server state is managed by TanStack Query, not Zustand
 */

import type { UIPreferences } from './config'
import type { ChatMessage } from './openai'

// Daemon status
export type DaemonStatus = 'starting' | 'running' | 'stopped' | 'error'

// Stored chat message (extends OpenAI ChatMessage with ID for local tracking)
export interface StoredChatMessage extends ChatMessage {
  id: string
}

// Conversation structure
export interface Conversation {
  id: string
  title: string
  model_id: string
  messages: StoredChatMessage[]
  created_at: number
  updated_at: number
  metadata?: {
    total_tokens?: number
    total_cost?: number
    [key: string]: unknown
  }
}

// App-wide state
export interface AppState {
  // Daemon status
  daemonStatus: DaemonStatus
  setDaemonStatus: (status: DaemonStatus) => void

  // Active selections
  activeModelId: string | null
  activeConversationId: string | null
  setActiveModel: (modelId: string | null) => void
  setActiveConversation: (conversationId: string | null) => void

  // UI state
  sidebarOpen: boolean
  commandPaletteOpen: boolean
  setSidebarOpen: (open: boolean) => void
  setCommandPaletteOpen: (open: boolean) => void
  toggleSidebar: () => void
  toggleCommandPalette: () => void

  // Theme (persisted)
  theme: 'light' | 'dark' | 'system'
  setTheme: (theme: 'light' | 'dark' | 'system') => void

  // UI preferences (persisted)
  preferences: UIPreferences
  updatePreferences: (preferences: Partial<UIPreferences>) => void

  // Notifications
  notifications: AppNotification[]
  addNotification: (notification: Omit<AppNotification, 'id'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void
}

// Chat UI state
export interface ChatUIState {
  // Composer state
  isComposing: boolean
  composerHeight: number
  setIsComposing: (composing: boolean) => void
  setComposerHeight: (height: number) => void

  // Draft message (persisted per conversation)
  drafts: Record<string, string>
  setDraft: (conversationId: string, draft: string) => void
  getDraft: (conversationId: string) => string
  clearDraft: (conversationId: string) => void

  // Sampling parameters
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  repetitionPenalty: number
  setTemperature: (temp: number) => void
  setTopP: (p: number) => void
  setTopK: (k: number) => void
  setMaxTokens: (tokens: number) => void
  setRepetitionPenalty: (penalty: number) => void
  resetSamplingParams: () => void

  // UI toggles
  showTokenCount: boolean
  showLatency: boolean
  showSamplingControls: boolean
  setShowTokenCount: (show: boolean) => void
  setShowLatency: (show: boolean) => void
  setShowSamplingControls: (show: boolean) => void

  // Streaming state
  isStreaming: boolean
  streamingConversationId: string | null
  setIsStreaming: (streaming: boolean) => void
  setStreamingConversation: (conversationId: string | null) => void
}

// Conversations state (local cache)
export interface ConversationsState {
  conversations: Conversation[]
  searchQuery: string
  setSearchQuery: (query: string) => void

  // CRUD operations
  addConversation: (conversation: Conversation) => void
  updateConversation: (id: string, updates: Partial<Conversation>) => void
  deleteConversation: (id: string) => void
  getConversation: (id: string) => Conversation | undefined

  // Filtering
  filteredConversations: () => Conversation[]

  // Bulk operations
  deleteAllConversations: () => void
  exportConversations: () => string
  importConversations: (json: string) => void
}

// Models UI state
export interface ModelsUIState {
  // Selection
  selectedModelIds: Set<string>
  toggleModelSelection: (modelId: string) => void
  selectAllModels: () => void
  clearSelection: () => void

  // Filtering and sorting
  searchQuery: string
  architectureFilter: string | null
  formatFilter: string | null
  quantFilter: string | null
  sortBy: 'name' | 'size' | 'created' | 'last_used'
  sortOrder: 'asc' | 'desc'
  setSearchQuery: (query: string) => void
  setArchitectureFilter: (arch: string | null) => void
  setFormatFilter: (format: string | null) => void
  setQuantFilter: (quant: string | null) => void
  setSortBy: (sortBy: ModelsUIState['sortBy']) => void
  setSortOrder: (order: 'asc' | 'desc') => void
  clearFilters: () => void

  // Detail drawer
  detailDrawerOpen: boolean
  detailDrawerModelId: string | null
  openDetailDrawer: (modelId: string) => void
  closeDetailDrawer: () => void

  // Import dialog
  importDialogOpen: boolean
  openImportDialog: () => void
  closeImportDialog: () => void
}

// Metrics UI state
export interface MetricsUIState {
  // Time range
  timeRange: '1h' | '6h' | '24h' | '7d' | 'all'
  setTimeRange: (range: MetricsUIState['timeRange']) => void

  // Auto-refresh
  autoRefresh: boolean
  refreshInterval: number // seconds
  setAutoRefresh: (enabled: boolean) => void
  setRefreshInterval: (interval: number) => void

  // Chart visibility
  visibleCharts: Set<string>
  toggleChartVisibility: (chartId: string) => void
}

// Settings UI state
export interface SettingsUIState {
  // Active tab
  activeTab: 'general' | 'performance' | 'paths' | 'updates' | 'privacy'
  setActiveTab: (tab: SettingsUIState['activeTab']) => void

  // Unsaved changes tracking
  hasUnsavedChanges: boolean
  setHasUnsavedChanges: (hasChanges: boolean) => void
}

// Logs UI state
export interface LogsUIState {
  // Filtering
  levelFilter: Set<'debug' | 'info' | 'warn' | 'error'>
  searchQuery: string
  setLevelFilter: (levels: Set<'debug' | 'info' | 'warn' | 'error'>) => void
  setSearchQuery: (query: string) => void
  toggleLevel: (level: 'debug' | 'info' | 'warn' | 'error') => void

  // Auto-scroll
  autoScroll: boolean
  setAutoScroll: (enabled: boolean) => void
}

// Notification types
export interface AppNotification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number // ms, undefined = persistent
  action?: {
    label: string
    onClick: () => void
  }
  timestamp: number
}

// Keyboard shortcuts state
export interface KeyboardShortcutsState {
  // Active shortcuts map
  shortcuts: Map<string, () => void>
  registerShortcut: (key: string, handler: () => void) => void
  unregisterShortcut: (key: string) => void
  clearShortcuts: () => void
}

// Combined store type (if using a single store)
export interface CombinedStore
  extends AppState,
    ChatUIState,
    ConversationsState,
    ModelsUIState,
    MetricsUIState,
    SettingsUIState,
    LogsUIState,
    KeyboardShortcutsState {}

// Persistence configuration
export interface StorePersistConfig {
  name: string
  version: number
  migrate?: (persistedState: unknown, version: number) => unknown
  partialize?: (state: unknown) => unknown
}

// Default sampling parameters
export const DEFAULT_SAMPLING_PARAMS = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  maxTokens: 2048,
  repetitionPenalty: 1.1,
}
