/**
 * Zustand store for client-side UI state management
 * Server state is managed by TanStack Query, not Zustand
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { generateId } from './utils'
import type {
  AppState,
  ChatUIState,
  ConversationsState,
  ModelsUIState,
  MetricsUIState,
  SettingsUIState,
  LogsUIState,
  KeyboardShortcutsState,
  CombinedStore,
  Conversation,
  AppNotification,
} from '@/types'
import { DEFAULT_SAMPLING_PARAMS, DEFAULT_UI_PREFERENCES } from '@/types'

/**
 * App-wide state store (persisted)
 */
export const useAppStore = create<AppState>()(
  persist(
    (set, _get) => ({
      // Daemon status
      daemonStatus: 'stopped',
      setDaemonStatus: status => set({ daemonStatus: status }),

      // Active selections
      activeModelId: null,
      activeConversationId: null,
      setActiveModel: modelId => set({ activeModelId: modelId }),
      setActiveConversation: conversationId =>
        set({ activeConversationId: conversationId }),

      // UI state
      sidebarOpen: true,
      commandPaletteOpen: false,
      setSidebarOpen: open => set({ sidebarOpen: open }),
      setCommandPaletteOpen: open => set({ commandPaletteOpen: open }),
      toggleSidebar: () => set(state => ({ sidebarOpen: !state.sidebarOpen })),
      toggleCommandPalette: () =>
        set(state => ({ commandPaletteOpen: !state.commandPaletteOpen })),

      // Theme (persisted)
      theme: 'system',
      setTheme: theme => set({ theme }),

      // UI preferences (persisted)
      preferences: DEFAULT_UI_PREFERENCES,
      updatePreferences: preferences =>
        set(state => ({
          preferences: { ...state.preferences, ...preferences },
        })),

      // Notifications (not persisted)
      notifications: [],
      addNotification: notification => {
        const id = generateId('notif')
        const newNotif: AppNotification = {
          ...notification,
          id,
          timestamp: Date.now(),
        }
        set(state => ({
          notifications: [...state.notifications, newNotif],
        }))

        // Auto-remove after duration if specified
        if (notification.duration) {
          setTimeout(() => {
            set(state => ({
              notifications: state.notifications.filter(n => n.id !== id),
            }))
          }, notification.duration)
        }

        return id
      },
      removeNotification: id =>
        set(state => ({
          notifications: state.notifications.filter(n => n.id !== id),
        })),
      clearNotifications: () => set({ notifications: [] }),
    }),
    {
      name: 'mlxr-app-storage',
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: state => ({
        activeModelId: state.activeModelId,
        activeConversationId: state.activeConversationId,
        sidebarOpen: state.sidebarOpen,
        theme: state.theme,
        preferences: state.preferences,
        // Don't persist: daemonStatus, commandPaletteOpen, notifications
      }),
    }
  )
)

/**
 * Chat UI state store (persisted)
 */
export const useChatUIStore = create<ChatUIState>()(
  persist(
    (set, get) => ({
      // Composer state
      isComposing: false,
      composerHeight: 100,
      setIsComposing: composing => set({ isComposing: composing }),
      setComposerHeight: height => set({ composerHeight: height }),

      // Draft messages (persisted per conversation)
      drafts: {},
      setDraft: (conversationId, draft) =>
        set(state => ({
          drafts: { ...state.drafts, [conversationId]: draft },
        })),
      getDraft: conversationId => get().drafts[conversationId] || '',
      clearDraft: conversationId =>
        set(state => {
          const { [conversationId]: _, ...rest } = state.drafts
          return { drafts: rest }
        }),

      // Sampling parameters (persisted)
      temperature: DEFAULT_SAMPLING_PARAMS.temperature,
      topP: DEFAULT_SAMPLING_PARAMS.topP,
      topK: DEFAULT_SAMPLING_PARAMS.topK,
      maxTokens: DEFAULT_SAMPLING_PARAMS.maxTokens,
      repetitionPenalty: DEFAULT_SAMPLING_PARAMS.repetitionPenalty,
      setTemperature: temp => set({ temperature: temp }),
      setTopP: p => set({ topP: p }),
      setTopK: k => set({ topK: k }),
      setMaxTokens: tokens => set({ maxTokens: tokens }),
      setRepetitionPenalty: penalty => set({ repetitionPenalty: penalty }),
      resetSamplingParams: () =>
        set({
          temperature: DEFAULT_SAMPLING_PARAMS.temperature,
          topP: DEFAULT_SAMPLING_PARAMS.topP,
          topK: DEFAULT_SAMPLING_PARAMS.topK,
          maxTokens: DEFAULT_SAMPLING_PARAMS.maxTokens,
          repetitionPenalty: DEFAULT_SAMPLING_PARAMS.repetitionPenalty,
        }),

      // UI toggles (persisted)
      showTokenCount: true,
      showLatency: true,
      showSamplingControls: false,
      setShowTokenCount: show => set({ showTokenCount: show }),
      setShowLatency: show => set({ showLatency: show }),
      setShowSamplingControls: show => set({ showSamplingControls: show }),

      // Streaming state (not persisted)
      isStreaming: false,
      streamingConversationId: null,
      setIsStreaming: streaming => set({ isStreaming: streaming }),
      setStreamingConversation: conversationId =>
        set({ streamingConversationId: conversationId }),
    }),
    {
      name: 'mlxr-chat-storage',
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: state => ({
        composerHeight: state.composerHeight,
        drafts: state.drafts,
        temperature: state.temperature,
        topP: state.topP,
        topK: state.topK,
        maxTokens: state.maxTokens,
        repetitionPenalty: state.repetitionPenalty,
        showTokenCount: state.showTokenCount,
        showLatency: state.showLatency,
        showSamplingControls: state.showSamplingControls,
        // Don't persist: isComposing, isStreaming, streamingConversationId
      }),
    }
  )
)

/**
 * Conversations state store (persisted)
 */
export const useConversationsStore = create<ConversationsState>()(
  persist(
    (set, get) => ({
      conversations: [],
      searchQuery: '',
      setSearchQuery: query => set({ searchQuery: query }),

      // CRUD operations
      addConversation: conversation =>
        set(state => ({
          conversations: [conversation, ...state.conversations],
        })),
      updateConversation: (id, updates) =>
        set(state => ({
          conversations: state.conversations.map(conv =>
            conv.id === id ? { ...conv, ...updates, updated_at: Date.now() } : conv
          ),
        })),
      deleteConversation: id =>
        set(state => ({
          conversations: state.conversations.filter(conv => conv.id !== id),
        })),
      getConversation: id => get().conversations.find(conv => conv.id === id),

      // Filtering
      filteredConversations: () => {
        const { conversations, searchQuery } = get()
        if (!searchQuery.trim()) return conversations

        const query = searchQuery.toLowerCase()
        return conversations.filter(
          conv =>
            conv.title.toLowerCase().includes(query) ||
            conv.messages.some(msg =>
              typeof msg.content === 'string'
                ? msg.content.toLowerCase().includes(query)
                : false
            )
        )
      },

      // Bulk operations
      deleteAllConversations: () => set({ conversations: [] }),
      exportConversations: () => JSON.stringify(get().conversations, null, 2),
      importConversations: json => {
        try {
          const imported = JSON.parse(json) as Conversation[]
          set({ conversations: imported })
        } catch (err) {
          console.error('Failed to import conversations:', err)
          throw new Error('Invalid conversation data')
        }
      },
    }),
    {
      name: 'mlxr-conversations-storage',
      version: 1,
      storage: createJSONStorage(() => localStorage),
    }
  )
)

/**
 * Models UI state store (not persisted)
 */
export const useModelsUIStore = create<ModelsUIState>((set, _get) => ({
  // Selection
  selectedModelIds: new Set<string>(),
  toggleModelSelection: modelId =>
    set(state => {
      const newSet = new Set(state.selectedModelIds)
      if (newSet.has(modelId)) {
        newSet.delete(modelId)
      } else {
        newSet.add(modelId)
      }
      return { selectedModelIds: newSet }
    }),
  selectAllModels: () => {
    // This would need access to the actual models list from TanStack Query
    // For now, we'll just clear selection
    set({ selectedModelIds: new Set() })
  },
  clearSelection: () => set({ selectedModelIds: new Set() }),

  // Filtering and sorting (persisted via useAppStore preferences)
  searchQuery: '',
  architectureFilter: null,
  formatFilter: null,
  quantFilter: null,
  sortBy: 'name',
  sortOrder: 'asc',
  setSearchQuery: query => set({ searchQuery: query }),
  setArchitectureFilter: arch => set({ architectureFilter: arch }),
  setFormatFilter: format => set({ formatFilter: format }),
  setQuantFilter: quant => set({ quantFilter: quant }),
  setSortBy: sortBy => set({ sortBy }),
  setSortOrder: order => set({ sortOrder: order }),
  clearFilters: () =>
    set({
      searchQuery: '',
      architectureFilter: null,
      formatFilter: null,
      quantFilter: null,
    }),

  // Detail drawer
  detailDrawerOpen: false,
  detailDrawerModelId: null,
  openDetailDrawer: modelId =>
    set({ detailDrawerOpen: true, detailDrawerModelId: modelId }),
  closeDetailDrawer: () =>
    set({ detailDrawerOpen: false, detailDrawerModelId: null }),

  // Import dialog
  importDialogOpen: false,
  openImportDialog: () => set({ importDialogOpen: true }),
  closeImportDialog: () => set({ importDialogOpen: false }),
}))

/**
 * Metrics UI state store (persisted)
 */
export const useMetricsUIStore = create<MetricsUIState>()(
  persist(
    (set, _get) => ({
      // Time range
      timeRange: '1h',
      setTimeRange: range => set({ timeRange: range }),

      // Auto-refresh
      autoRefresh: true,
      refreshInterval: 5, // seconds
      setAutoRefresh: enabled => set({ autoRefresh: enabled }),
      setRefreshInterval: interval => set({ refreshInterval: interval }),

      // Chart visibility
      visibleCharts: new Set([
        'latency',
        'throughput',
        'memory',
        'kv_cache',
        'requests',
      ]),
      toggleChartVisibility: chartId =>
        set(state => {
          const newSet = new Set(state.visibleCharts)
          if (newSet.has(chartId)) {
            newSet.delete(chartId)
          } else {
            newSet.add(chartId)
          }
          return { visibleCharts: newSet }
        }),
    }),
    {
      name: 'mlxr-metrics-storage',
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: state => ({
        timeRange: state.timeRange,
        autoRefresh: state.autoRefresh,
        refreshInterval: state.refreshInterval,
        // visibleCharts is a Set, need to convert to array for JSON
        visibleCharts: Array.from(state.visibleCharts),
      }),
      // Custom merge to handle Set conversion
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<MetricsUIState>
        return {
          ...currentState,
          ...persisted,
          visibleCharts: new Set(
            Array.isArray(persisted.visibleCharts)
              ? persisted.visibleCharts
              : currentState.visibleCharts
          ),
        }
      },
    }
  )
)

/**
 * Settings UI state store (not persisted)
 */
export const useSettingsUIStore = create<SettingsUIState>((set, _get) => ({
  // Active tab
  activeTab: 'general',
  setActiveTab: tab => set({ activeTab: tab }),

  // Unsaved changes tracking
  hasUnsavedChanges: false,
  setHasUnsavedChanges: hasChanges => set({ hasUnsavedChanges: hasChanges }),
}))

/**
 * Logs UI state store (persisted)
 */
export const useLogsUIStore = create<LogsUIState>()(
  persist(
    (set, _get) => ({
      // Filtering
      levelFilter: new Set(['debug', 'info', 'warn', 'error']),
      searchQuery: '',
      setLevelFilter: levels => set({ levelFilter: levels }),
      setSearchQuery: query => set({ searchQuery: query }),
      toggleLevel: level =>
        set(state => {
          const newSet = new Set(state.levelFilter)
          if (newSet.has(level)) {
            newSet.delete(level)
          } else {
            newSet.add(level)
          }
          return { levelFilter: newSet }
        }),

      // Auto-scroll
      autoScroll: true,
      setAutoScroll: enabled => set({ autoScroll: enabled }),
    }),
    {
      name: 'mlxr-logs-storage',
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: state => ({
        // levelFilter is a Set, need to convert to array for JSON
        levelFilter: Array.from(state.levelFilter),
        autoScroll: state.autoScroll,
        // Don't persist searchQuery
      }),
      // Custom merge to handle Set conversion
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<LogsUIState>
        return {
          ...currentState,
          ...persisted,
          levelFilter: new Set(
            Array.isArray(persisted.levelFilter)
              ? persisted.levelFilter
              : currentState.levelFilter
          ),
        }
      },
    }
  )
)

/**
 * Keyboard shortcuts state store (not persisted - runtime only)
 */
export const useKeyboardShortcutsStore = create<KeyboardShortcutsState>(
  (set, _get) => ({
    shortcuts: new Map<string, () => void>(),
    registerShortcut: (key, handler) =>
      set(state => {
        const newMap = new Map(state.shortcuts)
        newMap.set(key, handler)
        return { shortcuts: newMap }
      }),
    unregisterShortcut: key =>
      set(state => {
        const newMap = new Map(state.shortcuts)
        newMap.delete(key)
        return { shortcuts: newMap }
      }),
    clearShortcuts: () => set({ shortcuts: new Map() }),
  })
)

/**
 * Combined store selector (if needed for complex cross-store operations)
 * Note: Generally prefer using individual stores for better performance
 */
export const useCombinedStore = (): CombinedStore => {
  const appStore = useAppStore()
  const chatUIStore = useChatUIStore()
  const conversationsStore = useConversationsStore()
  const modelsUIStore = useModelsUIStore()
  const metricsUIStore = useMetricsUIStore()
  const settingsUIStore = useSettingsUIStore()
  const logsUIStore = useLogsUIStore()
  const keyboardShortcutsStore = useKeyboardShortcutsStore()

  return {
    ...appStore,
    ...chatUIStore,
    ...conversationsStore,
    ...modelsUIStore,
    ...metricsUIStore,
    ...settingsUIStore,
    ...logsUIStore,
    ...keyboardShortcutsStore,
  }
}

/**
 * Utility: Clear all persisted state (for testing or reset)
 */
export const clearAllStores = () => {
  localStorage.removeItem('mlxr-app-storage')
  localStorage.removeItem('mlxr-chat-storage')
  localStorage.removeItem('mlxr-conversations-storage')
  localStorage.removeItem('mlxr-metrics-storage')
  localStorage.removeItem('mlxr-logs-storage')
}

/**
 * Utility: Export all state to JSON (for backup)
 */
export const exportAllState = () => {
  return {
    app: useAppStore.getState(),
    chat: useChatUIStore.getState(),
    conversations: useConversationsStore.getState().conversations,
    metrics: useMetricsUIStore.getState(),
    logs: useLogsUIStore.getState(),
    timestamp: Date.now(),
  }
}
