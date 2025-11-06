/**
 * useConversation Hook
 *
 * Conversation management with local storage
 * - Create, read, update, delete conversations
 * - Persist to localStorage
 * - Search and filter
 * - Export/import functionality
 */

import { useCallback, useState } from 'react'
import { generateId } from '@/lib/utils'
import { useLocalStorage } from './useLocalStorage'
import type { Conversation, StoredChatMessage } from '@/types/store'

const CONVERSATIONS_STORAGE_KEY = 'mlxr-conversations'

export interface UseConversationReturn {
  // Current conversation
  conversation: Conversation | undefined
  messages: StoredChatMessage[]

  // All conversations
  conversations: Conversation[]
  isLoading: boolean

  // Actions
  createConversation: (title: string, modelId: string) => Conversation
  updateConversation: (id: string, updates: Partial<Conversation>) => void
  deleteConversation: (id: string) => void
  addMessage: (conversationId: string, message: StoredChatMessage) => void
  updateMessage: (conversationId: string, messageId: string, updates: Partial<StoredChatMessage>) => void
  deleteMessage: (conversationId: string, messageId: string) => void
  clearMessages: (conversationId: string) => void

  // Search and filter
  searchConversations: (query: string) => Conversation[]
  filterByModel: (modelId: string) => Conversation[]

  // Export/import
  exportConversation: (id: string) => string
  exportAllConversations: () => string
  importConversations: (json: string) => void

  // Bulk operations
  deleteAllConversations: () => void
}

/**
 * Hook for conversation management
 */
export function useConversation(
  conversationId?: string
): UseConversationReturn {
  const [conversations, setConversations, removeConversations] = useLocalStorage<
    Conversation[]
  >(CONVERSATIONS_STORAGE_KEY, [])

  const [isLoading] = useState(false)

  // Get current conversation
  const conversation = conversationId
    ? conversations.find((c) => c.id === conversationId)
    : undefined

  // Get messages for current conversation
  const messages = conversation?.messages || []

  // Create new conversation
  const createConversation = useCallback(
    (title: string, modelId: string): Conversation => {
      const newConversation: Conversation = {
        id: generateId(),
        title,
        model_id: modelId,
        messages: [],
        created_at: Date.now(),
        updated_at: Date.now(),
        metadata: {
          total_tokens: 0,
          total_cost: 0,
        },
      }

      setConversations((prev) => [newConversation, ...prev])
      return newConversation
    },
    [setConversations]
  )

  // Update conversation
  const updateConversation = useCallback(
    (id: string, updates: Partial<Conversation>) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === id
            ? {
                ...c,
                ...updates,
                updated_at: Date.now(),
              }
            : c
        )
      )
    },
    [setConversations]
  )

  // Delete conversation
  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id))
    },
    [setConversations]
  )

  // Add message to conversation
  const addMessage = useCallback(
    (convId: string, message: StoredChatMessage) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [...c.messages, message],
                updated_at: Date.now(),
              }
            : c
        )
      )
    },
    [setConversations]
  )

  // Update message
  const updateMessage = useCallback(
    (convId: string, messageId: string, updates: Partial<StoredChatMessage>) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === messageId
                    ? { ...m, ...updates }
                    : m
                ),
                updated_at: Date.now(),
              }
            : c
        )
      )
    },
    [setConversations]
  )

  // Delete message
  const deleteMessage = useCallback(
    (convId: string, messageId: string) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.filter((m) => m.id !== messageId),
                updated_at: Date.now(),
              }
            : c
        )
      )
    },
    [setConversations]
  )

  // Clear all messages in conversation
  const clearMessages = useCallback(
    (convId: string) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: [],
                updated_at: Date.now(),
              }
            : c
        )
      )
    },
    [setConversations]
  )

  // Search conversations
  const searchConversations = useCallback(
    (query: string): Conversation[] => {
      const lowerQuery = query.toLowerCase()
      return conversations.filter(
        (c) =>
          c.title.toLowerCase().includes(lowerQuery) ||
          c.messages.some((m) =>
            m.content?.toLowerCase().includes(lowerQuery)
          )
      )
    },
    [conversations]
  )

  // Filter by model
  const filterByModel = useCallback(
    (modelId: string): Conversation[] => {
      return conversations.filter((c) => c.model_id === modelId)
    },
    [conversations]
  )

  // Export single conversation
  const exportConversation = useCallback(
    (id: string): string => {
      const conv = conversations.find((c) => c.id === id)
      if (!conv) {
        throw new Error(`Conversation ${id} not found`)
      }
      return JSON.stringify(conv, null, 2)
    },
    [conversations]
  )

  // Export all conversations
  const exportAllConversations = useCallback((): string => {
    return JSON.stringify(conversations, null, 2)
  }, [conversations])

  // Import conversations
  const importConversations = useCallback(
    (json: string) => {
      try {
        const imported = JSON.parse(json) as Conversation | Conversation[]
        const conversationsToImport = Array.isArray(imported)
          ? imported
          : [imported]

        setConversations((prev) => {
          // Merge with existing, avoiding duplicates
          const existingIds = new Set(prev.map((c) => c.id))
          const newConversations = conversationsToImport.filter(
            (c) => !existingIds.has(c.id)
          )
          return [...newConversations, ...prev]
        })
      } catch (error) {
        throw new Error(
          `Failed to import conversations: ${error instanceof Error ? error.message : 'Invalid JSON'}`
        )
      }
    },
    [setConversations]
  )

  // Delete all conversations
  const deleteAllConversations = useCallback(() => {
    removeConversations()
  }, [removeConversations])

  return {
    conversation,
    messages,
    conversations,
    isLoading,
    createConversation,
    updateConversation,
    deleteConversation,
    addMessage,
    updateMessage,
    deleteMessage,
    clearMessages,
    searchConversations,
    filterByModel,
    exportConversation,
    exportAllConversations,
    importConversations,
    deleteAllConversations,
  }
}

/**
 * Hook to get conversation list with pagination
 */
export function useConversationList(options?: {
  limit?: number
  offset?: number
  sortBy?: 'created_at' | 'updated_at' | 'title'
  sortOrder?: 'asc' | 'desc'
}) {
  const { conversations } = useConversation()
  const { limit = 50, offset = 0, sortBy = 'updated_at', sortOrder = 'desc' } = options || {}

  // Sort conversations
  const sortedConversations = [...conversations].sort((a, b) => {
    const aValue = a[sortBy]
    const bValue = b[sortBy]

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortOrder === 'asc'
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue)
    }

    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue
    }

    return 0
  })

  // Paginate
  const paginatedConversations = sortedConversations.slice(offset, offset + limit)

  return {
    conversations: paginatedConversations,
    total: conversations.length,
    hasMore: offset + limit < conversations.length,
  }
}

/**
 * Hook to get conversation statistics
 */
export function useConversationStats() {
  const { conversations } = useConversation()

  const stats = {
    total: conversations.length,
    totalMessages: conversations.reduce((sum, c) => sum + c.messages.length, 0),
    totalTokens: conversations.reduce(
      (sum, c) => sum + (c.metadata?.total_tokens || 0),
      0
    ),
    byModel: conversations.reduce(
      (acc, c) => {
        acc[c.model_id] = (acc[c.model_id] || 0) + 1
        return acc
      },
      {} as Record<string, number>
    ),
    recentActivity: conversations
      .sort((a, b) => b.updated_at - a.updated_at)
      .slice(0, 5),
  }

  return stats
}
