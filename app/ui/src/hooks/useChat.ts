/**
 * useChat Hook
 *
 * Manages chat state and message streaming
 * - Send messages with SSE streaming
 * - Message history management
 * - Conversation CRUD operations
 * - Token streaming with real-time updates
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { ChatMessage } from '@/types/openai'
import type { Conversation, StoredChatMessage } from '@/types/store'
import { openai } from '@/lib/api'
import { useConversationsStore, useAppStore } from '@/lib/store'
import { generateId } from '@/lib/utils'
import { queryKeys } from '@/lib/queryClient'

export interface UseChatOptions {
  conversationId?: string
  modelId?: string
  onError?: (error: Error) => void
  onTokenReceived?: (token: string) => void
  onComplete?: (message: ChatMessage) => void
}

export interface UseChatReturn {
  // State
  messages: StoredChatMessage[]
  isStreaming: boolean
  streamingMessage: string
  error: Error | null
  tokensPerSecond: number

  // Actions
  sendMessage: (content: string, role?: 'user' | 'system') => Promise<void>
  cancelStream: () => void
  regenerateLastMessage: () => Promise<void>
  clearMessages: () => void
  updateMessage: (messageIndex: number, updates: Partial<StoredChatMessage>) => void
  deleteMessage: (messageIndex: number) => void
}

/**
 * Chat hook for message management and streaming
 */
export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const { conversationId, modelId, onError, onTokenReceived, onComplete } =
    options

  const queryClient = useQueryClient()
  const conversationsStore = useConversationsStore()
  const appStore = useAppStore()

  // State
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState('')
  const [error, setError] = useState<Error | null>(null)
  const [tokensPerSecond, setTokensPerSecond] = useState(0)

  // Refs for streaming
  const abortControllerRef = useRef<AbortController | null>(null)
  const streamStartTimeRef = useRef<number>(0)
  const tokenCountRef = useRef<number>(0)

  // Get current conversation
  const conversation = conversationId
    ? conversationsStore.getConversation(conversationId)
    : undefined

  // Get messages from conversation or empty array
  const messages = conversation?.messages || []

  // Get active model
  const activeModelId = modelId || appStore.activeModelId || ''

  /**
   * Helper to add a message to a conversation
   */
  const addMessageToConversation = useCallback(
    (convId: string, message: ChatMessage) => {
      const conv = conversationsStore.getConversation(convId)
      if (conv) {
        // Ensure message has an ID
        const storedMessage: StoredChatMessage = {
          ...message,
          id: generateId('msg'),
        }
        conversationsStore.updateConversation(convId, {
          messages: [...conv.messages, storedMessage],
          updated_at: Date.now(),
        })
      }
    },
    [conversationsStore]
  )

  /**
   * Send a message and stream the response
   */
  const sendMessage = useCallback(
    async (content: string, role: 'user' | 'system' = 'user') => {
      if (!activeModelId) {
        const err = new Error('No model selected')
        setError(err)
        onError?.(err)
        return
      }

      setError(null)
      setIsStreaming(true)
      setStreamingMessage('')
      tokenCountRef.current = 0
      streamStartTimeRef.current = Date.now()

      // Create user message
      const userMessage: ChatMessage = {
        role,
        content,
      }

      // Create or update conversation
      let currentConversationId = conversationId
      if (!currentConversationId) {
        // Create new conversation with stored message (including ID)
        const storedUserMessage: StoredChatMessage = {
          ...userMessage,
          id: generateId('msg'),
        }
        const newConversation: Conversation = {
          id: generateId('conv'),
          title: content.slice(0, 50) + (content.length > 50 ? '...' : ''),
          model_id: activeModelId,
          messages: [storedUserMessage],
          created_at: Date.now(),
          updated_at: Date.now(),
        }
        conversationsStore.addConversation(newConversation)
        currentConversationId = newConversation.id
        appStore.setActiveConversation(currentConversationId)
      } else {
        // Add message to existing conversation
        addMessageToConversation(currentConversationId, userMessage)
      }

      // Build request messages (all messages in conversation + new user message)
      const currentConv = conversationsStore.getConversation(
        currentConversationId
      )
      // Strip IDs from stored messages for API (API expects ChatMessage without ID)
      const requestMessages: ChatMessage[] = currentConv?.messages.map(({ id: _, ...msg }) => msg) || [userMessage]

      try {
        // Create abort controller
        abortControllerRef.current = new AbortController()

        // Stream completion
        const stream = openai.streamChatCompletion({
          model: activeModelId,
          messages: requestMessages,
          stream: true,
        })

        let fullResponse = ''
        let lastUpdateTime = Date.now()

        for await (const chunk of stream) {
          // Check if aborted
          if (abortControllerRef.current?.signal.aborted) {
            break
          }

          const delta = chunk.choices[0]?.delta?.content
          if (delta) {
            fullResponse += delta
            setStreamingMessage(fullResponse)
            tokenCountRef.current++
            onTokenReceived?.(delta)

            // Update tokens per second every 100ms
            const now = Date.now()
            if (now - lastUpdateTime >= 100) {
              const elapsed = (now - streamStartTimeRef.current) / 1000
              setTokensPerSecond(
                elapsed > 0 ? Math.round(tokenCountRef.current / elapsed) : 0
              )
              lastUpdateTime = now
            }
          }

          // Check for finish reason
          const finishReason = chunk.choices[0]?.finish_reason
          if (finishReason) {
            // Create assistant message
            const assistantMessage: ChatMessage = {
              role: 'assistant',
              content: fullResponse,
            }

            // Add to conversation
            if (currentConversationId) {
              addMessageToConversation(currentConversationId, assistantMessage)
            }

            // Update query cache
            queryClient.invalidateQueries({
              queryKey: queryKeys.conversations.detail(currentConversationId!),
            })

            onComplete?.(assistantMessage)
            break
          }
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Unknown error')
        setError(error)
        onError?.(error)

        // Add error message to conversation
        if (currentConversationId) {
          const errorMessage: ChatMessage = {
            role: 'assistant',
            content: `Error: ${error.message}`,
          }
          addMessageToConversation(currentConversationId, errorMessage)
        }
      } finally {
        setIsStreaming(false)
        setStreamingMessage('')
        setTokensPerSecond(0)
        abortControllerRef.current = null
      }
    },
    [
      activeModelId,
      conversationId,
      conversationsStore,
      appStore,
      queryClient,
      onError,
      onTokenReceived,
      onComplete,
      addMessageToConversation,
    ]
  )

  /**
   * Cancel streaming
   */
  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsStreaming(false)
      setStreamingMessage('')
      setTokensPerSecond(0)
    }
  }, [])

  /**
   * Regenerate last assistant message
   */
  const regenerateLastMessage = useCallback(async () => {
    if (!conversation || conversation.messages.length < 2) {
      return
    }

    // Find last user message
    let lastUserMessageIndex = -1
    for (let i = conversation.messages.length - 1; i >= 0; i--) {
      if (conversation.messages[i].role === 'user') {
        lastUserMessageIndex = i
        break
      }
    }

    if (lastUserMessageIndex === -1) {
      return
    }

    // Remove messages after last user message
    const messagesToKeep = conversation.messages.slice(
      0,
      lastUserMessageIndex + 1
    )
    conversationsStore.updateConversation(conversation.id, {
      messages: messagesToKeep,
    })

    // Resend last user message
    const lastUserMessage = messagesToKeep[messagesToKeep.length - 1]
    const content =
      typeof lastUserMessage.content === 'string'
        ? lastUserMessage.content
        : ''
    await sendMessage(content, lastUserMessage.role as 'user')
  }, [conversation, conversationsStore, sendMessage])

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    if (conversationId) {
      conversationsStore.updateConversation(conversationId, {
        messages: [],
        updated_at: Date.now(),
      })
    }
  }, [conversationId, conversationsStore])

  /**
   * Update a specific message by index
   */
  const updateMessage = useCallback(
    (messageIndex: number, updates: Partial<StoredChatMessage>) => {
      if (!conversation) return

      const updatedMessages = conversation.messages.map((msg, index) => {
        if (index === messageIndex) {
          return { ...msg, ...updates }
        }
        return msg
      })

      conversationsStore.updateConversation(conversation.id, {
        messages: updatedMessages,
        updated_at: Date.now(),
      })
    },
    [conversation, conversationsStore]
  )

  /**
   * Delete a specific message by index
   */
  const deleteMessage = useCallback(
    (messageIndex: number) => {
      if (!conversation) return

      const updatedMessages = conversation.messages.filter(
        (_msg, index) => index !== messageIndex
      )

      conversationsStore.updateConversation(conversation.id, {
        messages: updatedMessages,
        updated_at: Date.now(),
      })
    },
    [conversation, conversationsStore]
  )

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  return {
    messages,
    isStreaming,
    streamingMessage,
    error,
    tokensPerSecond,
    sendMessage,
    cancelStream,
    regenerateLastMessage,
    clearMessages,
    updateMessage,
    deleteMessage,
  }
}

/**
 * Hook for managing conversation list
 */
export function useConversations() {
  const conversationsStore = useConversationsStore()
  const appStore = useAppStore()

  const createConversation = useCallback(
    (modelIdParam: string, title?: string) => {
      const conversation: Conversation = {
        id: generateId('conv'),
        title: title || 'New Conversation',
        model_id: modelIdParam,
        messages: [],
        created_at: Date.now(),
        updated_at: Date.now(),
      }
      conversationsStore.addConversation(conversation)
      return conversation
    },
    [conversationsStore]
  )

  return {
    conversations: conversationsStore.conversations,
    activeConversationId: appStore.activeConversationId,
    setActiveConversation: appStore.setActiveConversation,
    createConversation,
    updateConversation: conversationsStore.updateConversation,
    deleteConversation: conversationsStore.deleteConversation,
    searchQuery: conversationsStore.searchQuery,
    setSearchQuery: conversationsStore.setSearchQuery,
    filteredConversations: conversationsStore.filteredConversations,
  }
}
