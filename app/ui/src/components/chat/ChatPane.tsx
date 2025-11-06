/**
 * ChatPane Component
 *
 * Main chat interface that combines:
 * - MessageList for displaying messages
 * - Composer for input
 * - TokenStream for streaming stats
 * - Integration with useChat hook
 */

import { useState } from 'react'
import { MessageList } from './MessageList'
import { Composer } from './Composer'
import { TokenStream } from './TokenStream'
import { useChat } from '@/hooks/useChat'
import { useAppStore } from '@/lib/store'

export interface ChatPaneProps {
  conversationId?: string
}

export function ChatPane({ conversationId }: ChatPaneProps) {
  const [inputValue, setInputValue] = useState('')
  const activeModelId = useAppStore((state) => state.activeModelId)

  const {
    messages,
    isStreaming,
    streamingMessage,
    tokensPerSecond,
    sendMessage,
    cancelStream,
    deleteMessage,
  } = useChat({
    conversationId,
    modelId: activeModelId || undefined,
  })

  const handleSubmit = async () => {
    if (!inputValue.trim() || isStreaming) return

    const message = inputValue.trim()
    setInputValue('') // Clear input immediately

    try {
      await sendMessage(message)
    } catch (error) {
      console.error('Failed to send message:', error)
    }
  }

  const handleEditMessage = (messageId: string) => {
    // Find message and set it as input
    const message = messages.find((m) => m.id === messageId)
    if (message && message.content) {
      setInputValue(message.content)
    }
  }

  const handleDeleteMessage = (messageId: string) => {
    const messageIndex = messages.findIndex((m) => m.id === messageId)
    if (messageIndex !== -1) {
      deleteMessage(messageIndex)
    }
  }

  const handleCopyMessage = (messageId: string) => {
    const message = messages.find((m) => m.id === messageId)
    if (message?.content) {
      navigator.clipboard.writeText(message.content)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Message List */}
      <MessageList
        messages={messages}
        isStreaming={isStreaming}
        streamingMessage={streamingMessage}
        showTokenCount={false}
        showLatency={false}
        autoScroll={true}
        onEditMessage={handleEditMessage}
        onDeleteMessage={handleDeleteMessage}
        onCopyMessage={handleCopyMessage}
      />

      {/* Streaming Stats */}
      {isStreaming && (
        <div className="px-4 pb-2">
          <TokenStream
            isStreaming={isStreaming}
            tokensPerSecond={tokensPerSecond}
            totalTokens={streamingMessage.length}
            onStop={cancelStream}
          />
        </div>
      )}

      {/* Message Composer */}
      <Composer
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSubmit}
        disabled={isStreaming}
        showCharCount={true}
        placeholder={
          activeModelId
            ? 'Type a message...'
            : 'Select a model to start chatting'
        }
      />
    </div>
  )
}
