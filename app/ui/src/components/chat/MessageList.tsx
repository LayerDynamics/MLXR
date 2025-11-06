/**
 * MessageList Component
 *
 * Scrollable list of messages with:
 * - Auto-scroll to bottom on new messages
 * - Virtualization for performance (optional)
 * - Empty state
 * - Loading state
 */

import { useEffect, useRef } from 'react'
import { Message } from './Message'
import type { StoredChatMessage } from '@/types/store'
import { MessageSquare } from 'lucide-react'

export interface MessageListProps {
  messages: StoredChatMessage[]
  isStreaming?: boolean
  streamingMessage?: string
  showTokenCount?: boolean
  showLatency?: boolean
  autoScroll?: boolean
  onEditMessage?: (messageId: string) => void
  onDeleteMessage?: (messageId: string) => void
  onCopyMessage?: (messageId: string) => void
}

export function MessageList({
  messages,
  isStreaming = false,
  streamingMessage = '',
  showTokenCount = false,
  showLatency = false,
  autoScroll = true,
  onEditMessage,
  onDeleteMessage,
  onCopyMessage,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages.length, streamingMessage, autoScroll])

  // Empty state
  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center">
        <MessageSquare className="mb-4 h-12 w-12 text-muted-foreground/50" />
        <h3 className="mb-2 text-lg font-medium">No messages yet</h3>
        <p className="text-sm text-muted-foreground">
          Start a conversation by typing a message below
        </p>
      </div>
    )
  }

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto scroll-smooth"
    >
      <div className="flex flex-col">
        {/* Render all messages */}
        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            showActions={true}
            showTokenCount={showTokenCount}
            showLatency={showLatency}
            onCopy={() => onCopyMessage?.(message.id)}
            onEdit={
              message.role === 'user'
                ? () => onEditMessage?.(message.id)
                : undefined
            }
            onDelete={() => onDeleteMessage?.(message.id)}
          />
        ))}

        {/* Streaming message (assistant is typing) */}
        {isStreaming && streamingMessage && (
          <Message
            message={{
              id: 'streaming',
              role: 'assistant',
              content: streamingMessage,
            }}
            isStreaming={true}
            showActions={false}
          />
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
