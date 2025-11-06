/**
 * ConversationList Component
 *
 * List of conversations with:
 * - Search/filter
 * - Conversation items with title and timestamp
 * - Active state
 * - Delete/edit actions
 * - New conversation button
 */

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { MessageSquare, Plus, Search, Trash2 } from 'lucide-react'
import { useConversations } from '@/hooks/useChat'
import { useDebounce } from '@/hooks/useDebounce'
import type { Conversation } from '@/types/store'

export interface ConversationListProps {
  onSelectConversation?: (id: string) => void
  onNewConversation?: () => void
  className?: string
}

export function ConversationList({
  onSelectConversation,
  onNewConversation,
  className,
}: ConversationListProps) {
  const {
    conversations,
    activeConversationId,
    setActiveConversation,
    deleteConversation,
    setSearchQuery,
    filteredConversations,
  } = useConversations()

  const [localSearchQuery, setLocalSearchQuery] = useState('')
  const debouncedSearchQuery = useDebounce(localSearchQuery, 300)

  // Update search query in store when debounced value changes
  useEffect(() => {
    setSearchQuery(debouncedSearchQuery)
  }, [debouncedSearchQuery, setSearchQuery])

  const handleNewConversation = () => {
    onNewConversation?.()
  }

  const handleSelectConversation = (id: string) => {
    setActiveConversation(id)
    onSelectConversation?.(id)
  }

  const handleDeleteConversation = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteConversation(id)
  }

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) {
      return 'Today'
    } else if (days === 1) {
      return 'Yesterday'
    } else if (days < 7) {
      return `${days} days ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const displayedConversations =
    debouncedSearchQuery.length > 0 ? filteredConversations() : conversations

  return (
    <div className={cn('flex h-full flex-col', className)}>
      {/* Header */}
      <div className="space-y-2 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Conversations</h2>
          <Button
            size="sm"
            onClick={handleNewConversation}
            className="h-8 w-8 p-0"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={localSearchQuery}
            onChange={(e) => setLocalSearchQuery(e.target.value)}
            placeholder="Search conversations..."
            className="pl-9"
          />
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto">
        <div className="space-y-1 p-2">
          {displayedConversations.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <MessageSquare className="mb-2 h-8 w-8 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">
                {debouncedSearchQuery
                  ? 'No conversations found'
                  : 'No conversations yet'}
              </p>
            </div>
          ) : (
            displayedConversations.map((conversation) => (
              <ConversationItem
                key={conversation.id}
                conversation={conversation}
                isActive={conversation.id === activeConversationId}
                onClick={() => handleSelectConversation(conversation.id)}
                onDelete={(e) => handleDeleteConversation(conversation.id, e)}
                formatTimestamp={formatTimestamp}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

interface ConversationItemProps {
  conversation: Conversation
  isActive: boolean
  onClick: () => void
  onDelete: (e: React.MouseEvent) => void
  formatTimestamp: (timestamp: number) => string
}

function ConversationItem({
  conversation,
  isActive,
  onClick,
  onDelete,
  formatTimestamp,
}: ConversationItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'group flex w-full items-start gap-3 rounded-lg p-3 text-left transition-colors',
        'hover:bg-accent',
        isActive && 'bg-accent'
      )}
    >
      <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="flex-1 overflow-hidden">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-sm font-medium">{conversation.title}</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDelete}
            className="h-6 w-6 shrink-0 p-0 opacity-0 transition-opacity group-hover:opacity-100"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{formatTimestamp(conversation.updated_at)}</span>
          {conversation.messages.length > 0 && (
            <>
              <span>"</span>
              <span>{conversation.messages.length} messages</span>
            </>
          )}
        </div>
      </div>
    </button>
  )
}
