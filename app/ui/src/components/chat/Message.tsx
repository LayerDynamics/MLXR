/**
 * Message Component
 *
 * Individual message display with:
 * - User/assistant/system role styling
 * - Markdown rendering
 * - Code block highlighting
 * - Copy/edit/delete actions
 * - Token count and latency display
 */

import { memo } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Copy, Edit, Trash2, User, Bot, Terminal } from 'lucide-react'
import type { StoredChatMessage } from '@/types/store'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

export interface MessageProps {
  message: StoredChatMessage
  isStreaming?: boolean
  showActions?: boolean
  showTokenCount?: boolean
  showLatency?: boolean
  tokenCount?: number
  latencyMs?: number
  onCopy?: () => void
  onEdit?: () => void
  onDelete?: () => void
}

function MessageComponent({
  message,
  isStreaming = false,
  showActions = true,
  showTokenCount = false,
  showLatency = false,
  tokenCount,
  latencyMs,
  onCopy,
  onEdit,
  onDelete,
}: MessageProps) {
  const { role, content } = message

  // Get role-specific styling
  const getRoleIcon = () => {
    switch (role) {
      case 'user':
        return <User className="h-5 w-5" />
      case 'assistant':
        return <Bot className="h-5 w-5" />
      case 'system':
        return <Terminal className="h-5 w-5" />
      default:
        return null
    }
  }

  const getRoleLabel = () => {
    switch (role) {
      case 'user':
        return 'You'
      case 'assistant':
        return 'Assistant'
      case 'system':
        return 'System'
      default:
        return role
    }
  }

  const getRoleBgColor = () => {
    switch (role) {
      case 'user':
        return 'bg-primary/5'
      case 'assistant':
        return 'bg-muted/30'
      case 'system':
        return 'bg-accent/20'
      default:
        return 'bg-background'
    }
  }

  const handleCopy = () => {
    if (content) {
      navigator.clipboard.writeText(content)
      onCopy?.()
    }
  }

  return (
    <div
      className={cn(
        'group relative px-4 py-6 transition-colors hover:bg-muted/50',
        getRoleBgColor()
      )}
    >
      <div className="mx-auto flex max-w-3xl gap-4">
        {/* Role Icon */}
        <div
          className={cn(
            'flex h-8 w-8 shrink-0 items-center justify-center rounded-full',
            role === 'user' && 'bg-primary text-primary-foreground',
            role === 'assistant' && 'bg-secondary text-secondary-foreground',
            role === 'system' && 'bg-accent text-accent-foreground'
          )}
        >
          {getRoleIcon()}
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-2">
          {/* Role Label */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">{getRoleLabel()}</span>
            {(showTokenCount || showLatency) && (
              <div className="flex gap-3 text-xs text-muted-foreground">
                {showTokenCount && tokenCount !== undefined && (
                  <span>{tokenCount} tokens</span>
                )}
                {showLatency && latencyMs !== undefined && (
                  <span>{latencyMs}ms</span>
                )}
              </div>
            )}
          </div>

          {/* Message Text */}
          <div className="prose prose-sm dark:prose-invert max-w-none">
            {isStreaming && role === 'assistant' ? (
              <div className="flex items-center gap-2">
                <span>{content}</span>
                <span className="inline-block h-4 w-1 animate-pulse bg-primary" />
              </div>
            ) : (
              <ReactMarkdown
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    const language = match ? match[1] : ''
                    const inline = !className

                    return !inline && language ? (
                      <SyntaxHighlighter
                        style={oneDark as any}
                        language={language}
                        PreTag="div"
                        className="rounded-md"
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {content || ''}
              </ReactMarkdown>
            )}
          </div>

          {/* Action Buttons */}
          {showActions && !isStreaming && (
            <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="h-7 px-2"
              >
                <Copy className="h-3 w-3" />
              </Button>
              {role === 'user' && onEdit && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onEdit}
                  className="h-7 px-2"
                >
                  <Edit className="h-3 w-3" />
                </Button>
              )}
              {onDelete && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onDelete}
                  className="h-7 px-2 text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export const Message = memo(MessageComponent)
