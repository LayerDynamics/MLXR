/**
 * Composer Component
 *
 * Message input area with:
 * - Auto-resizing textarea
 * - Keyboard shortcuts (Enter to send, Shift+Enter for new line)
 * - Character/token count
 * - Send button
 * - Attachment button integration
 * - Disabled state when streaming
 */

import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Send } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  placeholder?: string
  disabled?: boolean
  maxLength?: number
  showCharCount?: boolean
  className?: string
}

export function Composer({
  value,
  onChange,
  onSubmit,
  placeholder = 'Type a message...',
  disabled = false,
  maxLength = 4000,
  showCharCount = false,
  className,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [rows, setRows] = useState(1)

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      // Reset height to measure scrollHeight accurately
      textareaRef.current.style.height = 'auto'
      const scrollHeight = textareaRef.current.scrollHeight
      const lineHeight = 24 // Approximate line height in pixels
      const newRows = Math.min(Math.max(Math.ceil(scrollHeight / lineHeight), 1), 10)
      setRows(newRows)
    }
  }, [value])

  // Focus textarea on mount
  useEffect(() => {
    if (textareaRef.current && !disabled) {
      textareaRef.current.focus()
    }
  }, [disabled])

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (value.trim() && !disabled) {
        onSubmit()
      }
    }
  }

  const handleSubmit = () => {
    if (value.trim() && !disabled) {
      onSubmit()
    }
  }

  const charCount = value.length
  const isOverLimit = charCount > maxLength
  const canSend = value.trim().length > 0 && !disabled && !isOverLimit

  return (
    <div className={cn('border-t bg-background p-4', className)}>
      <div className="mx-auto max-w-3xl space-y-2">
        <div className="relative flex items-end gap-2">
          {/* Textarea */}
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={rows}
            maxLength={maxLength}
            className={cn(
              'min-h-[60px] resize-none',
              isOverLimit && 'border-destructive focus-visible:ring-destructive'
            )}
          />

          {/* Send Button */}
          <Button
            onClick={handleSubmit}
            disabled={!canSend}
            size="icon"
            className="h-10 w-10 shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {/* Character Count */}
        {showCharCount && (
          <div
            className={cn(
              'text-right text-xs',
              isOverLimit ? 'text-destructive' : 'text-muted-foreground'
            )}
          >
            {charCount} / {maxLength}
          </div>
        )}

        {/* Help Text */}
        <div className="text-xs text-muted-foreground">
          Press <kbd className="rounded bg-muted px-1 py-0.5 font-mono">Enter</kbd> to send,{' '}
          <kbd className="rounded bg-muted px-1 py-0.5 font-mono">Shift+Enter</kbd> for new line
        </div>
      </div>
    </div>
  )
}
