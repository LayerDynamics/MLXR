/**
 * LogEntry Component
 *
 * Individual log entry with:
 * - Timestamp display
 * - Level badge (color-coded)
 * - Message with syntax highlighting (JSON)
 * - Copy button
 * - Expandable for long messages
 * - Stack trace display (errors)
 */

import { useState } from 'react'
import { LogEntry as LogEntryType, LogLevel } from '@/types/logs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Copy, ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface LogEntryProps {
  entry: LogEntryType
  className?: string
}

const levelColors: Record<LogLevel, string> = {
  [LogLevel.DEBUG]: 'bg-gray-500 text-white',
  [LogLevel.INFO]: 'bg-blue-500 text-white',
  [LogLevel.WARN]: 'bg-yellow-500 text-black',
  [LogLevel.ERROR]: 'bg-red-500 text-white',
}

export function LogEntry({ entry, className }: LogEntryProps) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const date = new Date(entry.timestamp)
  const timeStr = date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
  const ms = String(date.getMilliseconds()).padStart(3, '0')
  const timestamp = `${timeStr}.${ms}`

  const handleCopy = () => {
    const text = JSON.stringify(entry, null, 2)
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Check if message is JSON
  const isJson = entry.message.trim().startsWith('{') || entry.message.trim().startsWith('[')
  const hasExtras = entry.context || entry.stack_trace

  return (
    <div
      className={cn(
        'group flex gap-3 border-b py-2 px-3 hover:bg-muted/50 font-mono text-xs',
        className
      )}
    >
      <div className="flex-shrink-0 text-muted-foreground">{timestamp}</div>
      <Badge className={cn('flex-shrink-0', levelColors[entry.level])}>{entry.level.toUpperCase()}</Badge>
      {entry.source && (
        <div className="flex-shrink-0 text-muted-foreground">[{entry.source}]</div>
      )}
      <div className="flex-1 min-w-0">
        <div className="flex items-start gap-2">
          {hasExtras && (
            <Button
              variant="ghost"
              size="sm"
              className="h-4 w-4 p-0"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
            </Button>
          )}
          <div className={cn('flex-1', isJson ? 'text-primary' : '')}>
            {entry.message}
          </div>
        </div>

        {expanded && entry.context && (
          <div className="mt-2 pl-6">
            <div className="text-xs text-muted-foreground mb-1">Context:</div>
            <pre className="bg-muted p-2 rounded text-xs overflow-x-auto">
              {JSON.stringify(entry.context, null, 2)}
            </pre>
          </div>
        )}

        {expanded && entry.stack_trace && (
          <div className="mt-2 pl-6">
            <div className="text-xs text-muted-foreground mb-1">Stack Trace:</div>
            <pre className="bg-muted p-2 rounded text-xs overflow-x-auto text-red-600 dark:text-red-400">
              {entry.stack_trace}
            </pre>
          </div>
        )}
      </div>
      <Button
        variant="ghost"
        size="sm"
        className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
        onClick={handleCopy}
      >
        <Copy className="h-3 w-3" />
        {copied && <span className="sr-only">Copied</span>}
      </Button>
    </div>
  )
}
