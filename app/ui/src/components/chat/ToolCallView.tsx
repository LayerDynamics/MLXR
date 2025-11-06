/**
 * ToolCallView Component
 *
 * Displays tool/function calls in chat messages:
 * - Function name and arguments
 * - Collapsible details
 * - Syntax highlighting for JSON arguments
 */

import { useState } from 'react'
import { ChevronDown, ChevronRight, Wrench } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { ToolCall } from '@/types/openai'

export interface ToolCallViewProps {
  toolCalls: ToolCall[]
  className?: string
}

export function ToolCallView({ toolCalls, className }: ToolCallViewProps) {
  if (!toolCalls || toolCalls.length === 0) {
    return null
  }

  return (
    <div className={cn('space-y-2', className)}>
      {toolCalls.map((toolCall) => (
        <ToolCallItem key={toolCall.id} toolCall={toolCall} />
      ))}
    </div>
  )
}

interface ToolCallItemProps {
  toolCall: ToolCall
}

function ToolCallItem({ toolCall }: ToolCallItemProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const { function: fn } = toolCall

  // Parse arguments if they're a string
  let parsedArgs: unknown
  try {
    parsedArgs = typeof fn.arguments === 'string'
      ? JSON.parse(fn.arguments)
      : fn.arguments
  } catch {
    parsedArgs = fn.arguments
  }

  return (
    <div className="rounded-lg border bg-muted/30 p-3">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsExpanded(!isExpanded)}
        className="h-auto w-full justify-start p-0 font-normal"
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <Wrench className="h-4 w-4 text-primary" />
          <span className="font-mono text-sm font-medium">{fn.name}</span>
        </div>
      </Button>

      {isExpanded && (
        <div className="mt-2 pl-6">
          <div className="rounded-md bg-background p-3">
            <div className="text-xs font-medium text-muted-foreground mb-2">
              Arguments:
            </div>
            <pre className="overflow-x-auto text-xs">
              <code>{JSON.stringify(parsedArgs, null, 2)}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
