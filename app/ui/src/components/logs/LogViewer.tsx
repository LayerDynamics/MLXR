/**
 * LogViewer Component
 *
 * Main log viewer with:
 * - TanStack Virtual list for performance
 * - Level filter (debug, info, warn, error)
 * - Search/filter input
 * - Auto-scroll toggle
 * - Clear logs button
 * - Export logs button
 */

import { useState, useRef, useEffect } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { LogEntry as LogEntryType, LogLevel, LogFilter } from '@/types/logs'
import { LogEntry } from './LogEntry'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Trash2,
  Download,
  Search,
  Filter,
  ArrowDown,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

export interface LogViewerProps {
  logs: LogEntryType[]
  onClear?: () => void
  onExport?: () => void
  className?: string
}

export function LogViewer({ logs, onClear, onExport, className }: LogViewerProps) {
  const [filter, setFilter] = useState<LogFilter>({
    levels: [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR],
    search: '',
  })
  const [autoScroll, setAutoScroll] = useState(true)
  const parentRef = useRef<HTMLDivElement>(null)

  // Filter logs based on current filter
  const filteredLogs = logs.filter((log) => {
    if (!filter.levels.includes(log.level)) return false
    if (filter.search && !log.message.toLowerCase().includes(filter.search.toLowerCase())) {
      return false
    }
    return true
  })

  const virtualizer = useVirtualizer({
    count: filteredLogs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 35,
    overscan: 10,
  })

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && filteredLogs.length > 0) {
      virtualizer.scrollToIndex(filteredLogs.length - 1, { align: 'end' })
    }
  }, [filteredLogs.length, autoScroll, virtualizer])

  const toggleLevel = (level: LogLevel) => {
    setFilter((prev) => ({
      ...prev,
      levels: prev.levels.includes(level)
        ? prev.levels.filter((l) => l !== level)
        : [...prev.levels, level],
    }))
  }

  const handleExport = () => {
    if (!onExport) {
      // Default export implementation
      const blob = new Blob([JSON.stringify(filteredLogs, null, 2)], {
        type: 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `logs-${new Date().toISOString()}.json`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      onExport()
    }
  }

  const levelCounts = logs.reduce(
    (acc, log) => {
      acc[log.level] = (acc[log.level] || 0) + 1
      return acc
    },
    {} as Record<LogLevel, number>
  )

  return (
    <Card className={cn('flex flex-col', className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Logs</CardTitle>
            <CardDescription>
              {filteredLogs.length} of {logs.length} entries
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
              className={cn(autoScroll && 'bg-primary/10')}
            >
              <ArrowDown className="h-4 w-4 mr-2" />
              Auto-scroll
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="outline" size="sm" onClick={onClear}>
              <Trash2 className="h-4 w-4 mr-2" />
              Clear
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-2 mt-4">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search logs..."
              value={filter.search}
              onChange={(e) =>
                setFilter((prev) => ({ ...prev, search: e.target.value }))
              }
              className="pl-8"
            />
            {filter.search && (
              <Button
                variant="ghost"
                size="sm"
                className="absolute right-1 top-1 h-7 w-7 p-0"
                onClick={() => setFilter((prev) => ({ ...prev, search: '' }))}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Filter className="h-4 w-4 mr-2" />
                Filter
                <Badge variant="secondary" className="ml-2">
                  {filter.levels.length}
                </Badge>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {Object.values(LogLevel).map((level) => (
                <DropdownMenuCheckboxItem
                  key={level}
                  checked={filter.levels.includes(level)}
                  onCheckedChange={() => toggleLevel(level)}
                >
                  {level.toUpperCase()}
                  {levelCounts[level] && (
                    <span className="ml-auto text-xs text-muted-foreground">
                      {levelCounts[level]}
                    </span>
                  )}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden">
        <div
          ref={parentRef}
          className="h-[600px] overflow-auto bg-background"
          onScroll={(e) => {
            // Disable auto-scroll if user scrolls up
            const target = e.target as HTMLDivElement
            const isAtBottom =
              Math.abs(
                target.scrollHeight - target.scrollTop - target.clientHeight
              ) < 10
            if (!isAtBottom && autoScroll) {
              setAutoScroll(false)
            }
          }}
        >
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative',
            }}
          >
            {virtualizer.getVirtualItems().map((virtualItem) => {
              const log = filteredLogs[virtualItem.index]
              return (
                <div
                  key={virtualItem.key}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    transform: `translateY(${virtualItem.start}px)`,
                  }}
                >
                  <LogEntry entry={log} />
                </div>
              )
            })}
          </div>

          {filteredLogs.length === 0 && (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              {logs.length === 0 ? 'No logs yet' : 'No logs match the current filter'}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
