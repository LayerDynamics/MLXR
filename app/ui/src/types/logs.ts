/**
 * Log types for daemon log viewer
 */

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export interface LogEntry {
  id: string
  timestamp: number
  level: LogLevel
  message: string
  context?: Record<string, unknown>
  stack_trace?: string
  source?: string
}

export interface LogFilter {
  levels: LogLevel[]
  search: string
  start_time?: number
  end_time?: number
}

export interface LogExportOptions {
  format: 'json' | 'text'
  filter?: LogFilter
}
