/**
 * SSE (Server-Sent Events) streaming implementation with retry logic
 * Supports exponential backoff and proper cleanup
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { getErrorMessage } from './utils'

export interface SSEOptions {
  maxRetries?: number
  initialRetryDelay?: number
  maxRetryDelay?: number
  onOpen?: () => void
  onError?: (error: Error) => void
  onClose?: () => void
}

export interface SSEMessage {
  id?: string
  event?: string
  data: string
}

/**
 * Parse SSE message from raw line
 */
function parseSSELine(line: string): Partial<SSEMessage> | null {
  if (line.startsWith('data: ')) {
    return { data: line.slice(6) }
  } else if (line.startsWith('id: ')) {
    return { id: line.slice(4) }
  } else if (line.startsWith('event: ')) {
    return { event: line.slice(7) }
  }
  return null
}

/**
 * Stream SSE events from a URL
 * @param url - URL to connect to
 * @param options - SSE options
 * @param signal - AbortSignal for cancellation
 * @returns AsyncGenerator of parsed messages
 */
export async function* streamSSE(
  url: string,
  options: RequestInit = {},
  signal?: AbortSignal
): AsyncGenerator<string, void, unknown> {
  const response = await fetch(url, {
    ...options,
    headers: {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
      ...options.headers,
    },
    signal,
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }

  if (!response.body) {
    throw new Error('Response body is null')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      let currentMessage: Partial<SSEMessage> = {}

      for (const line of lines) {
        const trimmedLine = line.trim()

        if (!trimmedLine) {
          // Empty line indicates end of message
          if (currentMessage.data) {
            yield currentMessage.data
          }
          currentMessage = {}
          continue
        }

        const parsed = parseSSELine(trimmedLine)
        if (parsed) {
          Object.assign(currentMessage, parsed)
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * React hook for SSE streaming with automatic retry
 */
export function useSSE<T = unknown>(
  url: string | null,
  options: SSEOptions = {}
) {
  const {
    maxRetries = 5,
    initialRetryDelay = 1000,
    maxRetryDelay = 30000,
    onOpen,
    onError,
    onClose,
  } = options

  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [retryCount, setRetryCount] = useState(0)

  const abortControllerRef = useRef<AbortController | null>(null)
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const scheduleReconnect = useCallback(() => {
    if (retryCount >= maxRetries) {
      setError(new Error(`Max retries (${maxRetries}) exceeded`))
      return
    }

    const delay = Math.min(
      initialRetryDelay * Math.pow(2, retryCount),
      maxRetryDelay
    )

    retryTimeoutRef.current = setTimeout(() => {
      setRetryCount(prev => prev + 1)
    }, delay)
  }, [retryCount, maxRetries, initialRetryDelay, maxRetryDelay])

  const connect = useCallback(async () => {
    if (!url) return

    // Cancel previous connection
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`SSE connection failed: HTTP ${response.status}`)
      }

      if (!response.body) {
        throw new Error('Response body is null')
      }

      setIsConnected(true)
      setRetryCount(0)
      setError(null)
      onOpen?.()

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()

        if (done) {
          setIsConnected(false)
          onClose?.()
          scheduleReconnect()
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentMessage: Partial<SSEMessage> = {}

        for (const line of lines) {
          const trimmedLine = line.trim()

          if (!trimmedLine) {
            if (currentMessage.data) {
              try {
                const parsed = JSON.parse(currentMessage.data)
                setData(parsed)
              } catch {
                // Not JSON, use as-is
                setData(currentMessage.data as T)
              }
            }
            currentMessage = {}
            continue
          }

          const parsed = parseSSELine(trimmedLine)
          if (parsed) {
            Object.assign(currentMessage, parsed)
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return // Intentional abort, don't retry
      }

      const errorObj =
        err instanceof Error ? err : new Error('SSE connection error')
      setError(errorObj)
      setIsConnected(false)
      onError?.(errorObj)
      scheduleReconnect()
    }
  }, [url, onOpen, onError, onClose, scheduleReconnect])

  const disconnect = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current)
      retryTimeoutRef.current = null
    }
    setIsConnected(false)
  }, [])

  const reconnect = useCallback(() => {
    disconnect()
    setRetryCount(0)
    setError(null)
    connect()
  }, [disconnect, connect])

  // Auto-reconnect when retryCount changes
  useEffect(() => {
    if (retryCount > 0) {
      connect()
    }
  }, [retryCount, connect])

  useEffect(() => {
    if (url) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [url, connect, disconnect])

  return {
    data,
    error,
    isConnected,
    retryCount,
    disconnect,
    reconnect,
  }
}

/**
 * Hook for streaming tokens from chat completions
 */
export function useTokenStream(
  url: string | null,
  body: unknown,
  options: SSEOptions = {}
) {
  const [tokens, setTokens] = useState<string[]>([])
  const [fullText, setFullText] = useState('')
  const [isDone, setIsDone] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)

  const startStream = useCallback(async () => {
    if (!url || !body) return

    setTokens([])
    setFullText('')
    setIsDone(false)
    setIsStreaming(true)
    setError(null)

    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: JSON.stringify(body),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      if (!response.body) {
        throw new Error('Response body is null')
      }

      options.onOpen?.()

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulatedText = ''

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done } = await reader.read()

        if (done) {
          setIsDone(true)
          setIsStreaming(false)
          options.onClose?.()
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmedLine = line.trim()

          if (trimmedLine.startsWith('data: ')) {
            const dataStr = trimmedLine.slice(6)

            if (dataStr === '[DONE]') {
              setIsDone(true)
              setIsStreaming(false)
              options.onClose?.()
              continue
            }

            try {
              const data = JSON.parse(dataStr)

              // OpenAI format
              const token =
                data.choices?.[0]?.delta?.content ||
                data.choices?.[0]?.text ||
                ''

              if (token) {
                accumulatedText += token
                setTokens(prev => [...prev, token])
                setFullText(accumulatedText)
              }

              // Check for finish reason
              if (data.choices?.[0]?.finish_reason) {
                setIsDone(true)
                setIsStreaming(false)
                options.onClose?.()
              }
            } catch (err) {
              console.warn('Failed to parse SSE data:', dataStr, err)
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        const errorObj =
          err instanceof Error ? err : new Error('Stream error')
        setError(errorObj)
        options.onError?.(errorObj)
      }
      setIsStreaming(false)
      setIsDone(true)
    }
  }, [url, body, options])

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsStreaming(false)
    setIsDone(true)
  }, [])

  useEffect(() => {
    if (url && body) {
      startStream()
    }

    return () => {
      stopStream()
    }
  }, [url, body, startStream, stopStream])

  return {
    tokens,
    fullText,
    isDone,
    isStreaming,
    error,
    stopStream,
    startStream,
  }
}

/**
 * Lower-level SSE stream function for custom processing
 */
export async function createSSEStream(
  url: string,
  options: RequestInit = {},
  onMessage: (data: string) => void,
  onError?: (error: Error) => void,
  onClose?: () => void
): Promise<() => void> {
  const abortController = new AbortController()

  const stream = async () => {
    try {
      for await (const data of streamSSE(url, options, abortController.signal)) {
        if (data === '[DONE]') {
          onClose?.()
          return
        }
        onMessage(data)
      }
      onClose?.()
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        onError?.(error instanceof Error ? error : new Error(getErrorMessage(error)))
      }
    }
  }

  stream()

  return () => {
    abortController.abort()
  }
}
