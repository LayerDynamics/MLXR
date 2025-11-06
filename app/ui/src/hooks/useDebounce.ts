/**
 * useDebounce Hook
 *
 * Debounces a value with configurable delay
 * - Delays updating the value until after the delay has passed
 * - Cancels pending updates on unmount
 * - Useful for search inputs, API calls, etc.
 */

import { useEffect, useState } from 'react'

/**
 * Hook to debounce a value
 *
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 500)
 * @returns The debounced value
 *
 * @example
 * ```tsx
 * const [searchQuery, setSearchQuery] = useState('')
 * const debouncedQuery = useDebounce(searchQuery, 500)
 *
 * useEffect(() => {
 *   // This will only run 500ms after the user stops typing
 *   if (debouncedQuery) {
 *     fetchSearchResults(debouncedQuery)
 *   }
 * }, [debouncedQuery])
 * ```
 */
export function useDebounce<T>(value: T, delay = 500): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    // Set up the timeout
    const timeoutId = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    // Clean up the timeout if value changes or component unmounts
    return () => {
      clearTimeout(timeoutId)
    }
  }, [value, delay])

  return debouncedValue
}

/**
 * Hook to create a debounced callback function
 *
 * @param callback - The function to debounce
 * @param delay - Delay in milliseconds (default: 500)
 * @returns The debounced function
 *
 * @example
 * ```tsx
 * const handleSearch = useDebouncedCallback((query: string) => {
 *   fetchSearchResults(query)
 * }, 500)
 *
 * return <input onChange={(e) => handleSearch(e.target.value)} />
 * ```
 */
export function useDebouncedCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  delay = 500
): (...args: Parameters<T>) => void {
  const [timeoutId, setTimeoutId] = useState<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    // Clean up on unmount
    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [timeoutId])

  return (...args: Parameters<T>) => {
    // Clear existing timeout
    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    // Set new timeout
    const newTimeoutId = setTimeout(() => {
      callback(...args)
    }, delay)

    setTimeoutId(newTimeoutId)
  }
}
