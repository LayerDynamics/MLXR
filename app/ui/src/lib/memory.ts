/**
 * Memory and performance utilities for MLXR
 * Provides utilities for monitoring memory usage, performance tracking, and optimization
 */

/**
 * Memory info structure
 */
export interface MemoryInfo {
  usedJSHeapSize?: number
  totalJSHeapSize?: number
  jsHeapSizeLimit?: number
  timestamp: number
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  navigationTiming?: PerformanceNavigationTiming
  resourceTimings?: PerformanceResourceTiming[]
  marks?: PerformanceMark[]
  measures?: PerformanceMeasure[]
  memory?: MemoryInfo
  timestamp: number
}

/**
 * Get current memory usage
 * Returns memory info if available, null otherwise
 */
export function getMemoryInfo(): MemoryInfo | null {
  if (typeof performance === 'undefined') {
    return null
  }

  const memory = (performance as Performance & { memory?: {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
  }}).memory

  if (!memory) {
    return null
  }

  return {
    usedJSHeapSize: memory.usedJSHeapSize,
    totalJSHeapSize: memory.totalJSHeapSize,
    jsHeapSizeLimit: memory.jsHeapSizeLimit,
    timestamp: Date.now(),
  }
}

/**
 * Format memory size to human-readable string
 */
export function formatMemorySize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }

  return `${size.toFixed(2)} ${units[unitIndex]}`
}

/**
 * Get memory usage percentage
 */
export function getMemoryUsagePercentage(): number | null {
  const memory = getMemoryInfo()
  if (!memory || !memory.usedJSHeapSize || !memory.jsHeapSizeLimit) {
    return null
  }

  return (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
}

/**
 * Check if memory usage is high (>80%)
 */
export function isMemoryHigh(): boolean {
  const percentage = getMemoryUsagePercentage()
  return percentage !== null && percentage > 80
}

/**
 * Get performance metrics
 */
export function getPerformanceMetrics(): PerformanceMetrics {
  const metrics: PerformanceMetrics = {
    timestamp: Date.now(),
  }

  if (typeof performance === 'undefined') {
    return metrics
  }

  // Navigation timing
  const navTiming = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined
  if (navTiming) {
    metrics.navigationTiming = navTiming
  }

  // Resource timings
  const resourceTimings = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
  if (resourceTimings.length > 0) {
    metrics.resourceTimings = resourceTimings
  }

  // Performance marks
  const marks = performance.getEntriesByType('mark') as PerformanceMark[]
  if (marks.length > 0) {
    metrics.marks = marks
  }

  // Performance measures
  const measures = performance.getEntriesByType('measure') as PerformanceMeasure[]
  if (measures.length > 0) {
    metrics.measures = measures
  }

  // Memory info
  const memory = getMemoryInfo()
  if (memory) {
    metrics.memory = memory
  }

  return metrics
}

/**
 * Create a performance mark
 */
export function mark(name: string): void {
  if (typeof performance !== 'undefined' && performance.mark) {
    performance.mark(name)
  }
}

/**
 * Create a performance measure between two marks
 */
export function measure(name: string, startMark: string, endMark?: string): number | null {
  if (typeof performance === 'undefined' || !performance.measure) {
    return null
  }

  try {
    if (endMark) {
      performance.measure(name, startMark, endMark)
    } else {
      performance.measure(name, startMark)
    }

    const measures = performance.getEntriesByName(name, 'measure')
    if (measures.length > 0) {
      return measures[measures.length - 1].duration
    }
  } catch (error) {
    console.warn('Failed to create performance measure:', error)
  }

  return null
}

/**
 * Clear performance marks and measures
 */
export function clearPerformanceEntries(type?: 'mark' | 'measure'): void {
  if (typeof performance === 'undefined') {
    return
  }

  if (type === 'mark') {
    performance.clearMarks()
  } else if (type === 'measure') {
    performance.clearMeasures()
  } else {
    performance.clearMarks()
    performance.clearMeasures()
  }
}

/**
 * Time a function execution
 */
export async function timeFunction<T>(
  fn: () => T | Promise<T>,
  label?: string
): Promise<{ result: T; duration: number }> {
  const startMark = `${label || 'fn'}-start`
  const endMark = `${label || 'fn'}-end`
  const measureName = label || 'fn-execution'

  mark(startMark)
  const result = await fn()
  mark(endMark)

  const duration = measure(measureName, startMark, endMark) || 0

  return { result, duration }
}

/**
 * Monitor memory usage over time
 */
export class MemoryMonitor {
  private interval: ReturnType<typeof setInterval> | null = null
  private samples: MemoryInfo[] = []
  private maxSamples: number

  constructor(maxSamples = 100) {
    this.maxSamples = maxSamples
  }

  /**
   * Start monitoring
   */
  start(intervalMs = 1000): void {
    this.stop()

    this.interval = setInterval(() => {
      const info = getMemoryInfo()
      if (info) {
        this.samples.push(info)
        if (this.samples.length > this.maxSamples) {
          this.samples.shift()
        }
      }
    }, intervalMs)
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.interval) {
      clearInterval(this.interval)
      this.interval = null
    }
  }

  /**
   * Get all samples
   */
  getSamples(): MemoryInfo[] {
    return [...this.samples]
  }

  /**
   * Get latest sample
   */
  getLatest(): MemoryInfo | null {
    return this.samples.length > 0 ? this.samples[this.samples.length - 1] : null
  }

  /**
   * Get average memory usage
   */
  getAverage(): number | null {
    if (this.samples.length === 0) {
      return null
    }

    const sum = this.samples.reduce((acc, sample) => {
      return acc + (sample.usedJSHeapSize || 0)
    }, 0)

    return sum / this.samples.length
  }

  /**
   * Get peak memory usage
   */
  getPeak(): number | null {
    if (this.samples.length === 0) {
      return null
    }

    return Math.max(...this.samples.map(s => s.usedJSHeapSize || 0))
  }

  /**
   * Clear all samples
   */
  clear(): void {
    this.samples = []
  }
}

/**
 * Debounce function to reduce memory pressure from frequent calls
 */
export function createMemoryEfficientDebounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): ((...args: Parameters<T>) => void) & { cancel: () => void } {
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  let lastArgs: Parameters<T> | null = null

  const debounced = (...args: Parameters<T>) => {
    lastArgs = args

    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    timeoutId = setTimeout(() => {
      if (lastArgs) {
        fn(...lastArgs)
        lastArgs = null
      }
      timeoutId = null
    }, delay)
  }

  debounced.cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId)
      timeoutId = null
    }
    lastArgs = null
  }

  return debounced
}

/**
 * Create a memoized function with memory-aware cache
 */
export function memoize<T extends (...args: unknown[]) => unknown>(
  fn: T,
  options: {
    maxSize?: number
    keyFn?: (...args: Parameters<T>) => string
  } = {}
): T {
  const { maxSize = 100, keyFn } = options
  const cache = new Map<string, ReturnType<T>>()

  return ((...args: Parameters<T>) => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args)

    if (cache.has(key)) {
      return cache.get(key)
    }

    const result = fn(...args) as ReturnType<T>
    cache.set(key, result)

    // Evict oldest entry if cache is full
    if (cache.size > maxSize) {
      const firstKey = cache.keys().next().value
      if (firstKey !== undefined) {
        cache.delete(firstKey)
      }
    }

    return result
  }) as T
}

/**
 * Create a weak memoized function that allows garbage collection
 */
export function weakMemoize<T extends object, R>(
  fn: (arg: T) => R
): (arg: T) => R {
  const cache = new WeakMap<T, R>()

  return (arg: T) => {
    if (cache.has(arg)) {
      return cache.get(arg)!
    }

    const result = fn(arg)
    cache.set(arg, result)
    return result
  }
}

/**
 * Request garbage collection (if available)
 * Note: This is only available in some environments (e.g., Node.js with --expose-gc)
 */
export function requestGarbageCollection(): void {
  const globalThis_ = typeof globalThis !== 'undefined' ? globalThis : (typeof window !== 'undefined' ? window : undefined)
  if (globalThis_ && typeof (globalThis_ as unknown as { gc?: () => void }).gc === 'function') {
    ((globalThis_ as unknown) as { gc: () => void }).gc()
  } else {
    console.warn('Garbage collection is not exposed. Run with --expose-gc flag.')
  }
}

/**
 * Estimate object size in bytes (rough approximation)
 */
export function estimateObjectSize(obj: unknown): number {
  const seen = new WeakSet()

  function sizeOf(value: unknown): number {
    if (value === null || value === undefined) {
      return 0
    }

    const type = typeof value
    switch (type) {
      case 'boolean':
        return 4
      case 'number':
        return 8
      case 'string':
        return (value as string).length * 2
      case 'object': {
        if (seen.has(value as object)) {
          return 0
        }
        seen.add(value as object)

        let size = 0
        if (Array.isArray(value)) {
          value.forEach(item => {
            size += sizeOf(item)
          })
        } else {
          Object.keys(value as object).forEach(key => {
            size += key.length * 2
            size += sizeOf((value as Record<string, unknown>)[key])
          })
        }
        return size
      }
      default:
        return 0
    }
  }

  return sizeOf(obj)
}

/**
 * Check if an object is too large (>1MB)
 */
export function isObjectTooLarge(obj: unknown, thresholdBytes = 1024 * 1024): boolean {
  return estimateObjectSize(obj) > thresholdBytes
}

/**
 * Create a memory-efficient pool for reusable objects
 */
export class ObjectPool<T> {
  private pool: T[] = []
  private factory: () => T
  private reset: (obj: T) => void
  private maxSize: number

  constructor(factory: () => T, reset: (obj: T) => void, maxSize = 50) {
    this.factory = factory
    this.reset = reset
    this.maxSize = maxSize
  }

  /**
   * Acquire an object from the pool
   */
  acquire(): T {
    return this.pool.pop() || this.factory()
  }

  /**
   * Release an object back to the pool
   */
  release(obj: T): void {
    if (this.pool.length < this.maxSize) {
      this.reset(obj)
      this.pool.push(obj)
    }
  }

  /**
   * Get current pool size
   */
  size(): number {
    return this.pool.length
  }

  /**
   * Clear the pool
   */
  clear(): void {
    this.pool = []
  }
}

/**
 * Get long task entries (tasks >50ms)
 */
export function getLongTasks(): PerformanceEntry[] {
  if (typeof performance === 'undefined') {
    return []
  }

  const entries = performance.getEntriesByType('longtask')
  return entries
}

/**
 * Monitor for long tasks
 */
export function observeLongTasks(callback: (entry: PerformanceEntry) => void): () => void {
  if (typeof PerformanceObserver === 'undefined') {
    console.warn('PerformanceObserver not supported')
    return () => {}
  }

  const observer = new PerformanceObserver((list) => {
    list.getEntries().forEach(callback)
  })

  try {
    observer.observe({ entryTypes: ['longtask'] })
  } catch (error) {
    console.warn('Failed to observe long tasks:', error)
  }

  return () => observer.disconnect()
}

export default {
  getMemoryInfo,
  formatMemorySize,
  getMemoryUsagePercentage,
  isMemoryHigh,
  getPerformanceMetrics,
  mark,
  measure,
  clearPerformanceEntries,
  timeFunction,
  MemoryMonitor,
  createMemoryEfficientDebounce,
  memoize,
  weakMemoize,
  requestGarbageCollection,
  estimateObjectSize,
  isObjectTooLarge,
  ObjectPool,
  getLongTasks,
  observeLongTasks,
}
