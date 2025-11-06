/**
 * useIntersectionObserver Hook
 *
 * Observes element visibility in viewport
 * - For lazy loading images
 * - For infinite scroll
 * - For animations on scroll
 * - SSR-safe
 */

import { useEffect, useRef, useState } from 'react'

export interface UseIntersectionObserverOptions {
  /**
   * Root element for intersection. Defaults to viewport.
   */
  root?: Element | null

  /**
   * Margin around root. Can have values similar to CSS margin property.
   * e.g., "10px 20px 30px 40px" (top, right, bottom, left)
   */
  rootMargin?: string

  /**
   * Number between 0 and 1 indicating percentage of target visibility
   * which should trigger the callback. Can also be an array of thresholds.
   */
  threshold?: number | number[]

  /**
   * If true, stops observing after first intersection
   */
  triggerOnce?: boolean

  /**
   * Initial state of isIntersecting (useful for SSR)
   */
  initialIsIntersecting?: boolean
}

/**
 * Hook to observe element intersection with viewport
 *
 * @param options - IntersectionObserver options
 * @returns Tuple of [ref, isIntersecting, entry]
 *
 * @example
 * ```tsx
 * // Lazy load image
 * const [ref, isVisible] = useIntersectionObserver({ threshold: 0.1, triggerOnce: true })
 *
 * return (
 *   <div ref={ref}>
 *     {isVisible && <img src="image.jpg" alt="Lazy loaded" />}
 *   </div>
 * )
 *
 * // Infinite scroll
 * const [ref, isVisible] = useIntersectionObserver({ threshold: 1.0 })
 *
 * useEffect(() => {
 *   if (isVisible) {
 *     loadMoreItems()
 *   }
 * }, [isVisible])
 *
 * return <div ref={ref}>Loading...</div>
 * ```
 */
export function useIntersectionObserver<T extends Element = HTMLDivElement>(
  options: UseIntersectionObserverOptions = {}
): [React.RefObject<T>, boolean, IntersectionObserverEntry | null] {
  const {
    root = null,
    rootMargin = '0px',
    threshold = 0,
    triggerOnce = false,
    initialIsIntersecting = false,
  } = options

  const ref = useRef<T>(null)
  const [isIntersecting, setIsIntersecting] = useState(initialIsIntersecting)
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null)

  useEffect(() => {
    const element = ref.current

    // Check if IntersectionObserver is supported
    if (!element || typeof IntersectionObserver === 'undefined') {
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries
        setEntry(entry)
        setIsIntersecting(entry.isIntersecting)

        // If triggerOnce is true, disconnect after first intersection
        if (entry.isIntersecting && triggerOnce) {
          observer.disconnect()
        }
      },
      {
        root,
        rootMargin,
        threshold,
      }
    )

    observer.observe(element)

    // Cleanup
    return () => {
      observer.disconnect()
    }
  }, [root, rootMargin, threshold, triggerOnce])

  return [ref, isIntersecting, entry]
}

/**
 * Hook to observe multiple elements
 *
 * @param options - IntersectionObserver options
 * @returns Function to create refs and get intersection state
 *
 * @example
 * ```tsx
 * const { observe, intersections } = useIntersectionObserverMultiple()
 *
 * return (
 *   <div>
 *     {items.map((item) => (
 *       <div key={item.id} ref={observe(item.id)}>
 *         {intersections[item.id] ? 'Visible' : 'Hidden'}
 *       </div>
 *     ))}
 *   </div>
 * )
 * ```
 */
export function useIntersectionObserverMultiple(
  options: UseIntersectionObserverOptions = {}
) {
  const {
    root = null,
    rootMargin = '0px',
    threshold = 0,
    triggerOnce = false,
  } = options

  const [intersections, setIntersections] = useState<Record<string, boolean>>({})
  const observerRef = useRef<IntersectionObserver | null>(null)
  const elementsRef = useRef<Map<string, Element>>(new Map())

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') {
      return
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const updates: Record<string, boolean> = {}

        entries.forEach((entry) => {
          const id = entry.target.getAttribute('data-observe-id')
          if (id) {
            updates[id] = entry.isIntersecting

            if (entry.isIntersecting && triggerOnce) {
              observerRef.current?.unobserve(entry.target)
              elementsRef.current.delete(id)
            }
          }
        })

        setIntersections((prev) => ({ ...prev, ...updates }))
      },
      {
        root,
        rootMargin,
        threshold,
      }
    )

    // Observe all existing elements
    elementsRef.current.forEach((element) => {
      observerRef.current?.observe(element)
    })

    return () => {
      observerRef.current?.disconnect()
    }
  }, [root, rootMargin, threshold, triggerOnce])

  const observe = (id: string) => (element: Element | null) => {
    if (!element) {
      // Element unmounted, clean up
      if (elementsRef.current.has(id)) {
        const oldElement = elementsRef.current.get(id)
        if (oldElement) {
          observerRef.current?.unobserve(oldElement)
        }
        elementsRef.current.delete(id)
      }
      return
    }

    // Set the ID as a data attribute so we can identify it in the callback
    element.setAttribute('data-observe-id', id)

    // Store the element
    elementsRef.current.set(id, element)

    // Start observing if observer is ready
    if (observerRef.current) {
      observerRef.current.observe(element)
    }
  }

  return { observe, intersections }
}

/**
 * Hook for infinite scroll functionality
 *
 * @param callback - Function to call when user reaches the bottom
 * @param hasMore - Whether there are more items to load
 * @param options - IntersectionObserver options
 * @returns Ref to attach to the sentinel element
 *
 * @example
 * ```tsx
 * const loadMore = () => {
 *   // Load more items
 * }
 *
 * const sentinelRef = useInfiniteScroll(loadMore, hasMore)
 *
 * return (
 *   <div>
 *     {items.map((item) => <div key={item.id}>{item.name}</div>)}
 *     <div ref={sentinelRef}>Loading...</div>
 *   </div>
 * )
 * ```
 */
export function useInfiniteScroll(
  callback: () => void,
  hasMore: boolean,
  options: UseIntersectionObserverOptions = {}
): React.RefObject<HTMLDivElement> {
  const [ref, isIntersecting] = useIntersectionObserver<HTMLDivElement>({
    threshold: 1.0,
    ...options,
  })

  useEffect(() => {
    if (isIntersecting && hasMore) {
      callback()
    }
  }, [isIntersecting, hasMore, callback])

  return ref
}
