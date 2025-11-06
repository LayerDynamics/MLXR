/**
 * useMediaQuery Hook
 *
 * Responsive design helper for matching media queries
 * - SSR-safe
 * - Listens to window resize events
 * - Common breakpoint constants
 */

import { useEffect, useState } from 'react'

/**
 * Common breakpoint values (matching Tailwind defaults)
 */
export const BREAKPOINTS = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const

/**
 * Hook to match a media query
 *
 * @param query - Media query string (e.g., '(min-width: 768px)')
 * @returns Boolean indicating if the media query matches
 *
 * @example
 * ```tsx
 * const isMobile = useMediaQuery('(max-width: 768px)')
 * const isDesktop = useMediaQuery('(min-width: 1024px)')
 * const isDarkMode = useMediaQuery('(prefers-color-scheme: dark)')
 *
 * return (
 *   <div>
 *     {isMobile ? <MobileNav /> : <DesktopNav />}
 *   </div>
 * )
 * ```
 */
export function useMediaQuery(query: string): boolean {
  // Default to false for SSR
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || !window.matchMedia) {
      return
    }

    const mediaQuery = window.matchMedia(query)

    // Set initial value
    setMatches(mediaQuery.matches)

    // Define listener
    const handleChange = (event: MediaQueryListEvent) => {
      setMatches(event.matches)
    }

    // Add listener
    // Use addEventListener if available (modern browsers)
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange)
    } else {
      // Fallback for older browsers (deprecated but needed for compatibility)
      mediaQuery.addListener(handleChange)
    }

    // Cleanup
    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange)
      } else {
        // Fallback for older browsers (deprecated but needed for compatibility)
        mediaQuery.removeListener(handleChange)
      }
    }
  }, [query])

  return matches
}

/**
 * Hook for common breakpoint checks
 *
 * @returns Object with boolean flags for common breakpoints
 *
 * @example
 * ```tsx
 * const { isMobile, isTablet, isDesktop } = useBreakpoint()
 *
 * return (
 *   <div>
 *     {isMobile && <MobileView />}
 *     {isTablet && <TabletView />}
 *     {isDesktop && <DesktopView />}
 *   </div>
 * )
 * ```
 */
export function useBreakpoint() {
  const isMobile = useMediaQuery(`(max-width: ${BREAKPOINTS.md})`)
  const isTablet = useMediaQuery(
    `(min-width: ${BREAKPOINTS.md}) and (max-width: ${BREAKPOINTS.lg})`
  )
  const isDesktop = useMediaQuery(`(min-width: ${BREAKPOINTS.lg})`)
  const isSmall = useMediaQuery(`(max-width: ${BREAKPOINTS.sm})`)
  const isExtraLarge = useMediaQuery(`(min-width: ${BREAKPOINTS['2xl']})`)

  return {
    isMobile,
    isTablet,
    isDesktop,
    isSmall,
    isExtraLarge,
  }
}

/**
 * Hook to check if user prefers reduced motion
 *
 * @returns Boolean indicating if user prefers reduced motion
 */
export function usePrefersReducedMotion(): boolean {
  return useMediaQuery('(prefers-reduced-motion: reduce)')
}

/**
 * Hook to check if user prefers dark mode
 *
 * @returns Boolean indicating if user prefers dark color scheme
 */
export function usePrefersDarkMode(): boolean {
  return useMediaQuery('(prefers-color-scheme: dark)')
}

/**
 * Hook to get current window dimensions
 *
 * @returns Object with width and height
 */
export function useWindowSize() {
  const [size, setSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  })

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return size
}
