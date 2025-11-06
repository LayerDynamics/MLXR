/**
 * Keyboard shortcuts and hotkey management for MLXR
 * Provides a centralized system for registering and handling keyboard shortcuts
 */

import { useEffect } from 'react'
import { useKeyboardShortcutsStore } from './store'

export interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  meta?: boolean // Command on Mac
  alt?: boolean
  shift?: boolean
  description: string
  handler: () => void
}

/**
 * Parse keyboard event to shortcut string
 * Format: "Ctrl+Shift+K" or "Meta+K" or "Alt+A"
 */
export function parseKeyboardEvent(event: KeyboardEvent): string {
  const parts: string[] = []

  if (event.ctrlKey) parts.push('Ctrl')
  if (event.metaKey) parts.push('Meta')
  if (event.altKey) parts.push('Alt')
  if (event.shiftKey) parts.push('Shift')

  // Normalize key name
  const key = event.key.length === 1 ? event.key.toUpperCase() : event.key

  // Filter out modifier keys themselves
  if (!['Control', 'Meta', 'Alt', 'Shift'].includes(key)) {
    parts.push(key)
  }

  return parts.join('+')
}

/**
 * Create shortcut string from components
 */
export function createShortcutKey(options: {
  key: string
  ctrl?: boolean
  meta?: boolean
  alt?: boolean
  shift?: boolean
}): string {
  const parts: string[] = []

  if (options.ctrl) parts.push('Ctrl')
  if (options.meta) parts.push('Meta')
  if (options.alt) parts.push('Alt')
  if (options.shift) parts.push('Shift')
  parts.push(options.key.toUpperCase())

  return parts.join('+')
}

/**
 * Format shortcut for display
 * Converts "Meta+K" to "K" on Mac, "Ctrl+K" on other platforms
 */
export function formatShortcutForDisplay(shortcut: string): string {
  const isMac = typeof navigator !== 'undefined' && navigator.userAgent.toUpperCase().includes('MAC')

  let formatted = shortcut
    .replace(/Meta/g, isMac ? '' : 'Ctrl')
    .replace(/Ctrl/g, isMac ? '' : 'Ctrl')
    .replace(/Alt/g, isMac ? '%' : 'Alt')
    .replace(/Shift/g, isMac ? '�' : 'Shift')

  // Remove '+' for Mac symbols
  if (isMac) {
    formatted = formatted.replace(/\+/g, '')
  }

  return formatted
}

/**
 * Check if element should ignore keyboard shortcuts
 * Returns true if focus is in an input, textarea, or contenteditable
 */
export function shouldIgnoreShortcut(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) {
    return false
  }

  const tagName = target.tagName.toLowerCase()
  const isContentEditable = target.isContentEditable

  return (
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select' ||
    isContentEditable
  )
}

/**
 * Global keyboard event handler
 */
function handleKeyboardEvent(event: KeyboardEvent) {
  // Ignore shortcuts when typing in inputs
  if (shouldIgnoreShortcut(event.target)) {
    return
  }

  const shortcut = parseKeyboardEvent(event)
  const { shortcuts } = useKeyboardShortcutsStore.getState()
  const handler = shortcuts.get(shortcut)

  if (handler) {
    event.preventDefault()
    event.stopPropagation()
    handler()
  }
}

/**
 * Initialize global keyboard listener
 * Call this once at app startup
 */
export function initializeKeyboardShortcuts() {
  if (typeof window === 'undefined') return

  window.addEventListener('keydown', handleKeyboardEvent)

  return () => {
    window.removeEventListener('keydown', handleKeyboardEvent)
  }
}

/**
 * React hook for registering keyboard shortcuts
 * Automatically cleans up on unmount
 */
export function useKeyboardShortcut(
  shortcut: string | KeyboardShortcut,
  handler?: () => void
) {
  const { registerShortcut, unregisterShortcut } = useKeyboardShortcutsStore()

  const shortcutKey =
    typeof shortcut === 'string'
      ? shortcut
      : createShortcutKey(shortcut)

  const shortcutHandler =
    typeof shortcut === 'string' ? handler : shortcut.handler

  useEffect(() => {
    if (!shortcutHandler) return

    registerShortcut(shortcutKey, shortcutHandler)

    return () => {
      unregisterShortcut(shortcutKey)
    }
  }, [shortcutKey, shortcutHandler, registerShortcut, unregisterShortcut])
}

/**
 * React hook for registering multiple keyboard shortcuts
 */
export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const { registerShortcut, unregisterShortcut } = useKeyboardShortcutsStore()

  useEffect(() => {
    const keys: string[] = []

    shortcuts.forEach(shortcut => {
      const key = createShortcutKey(shortcut)
      keys.push(key)
      registerShortcut(key, shortcut.handler)
    })

    return () => {
      keys.forEach(key => unregisterShortcut(key))
    }
  }, [shortcuts, registerShortcut, unregisterShortcut])
}

/**
 * Default keyboard shortcuts for MLXR
 */
export const DEFAULT_SHORTCUTS: Record<string, Omit<KeyboardShortcut, 'handler'>> = {
  NEW_CHAT: {
    key: 'N',
    meta: true,
    description: 'Create new chat',
  },
  TOGGLE_SIDEBAR: {
    key: 'B',
    meta: true,
    description: 'Toggle sidebar',
  },
  COMMAND_PALETTE: {
    key: 'K',
    meta: true,
    description: 'Open command palette',
  },
  SEARCH: {
    key: 'F',
    meta: true,
    description: 'Search',
  },
  SETTINGS: {
    key: ',',
    meta: true,
    description: 'Open settings',
  },
  REFRESH: {
    key: 'R',
    meta: true,
    shift: true,
    description: 'Refresh data',
  },
  ESCAPE: {
    key: 'Escape',
    description: 'Close dialog/modal',
  },
  SAVE: {
    key: 'S',
    meta: true,
    description: 'Save changes',
  },
  COPY: {
    key: 'C',
    meta: true,
    description: 'Copy',
  },
  PASTE: {
    key: 'V',
    meta: true,
    description: 'Paste',
  },
  UNDO: {
    key: 'Z',
    meta: true,
    description: 'Undo',
  },
  REDO: {
    key: 'Z',
    meta: true,
    shift: true,
    description: 'Redo',
  },
  FOCUS_CHAT_INPUT: {
    key: '/',
    description: 'Focus chat input',
  },
  STOP_GENERATION: {
    key: 'Escape',
    description: 'Stop generation',
  },
  REGENERATE: {
    key: 'R',
    meta: true,
    description: 'Regenerate response',
  },
}

/**
 * Get shortcut key for a named shortcut
 */
export function getShortcutKey(name: keyof typeof DEFAULT_SHORTCUTS): string {
  const shortcut = DEFAULT_SHORTCUTS[name]
  return createShortcutKey(shortcut)
}

/**
 * Get formatted shortcut for display
 */
export function getShortcutDisplay(name: keyof typeof DEFAULT_SHORTCUTS): string {
  const key = getShortcutKey(name)
  return formatShortcutForDisplay(key)
}

/**
 * Check if keyboard shortcut is pressed in event
 */
export function isShortcutPressed(
  event: KeyboardEvent,
  shortcut: string | KeyboardShortcut
): boolean {
  const key = typeof shortcut === 'string' ? shortcut : createShortcutKey(shortcut)
  return parseKeyboardEvent(event) === key
}

/**
 * Prevent default for specific shortcuts
 */
export function preventDefaultShortcuts(shortcuts: string[]) {
  const handler = (event: KeyboardEvent) => {
    const pressed = parseKeyboardEvent(event)
    if (shortcuts.includes(pressed)) {
      event.preventDefault()
    }
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }
}

/**
 * Hook for handling Escape key
 */
export function useEscapeKey(handler: () => void) {
  useKeyboardShortcut('Escape', handler)
}

/**
 * Hook for handling Enter key
 */
export function useEnterKey(handler: () => void, withMeta = false) {
  const shortcut = withMeta ? 'Meta+Enter' : 'Enter'
  useKeyboardShortcut(shortcut, handler)
}

/**
 * Hook for handling Command/Ctrl + K (Command Palette)
 */
export function useCommandPalette(handler: () => void) {
  useKeyboardShortcut(getShortcutKey('COMMAND_PALETTE'), handler)
}

export default {
  parseKeyboardEvent,
  createShortcutKey,
  formatShortcutForDisplay,
  shouldIgnoreShortcut,
  initializeKeyboardShortcuts,
  useKeyboardShortcut,
  useKeyboardShortcuts,
  DEFAULT_SHORTCUTS,
  getShortcutKey,
  getShortcutDisplay,
  isShortcutPressed,
  preventDefaultShortcuts,
  useEscapeKey,
  useEnterKey,
  useCommandPalette,
}
