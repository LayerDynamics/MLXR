/// <reference types="vite/client" />

/**
 * Vite Environment Type Definitions
 *
 * Extends Vite's built-in types with MLXR-specific environment variables
 */

interface ImportMetaEnv {
  readonly MODE: string
  readonly BASE_URL: string
  readonly PROD: boolean
  readonly DEV: boolean
  readonly SSR: boolean

  // MLXR-specific environment variables
  readonly VITE_API_BASE_URL?: string
  readonly VITE_WS_BASE_URL?: string
  readonly VITE_DAEMON_PORT?: string
  readonly VITE_ENABLE_DEVTOOLS?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
  readonly hot?: {
    readonly data: unknown
    accept(): void
    accept(cb: (mod: unknown) => void): void
    accept(dep: string, cb: (mod: unknown) => void): void
    accept(deps: readonly string[], cb: (mods: unknown[]) => void): void
    dispose(cb: (data: unknown) => void): void
    decline(): void
    invalidate(): void
    on(event: string, cb: (...args: unknown[]) => void): void
  }
}

// Note: Window.__HOST__ bridge is declared in types/bridge.ts
// We don't re-declare it here to avoid type conflicts

export {}
