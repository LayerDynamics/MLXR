# MLXR React Frontend

Modern React application for the MLXR inference engine. Provides a comprehensive UI for chat, model management, metrics, logs, and system configuration.

## Status: ✅ COMPLETE

All 43 components implemented and building successfully (as of 2025-11-06).

See [docs/FRONTEND_COMPLETION.md](../../docs/FRONTEND_COMPLETION.md) for detailed documentation.

## Quick Start

### Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
# Opens at http://localhost:5173
```

### Production Build

```bash
# Build for production
npm run build
# Output: dist/

# Preview production build
npm run preview
```

### Type Checking & Linting

```bash
# Check types
npm run type-check

# Lint code
npm run lint

# Format code
npm run format
```

## Tech Stack

- **React** 18.3 + TypeScript 5.4
- **Vite** 5.2 for build tooling
- **TailwindCSS** 3.4 + **shadcn/ui** components
- **Zustand** for UI state, **TanStack Query** for server state
- **React Router** for navigation
- **cmdk** for command palette

## Project Structure

```
src/
├── components/        # React components (43 total)
│   ├── chat/         # Chat interface (10 components)
│   ├── model/        # Model management (7 components)
│   ├── settings/     # Settings panels (10 components)
│   ├── metrics/      # Metrics dashboard (8 components)
│   ├── logs/         # Log viewer (2 components)
│   ├── playground/   # API testing (3 components)
│   ├── layout/       # Navigation & system (3 components)
│   └── ui/           # shadcn/ui primitives
├── pages/            # Page components (6 pages)
├── hooks/            # Custom React hooks (4 hooks)
├── lib/              # Utilities and API client
├── types/            # TypeScript type definitions
└── styles/           # Global styles and Tailwind config
```

## Component Categories

### 1. Chat Components (10)
Message display, composition, streaming, model selection, sampling controls

### 2. Model Components (7)
Registry table, model cards, import/pull dialogs, stats, actions

### 3. Settings Components (10)
Configuration panels, path pickers, daemon control, keyboard shortcuts

### 4. Metrics Components (8)
Live metrics, charts (throughput, latency, KV, kernels), stats cards

### 5. Logs Components (2)
Virtual log viewer with filtering, expandable log entries

### 6. Playground Components (3)
Completion, embeddings, and vision API testing interfaces

### 7. Layout Components (3)
Navigation tabs, command palette (⌘K), system tray popover

## Integration with Backend

### WebView Bridge

Communicates with macOS host via `window.__HOST__`:

```typescript
interface HostBridge {
  request(path: string, init?: RequestInit): Promise<Response>
  openPathDialog(type: 'models' | 'cache'): Promise<string>
  readConfig(): Promise<string>
  writeConfig(yaml: string): Promise<void>
  startDaemon(): Promise<void>
  stopDaemon(): Promise<void>
  getVersion(): Promise<{ app: string; daemon: string }>
}
```

### API Client

All backend communication goes through `src/lib/api.ts` which:
- Uses WebView bridge for HTTP requests
- Handles SSE streaming for chat
- Provides type-safe methods
- Implements error handling

### Hooks

Custom hooks wrap TanStack Query for data fetching:

- `useMetrics()` - Real-time metrics with auto-refresh
- `useDaemon()` - Daemon status and control
- `useModels()` - Model registry queries
- `useConfig()` - Server configuration

## Type Safety

All backend types are defined in `src/types/`:

- **backend.ts** - C++ daemon structures (ModelInfo, SchedulerStats, etc.)
- **metrics.ts** - Metrics and telemetry types
- **logs.ts** - Log entry types
- **bridge.ts** - WebView bridge interface
- **openai.ts** - OpenAI API compatible types

## Build Output

Production build creates:
- `dist/index.html` - Entry point
- `dist/assets/` - Chunked JS and CSS
- **Total size**: 406KB (~130KB gzipped)
- **Chunks**: 21 files with vendor splitting

## Development

### Adding a New Component

1. Create component file in appropriate category folder
2. Import shadcn/ui primitives as needed
3. Define TypeScript interfaces for props
4. Use hooks for data fetching (TanStack Query)
5. Export from index file if needed

### Installing shadcn/ui Components

```bash
# Use shadcn CLI (not manual creation)
npx shadcn@latest add [component-name]

# Example
npx shadcn@latest add button
npx shadcn@latest add dialog
```

### Adding a New Page

1. Create page component in `src/pages/`
2. Add route in `src/App.tsx`
3. Add navigation item in `src/components/layout/Navigation.tsx`
4. Add command in `src/components/layout/CommandPalette.tsx`

## Testing

Testing setup to be implemented:

- **Unit tests**: Vitest + React Testing Library
- **E2E tests**: Playwright
- **Type checking**: `npm run type-check`

## Contributing

1. Follow existing component patterns
2. Use TypeScript strictly (no `any`)
3. Use shadcn/ui components for UI primitives
4. Follow TailwindCSS utility-first approach
5. Keep components small and focused
6. Use hooks for side effects and data fetching

## Next Steps

### macOS Integration (Priority)
- Swift/ObjC host application
- WebView bridge implementation
- Tray and dock integration
- Load `dist/` bundle in WebView

### Enhanced Features
- Replace placeholder charts with Recharts
- Add real-time data streaming for charts
- Implement comprehensive testing
- Add accessibility improvements

## Resources

- [MLXR Documentation](../../docs/)
- [Frontend Completion Details](../../docs/FRONTEND_COMPLETION.md)
- [Implementation Status](../../docs/IMPLEMENTATION_STATUS.md)
- [shadcn/ui](https://ui.shadcn.com/)
- [TanStack Query](https://tanstack.com/query)
- [TailwindCSS](https://tailwindcss.com/)

## License

See repository root for license information.
