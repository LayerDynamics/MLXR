# MLXR Frontend Component Index

Complete list of all 43 implemented React components.

## Chat Components (10)

### Core Components
- **Message** (`src/components/chat/Message.tsx`)
  - Individual chat message with markdown rendering, code highlighting, streaming support
  - Props: `message`, `isStreaming`

- **MessageList** (`src/components/chat/MessageList.tsx`)
  - Virtualized message list with auto-scroll and infinite scroll
  - Props: `messages`, `onLoadMore`, `hasMore`

- **Composer** (`src/components/chat/Composer.tsx`)
  - Message input area with multi-line support, file attachments, send button
  - Props: `onSend`, `disabled`, `placeholder`

- **TokenStream** (`src/components/chat/TokenStream.tsx`)
  - Real-time token streaming visualization with metrics
  - Props: `tokens`, `tokensPerSecond`

### Supporting Components
- **ChatPane** (`src/components/chat/ChatPane.tsx`)
  - Main chat layout container combining MessageList and Composer
  - Props: `conversationId`

- **ConversationList** (`src/components/chat/ConversationList.tsx`)
  - Sidebar with conversation history, search, new chat button
  - Props: `activeId`, `onSelect`, `onNew`

- **ModelSelector** (`src/components/chat/ModelSelector.tsx`)
  - Dropdown to select active model with model info
  - Props: none (uses global state)

- **SamplingControls** (`src/components/chat/SamplingControls.tsx`)
  - Temperature, top-p, top-k, max tokens sliders
  - Props: `values`, `onChange`

- **AttachmentButton** (`src/components/chat/AttachmentButton.tsx`)
  - File/image attachment button with file picker
  - Props: `onAttach`, `accept`

- **ToolCallView** (`src/components/chat/ToolCallView.tsx`)
  - Display function/tool calls in chat messages
  - Props: `toolCall`

## Model Components (7)

### Core Components
- **RegistryTable** (`src/components/model/RegistryTable.tsx`)
  - Sortable table of all models with selection, search, filtering
  - Props: `models`, `onSelect`, `selectedIds`, `sortBy`, `sortOrder`

- **ModelCard** (`src/components/model/ModelCard.tsx`)
  - Card view of individual model with actions (load, delete, export)
  - Props: `model`, `onAction`

- **ModelImport** (`src/components/model/ModelImport.tsx`)
  - Dialog to import local model files (GGUF/safetensors/MLX)
  - Props: `onImport`, `onCancel`

- **ModelPullDialog** (`src/components/model/ModelPullDialog.tsx`)
  - Pull models from Ollama/HuggingFace with popular models list
  - Props: `onPull`, `onCancel`

### Supporting Components
- **ModelDetailDrawer** (`src/components/model/ModelDetailDrawer.tsx`)
  - Detailed model information drawer with architecture, parameters, etc.
  - Props: `model`, `open`, `onClose`

- **ModelStats** (`src/components/model/ModelStats.tsx`)
  - Usage statistics (requests, tokens, latency, cache hit rate)
  - Props: `stats`

- **ModelActions** (`src/components/model/ModelActions.tsx`)
  - Bulk actions for selected models (delete, export, quantize)
  - Props: `selectedIds`, `onAction`

## Settings Components (10)

### Settings Panels
- **General** (`src/components/settings/General.tsx`)
  - Theme, auto-start, notifications, language settings
  - Props: none (manages own state)

- **Performance** (`src/components/settings/Performance.tsx`)
  - GPU memory limit, batch size, KV cache configuration
  - Props: none (manages own state)

- **Paths** (`src/components/settings/Paths.tsx`)
  - Model storage, cache, logs directory configuration
  - Props: none (manages own state)

- **Updates** (`src/components/settings/Updates.tsx`)
  - Auto-update settings, check for updates, version info
  - Props: none (manages own state)

- **Privacy** (`src/components/settings/Privacy.tsx`)
  - Telemetry settings, data collection, clear data actions
  - Props: none (manages own state)

### Supporting Components
- **SettingRow** (`src/components/settings/SettingRow.tsx`)
  - Reusable setting row layout with label, description, and control
  - Props: `label`, `description`, `children`

- **PathPicker** (`src/components/settings/PathPicker.tsx`)
  - File/folder picker with browser button, uses WebView bridge
  - Props: `value`, `onChange`, `type`

- **ConfigEditor** (`src/components/settings/ConfigEditor.tsx`)
  - JSON/YAML config editor with validation and save/revert
  - Props: `config`, `onSave`, `format`

- **DaemonControl** (`src/components/settings/DaemonControl.tsx`)
  - Start/stop/restart daemon controls with status indicator
  - Props: none (uses daemon hook)

- **KeyboardShortcuts** (`src/components/settings/KeyboardShortcuts.tsx`)
  - Display all keyboard shortcuts organized by category
  - Props: none (static data)

## Metrics Components (8)

### Core Components
- **LiveMetrics** (`src/components/metrics/LiveMetrics.tsx`)
  - Real-time dashboard cards: tokens/s, requests, KV cache, GPU memory
  - Props: `className`

- **StatsCard** (`src/components/metrics/StatsCard.tsx`)
  - Individual metric card with icon, value, trend indicator
  - Props: `label`, `value`, `icon`, `trend`, `trendValue`

- **MetricsCard** (`src/components/metrics/MetricsCard.tsx`)
  - Container for grouped metrics with title and optional chart
  - Props: `title`, `description`, `metrics`, `chart`

- **MetricsFilter** (`src/components/metrics/MetricsFilter.tsx`)
  - Time range selector and refresh rate controls
  - Props: `timeRange`, `onTimeRangeChange`, `refreshRate`, `onRefreshRateChange`

### Chart Components (Placeholders)
- **ThroughputChart** (`src/components/metrics/ThroughputChart.tsx`)
  - Tokens/second over time chart (placeholder)
  - Props: `className`

- **LatencyChart** (`src/components/metrics/LatencyChart.tsx`)
  - TTFT and decode latency histograms (placeholder)
  - Props: `className`

- **KVChart** (`src/components/metrics/KVChart.tsx`)
  - KV cache utilization visualization (placeholder)
  - Props: `className`

- **KernelTimeChart** (`src/components/metrics/KernelTimeChart.tsx`)
  - Metal kernel performance breakdown (placeholder)
  - Props: `className`

## Logs Components (2)

- **LogViewer** (`src/components/logs/LogViewer.tsx`)
  - Main log viewer with virtual scrolling, filtering, search, export
  - Uses TanStack Virtual for performance (10,000+ entries)
  - Props: `logs`, `onClear`, `onExport`

- **LogEntry** (`src/components/logs/LogEntry.tsx`)
  - Individual log entry with timestamp, level badge, expandable context
  - Props: `entry`

## Playground Components (3)

- **CompletionPlayground** (`src/components/playground/CompletionPlayground.tsx`)
  - Test completion API with prompt input, sampling controls, output display
  - Shows TTFT, total time, tokens/s metrics
  - Props: `className`

- **EmbeddingsPlayground** (`src/components/playground/EmbeddingsPlayground.tsx`)
  - Test embeddings API with dual text input, vector display
  - Includes cosine similarity calculator with visual bar
  - Props: `className`

- **VisionPlayground** (`src/components/playground/VisionPlayground.tsx`)
  - Test multimodal models with image upload and prompt
  - Shows response with timing metrics
  - Props: `className`

## Layout Components (3)

- **Navigation** (`src/components/layout/Navigation.tsx`)
  - Top tab navigation with icons, active indicators, keyboard shortcuts
  - Tabs: Chat, Models, Playground, Metrics, Logs, Settings
  - Props: `className`

- **CommandPalette** (`src/components/layout/CommandPalette.tsx`)
  - Global command palette (⌘K) with fuzzy search via cmdk
  - Actions: navigate, import model, pull model, daemon control
  - Props: `className`

- **TrayPopover** (`src/components/layout/TrayPopover.tsx`)
  - System tray quick view with daemon status, metrics, quick actions
  - Props: `onOpenMainWindow`, `className`

## Component Conventions

### Naming
- PascalCase for component files and names
- Props interfaces named `{ComponentName}Props`
- Placed in category folders under `src/components/`

### Props Pattern
```typescript
export interface ComponentProps {
  className?: string  // Always optional for styling
  // ... other props
}

export function Component({ className, ...props }: ComponentProps) {
  return <div className={cn('base-styles', className)}>...</div>
}
```

### State Management
- Local state: `useState` for component-only state
- Global UI state: Zustand (`useAppStore`)
- Server state: TanStack Query hooks (`useMetrics`, `useModels`, etc.)

### Styling
- TailwindCSS utility classes
- `cn()` helper for conditional classes (from `src/lib/utils.ts`)
- shadcn/ui components for primitives

### Type Safety
- All props interfaces defined
- Backend types from `src/types/backend.ts`
- No `any` types used

## Usage Examples

### Chat Message
```tsx
import { Message } from '@/components/chat/Message'

<Message
  message={{
    id: '1',
    role: 'assistant',
    content: 'Hello!',
    timestamp: Date.now(),
  }}
  isStreaming={false}
/>
```

### Model Selector
```tsx
import { ModelSelector } from '@/components/chat/ModelSelector'

// Uses global state internally
<ModelSelector />
```

### Live Metrics
```tsx
import { LiveMetrics } from '@/components/metrics/LiveMetrics'

<LiveMetrics className="my-4" />
```

### Command Palette
```tsx
import { CommandPalette } from '@/components/layout/CommandPalette'

// Trigger with ⌘K
<CommandPalette />
```

## Testing Components

### Unit Tests (To Be Implemented)
```tsx
import { render, screen } from '@testing-library/react'
import { Message } from '@/components/chat/Message'

test('renders message content', () => {
  render(<Message message={mockMessage} />)
  expect(screen.getByText('Hello!')).toBeInTheDocument()
})
```

### Component Documentation
Each component file includes JSDoc comments:
```tsx
/**
 * Message Component
 *
 * Individual chat message with:
 * - Markdown rendering
 * - Code syntax highlighting
 * - Streaming support
 * - Copy button
 */
```

## Component Dependencies

### shadcn/ui Primitives Used
- Button, Input, Textarea, Select, Dialog, Sheet
- Card, Badge, Separator, Tooltip, Popover
- Tabs, Slider, Switch, Checkbox, Label
- Table, Alert, Alert Dialog, Dropdown Menu

### Third-party Libraries
- `lucide-react` - Icons
- `cmdk` - Command palette
- `@tanstack/react-virtual` - Virtual scrolling
- `react-router-dom` - Navigation
- `react-markdown` - Markdown rendering
- `react-syntax-highlighter` - Code highlighting

## Related Documentation

- [Frontend Completion](../../docs/FRONTEND_COMPLETION.md) - Full implementation details
- [Implementation Status](../../docs/IMPLEMENTATION_STATUS.md) - Project status
- [Frontend Plan](../../plan/FrontendImplementation.md) - Original plan
- [Main README](README.md) - Quick start and overview
