MLXR Frontend Complete Implementation Plan
Scope: Build production-ready React + TypeScript frontend for macOS WebView, following FrontendPlan.md specification with ZERO placeholders.
Architecture Decisions (Based on 2024-2025 Best Practices)
Core Stack
React 18.3+ with TypeScript 5.4+
Vite 5.2+ for build tooling
TailwindCSS 3.4+ with shadcn/ui components
Zustand 4.5+ for UI/client state
TanStack Query 5.28+ for server state
TanStack Virtual for list virtualization (modern replacement for react-window)
Framer Motion 11+ for animations
Recharts 2.12+ for metrics visualization
react-i18next 14.1+ for internationalization
Key Patterns
Bridge: Type-safe WKWebView messaging with dev fallback
SSE: Custom hooks with exponential backoff (1s→30s, 5 retries max)
State Separation: Zustand for UI state, TanStack Query for server data
Performance: <8ms frame budget, route code splitting, virtualization
Accessibility: WCAG AA compliance, keyboard shortcuts
Implementation Tasks (105 Total)
Phase 1: Project Configuration (10 tasks)
Task 1: Create package.json with complete dependencies
{
  "name": "@mlxr/ui",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:e2e": "playwright test",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.22.3",
    "zustand": "^4.5.2",
    "@tanstack/react-query": "^5.28.9",
    "@tanstack/react-virtual": "^3.2.0",
    "framer-motion": "^11.0.24",
    "recharts": "^2.12.2",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "react-syntax-highlighter": "^15.5.0",
    "yaml": "^2.4.1",
    "date-fns": "^3.6.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.2",
    "class-variance-authority": "^0.7.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-switch": "^1.0.3",
    "@radix-ui/react-checkbox": "^1.0.4",
    "@radix-ui/react-slider": "^1.1.2",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-separator": "^1.0.3",
    "@radix-ui/react-slot": "^1.0.2",
    "cmdk": "^1.0.0",
    "sonner": "^1.4.41",
    "react-i18next": "^14.1.0",
    "i18next": "^23.10.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.1",
    "@types/react-dom": "^18.3.0",
    "@types/react-syntax-highlighter": "^15.5.11",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.4.5",
    "vite": "^5.2.8",
    "tailwindcss": "^3.4.3",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "vitest": "^1.5.0",
    "@vitest/ui": "^1.5.0",
    "@testing-library/react": "^15.0.2",
    "@testing-library/jest-dom": "^6.4.2",
    "@testing-library/user-event": "^14.5.2",
    "@playwright/test": "^1.43.1",
    "eslint": "^8.57.0",
    "@typescript-eslint/eslint-plugin": "^7.7.0",
    "@typescript-eslint/parser": "^7.7.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.6",
    "eslint-plugin-jsx-a11y": "^6.8.0",
    "prettier": "^3.2.5",
    "prettier-plugin-tailwindcss": "^0.5.14"
  }
}
Task 2: Create tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
Task 3: Create vite.config.ts with optimizations Task 4: Create tailwind.config.js with shadcn/ui setup Task 5: Create postcss.config.js Task 6: Create .eslintrc.json with React + accessibility rules Task 7: Create .prettierrc with Tailwind plugin Task 8: Create vitest.config.ts for unit testing Task 9: Create playwright.config.ts for E2E testing Task 10: Update public/index.html with proper meta tags
Phase 2: Core Type Definitions (8 tasks)
Task 11: Create types/backend.ts
Mirror C++ structures exactly
ModelFormat, ModelArchitecture, QuantizationType enums
ModelInfo, AdapterInfo interfaces
SchedulerStats, RequestState, FinishReason
Complete type safety for all backend data
Task 12: Create types/openai.ts
ChatCompletionRequest/Response
CompletionRequest/Response
ChatCompletionChunk for streaming
EmbeddingsRequest/Response
OpenAI-compatible DTOs
Task 13: Create types/ollama.ts
OllamaGenerateRequest/Response
OllamaChatRequest/Response
OllamaTagsResponse
OllamaPullRequest with progress
OllamaModelInfo
Task 14: Create types/metrics.ts
MetricsSnapshot structure
Counter, Gauge, Histogram types
PrometheusMetrics format
Real-time metric streaming types
Task 15: Create types/config.ts
ServerConfig structure
SamplingParams interface
ModelConfig, AdapterConfig
YAML schema types
Task 16: Create types/bridge.ts
HostBridge interface
BridgeMessage/BridgeResponse
Error codes and types
Window augmentation
Task 17: Create types/store.ts
AppStore state interfaces
ChatUIStore interfaces
Action types
Persisted state shape
Task 18: Create types/index.ts - Re-export all types
Phase 3: Core Libraries (12 tasks)
Task 19: Implement lib/bridge.ts
Type-safe BridgeClient class
Message ID generation and tracking
Timeout handling (30s default)
Request/response correlation
Development fallback to localhost:8080
Error handling with specific codes
Window webkit detection
Task 20: Implement lib/sse.ts
useSSE hook with exponential backoff
useTokenStream hook for chat streaming
AbortController integration
Retry logic: 1s→2s→4s→8s→16s→30s (5 retries)
Buffer management for chunked data
[DONE] marker detection
Proper cleanup on unmount
Task 21: Implement lib/api.ts
MLXRAPIClient class with all endpoints
OpenAI API methods (chat, completion, embeddings)
Ollama API methods (generate, chat, tags, pull, ps)
Metrics endpoints (JSON, Prometheus)
Model management (list, get, delete)
Health check endpoint
Request/response interceptors
Error transformation
Task 22: Implement lib/store.ts
Zustand store with slices
UI state: sidebar, theme, compact mode
Session state: active conversation/model
Persistence with localStorage
Selective persistence (only preferences)
Action creators
Type-safe selectors
Task 23: Implement lib/chatUIStore.ts
Separate store for chat UI
Sampling parameters (temp, top_p, max_tokens)
isComposing, showTokens flags
Draft message state
Model selector state
Task 24: Implement lib/queryClient.ts
TanStack Query configuration
Retry strategy: 3 retries with exponential backoff
StaleTime: 5 minutes
GcTime: 30 minutes
Query key factory pattern
Mutation defaults
Task 25: Implement lib/theme.ts
useTheme hook
System preference detection
localStorage persistence
CSS class management
MediaQuery listener for system changes
Task 26: Implement lib/utils.ts
cn() for className merging
formatBytes() for file sizes
formatDuration() for timing
formatNumber() with locale
debounce() and throttle()
copyToClipboard()
Task 27: Implement lib/markdown.ts
React Markdown configuration
remark-gfm plugin setup
Syntax highlighting for code blocks
Custom renderers for components
Link handling
Task 28: Implement lib/i18n.ts
i18next initialization
en-US translations
Language detection
Namespace structure
Translation key extraction
Task 29: Implement lib/keyboard.ts
useKeyboardShortcuts hook
Command palette (⌘K)
Import model (⌘I)
Toggle logs (⌘/)
Global shortcut registration
Conflict prevention
Task 30: Implement lib/memory.ts
useMemoryMonitor hook
useChatMemoryCleanup hook
useFrameBudget debouncing
Performance monitoring utilities
Phase 4: Common UI Components (shadcn/ui) (18 tasks)
Task 31: Create components/ui/button.tsx
Radix Slot for asChild pattern
Variants: default, destructive, outline, ghost, link
Sizes: sm, default, lg, icon
Loading state support
Disabled state styling
Task 32: Create components/ui/input.tsx
Controlled input with ref forwarding
Error state styling
Prefix/suffix icon support
Type-safe HTML input props
Task 33: Create components/ui/textarea.tsx
Auto-resize capability
Character count display
Max length enforcement
Task 34: Create components/ui/dialog.tsx
Radix Dialog primitive
Overlay with backdrop blur
Close on escape
Trap focus
Slide-in animation
Task 35: Create components/ui/tooltip.tsx
Radix Tooltip primitive
Positioning: top, right, bottom, left
Delay configuration
Arrow support
Task 36: Create components/ui/select.tsx
Radix Select primitive
Searchable variant
Multi-select support
Group support
Custom trigger
Task 37: Create components/ui/checkbox.tsx
Radix Checkbox primitive
Checked, unchecked, indeterminate states
Label integration
Task 38: Create components/ui/switch.tsx
Radix Switch primitive
On/off states
Label positioning
Task 39: Create components/ui/badge.tsx
Variants: default, secondary, destructive, outline
Sizes: sm, default, lg
Removable variant with X button
Task 40: Create components/ui/card.tsx
Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
Hover effects
Clickable variant
Task 41: Create components/ui/tabs.tsx
Radix Tabs primitive
Horizontal and vertical layouts
Keyboard navigation
Task 42: Create components/ui/slider.tsx
Radix Slider primitive
Single and range values
Step configuration
Value display
Task 43: Create components/ui/separator.tsx
Horizontal and vertical variants
Decorative role
Task 44: Create components/ui/dropdown-menu.tsx
Radix Dropdown primitive
Menu items, separators, labels
Keyboard navigation
Nested submenus
Task 45: Create components/ui/toast.tsx using Sonner
Toast provider setup
Success, error, warning, info variants
Action buttons
Dismiss functionality
Queue management
Task 46: Create components/ui/progress.tsx
Determinate and indeterminate modes
Percentage display
Color variants
Task 47: Create components/ui/skeleton.tsx
Loading placeholder
Pulse animation
Various shapes (text, circle, rectangle)
Task 48: Create components/ui/error-boundary.tsx
Class-based error boundary
Error reporting
Retry functionality
Fallback UI customization
Phase 5: Chat Components (10 tasks)
Task 49: Create components/chat/ChatPane.tsx
Main container layout
Conversation list + message area split
Responsive design
Header with model selector
Task 50: Create components/chat/MessageList.tsx
TanStack Virtual for virtualization
Auto-scroll on new messages
Scroll to bottom button
Dynamic row heights
Overscan: 5 items
Task 51: Create components/chat/Message.tsx
User/assistant/system message variants
Markdown rendering
Code block syntax highlighting
Timestamp display
Copy message button
Regenerate button (assistant only)
Task 52: Create components/chat/Composer.tsx
Textarea with auto-resize
Send button (⌘Enter)
File attachment button (for vision)
Model selector dropdown
Sampling controls panel (expandable)
Draft persistence
Task 53: Create components/chat/TokenStream.tsx
Real-time token display
Cursor effect animation
Streaming indicator
Cancel button
Tokens/second counter
Task 54: Create components/chat/ToolCallView.tsx
Tool name and arguments display
JSON formatting with syntax highlighting
Approval UI (approve/reject buttons)
Execution status indicator
Result display
Task 55: Create components/chat/ConversationList.tsx
Virtualized conversation list
Search/filter input
Create new conversation button
Delete conversation action
Last message preview
Timestamp display
Task 56: Create components/chat/ModelSelector.tsx
Dropdown with model search
Model info display (params, context, quant)
Grouped by family
"Set as default" action
Load/unload indicator
Task 57: Create components/chat/SamplingControls.tsx
Temperature slider (0-2)
Top-p slider (0-1)
Max tokens input
Repetition penalty slider
Preset buttons (creative, balanced, precise)
Task 58: Create components/chat/AttachmentButton.tsx
File picker integration
Image preview
Remove attachment action
File size validation
Phase 6: Models Components (9 tasks)
Task 59: Create components/models/RegistryTable.tsx
TanStack Virtual table
Sortable columns (name, size, family, quant)
Filterable by format, architecture
Row selection
Bulk actions
Task 60: Create components/models/ModelCard.tsx
Model thumbnail/icon
Name, family, architecture
Parameter count, context length
Quantization badge
File size, disk usage
Actions menu (load, delete, export)
Task 61: Create components/models/ModelImport.tsx
Dialog with file picker
Drag-and-drop support
Format detection (GGUF, safetensors, MLX)
Validation feedback
Import progress bar
Error handling
Task 62: Create components/models/QuantBadge.tsx
Color-coded by quant level
Q2_K (red) → Q4_K (yellow) → Q8_K (green) → FP16 (blue)
Tooltip with quant details
Size variants
Task 63: Create components/models/ModelActions.tsx
Action dropdown menu
Load/unload model
Set as default
Test prompt dialog
Convert format
Quantize model
Delete with confirmation
Task 64: Create components/models/AdapterStack.tsx
List of LoRA adapters
Add adapter button
Remove adapter action
Reorder adapters (drag-and-drop)
Adapter scale slider
Enable/disable toggle
Task 65: Create components/models/ModelDetailDrawer.tsx
Slide-out drawer from right
Complete model information
Tokenizer details
RoPE scaling parameters
Chat template preview
Test prompt interface
Adapter management
Task 66: Create components/models/ModelPullDialog.tsx
Model search input
Popular models list
Pull progress with stages
Cancel pull action
Error handling
Task 67: Create components/models/ModelStats.tsx
Model usage statistics
Last used timestamp
Total tokens generated
Average latency
Mini chart of usage over time
Phase 7: Metrics Components (8 tasks)
Task 68: Create components/metrics/StatCards.tsx
Grid layout (2x2 or 3x2)
MetricCard for each stat
Real-time updates via polling
Animated value changes
Task 69: Create components/metrics/MetricCard.tsx
Large value display
Label and unit
Trend indicator (up/down arrow)
Sparkline mini-chart
Color coding (good/warning/bad)
Task 70: Create components/metrics/LatencyChart.tsx
Recharts histogram
Buckets: <10ms, 10-50ms, 50-100ms, 100-200ms, 200+ms
P50, P95, P99 markers
Tooltip with details
Time range selector
Task 71: Create components/metrics/ThroughputChart.tsx
Recharts line chart
Tokens/second over time
Dual axis: prefill vs decode
Zoom/pan support
Export to PNG
Task 72: Create components/metrics/KVChart.tsx
Heatmap visualization
KV block utilization
Color scale: free (green) → full (red)
Block details on hover
Eviction events marked
Task 73: Create components/metrics/KernelTimeChart.tsx
Stacked bar chart
GPU time vs CPU time
Per-kernel breakdown
Sorting by time
Percentage display
Task 74: Create components/metrics/MetricsFilter.tsx
Time range selector (1h, 6h, 24h, 7d)
Metric type filter
Model filter
Export options (JSON, CSV, PNG)
Task 75: Create components/metrics/LiveMetrics.tsx
Real-time metrics stream
Auto-refresh toggle
Refresh rate selector
Pause/resume
Phase 8: Settings Components (10 tasks)
Task 76: Create components/settings/General.tsx
Theme selector (light, dark, system)
Language selector
Launch at login checkbox
Compact mode toggle
Auto-updates toggle
Task 77: Create components/settings/Performance.tsx
Target latency slider (30-150ms)
Max batch tokens input
Max batch size input
Enable speculative decoding toggle
Draft model selector
Speculation length slider
Enable chunked prefill toggle
Prefill chunk size input
Task 78: Create components/settings/Paths.tsx
Models directory path picker
Cache directory path picker
Free space display for each path
Clear cache button with confirmation
Reset to defaults button
Task 79: Create components/settings/Updates.tsx
Current version display
Check for updates button
Auto-check on launch toggle
Release notes display
Update download progress
Install and restart button
Task 80: Create components/settings/Privacy.tsx
Telemetry opt-in toggle
Data collected explanation
Clear telemetry data button
Clear conversation history button
Export data button (JSON)
Task 81: Create components/settings/ConfigEditor.tsx
YAML editor with syntax highlighting
Validation on change
Error display with line numbers
Revert to saved button
Apply changes button
Reset to defaults button
Task 82: Create components/settings/SettingRow.tsx
Reusable setting layout
Label with description
Control slot (input, toggle, slider, etc.)
Help tooltip
Reset to default button
Task 83: Create components/settings/PathPicker.tsx
Input with current path
Browse button (calls bridge.openPathDialog)
Validation (path exists, writable)
Platform-specific path handling
Task 84: Create components/settings/DaemonControl.tsx
Daemon status indicator
Start/stop daemon buttons
Restart daemon button
Daemon logs viewer
PID display
Task 85: Create components/settings/KeyboardShortcuts.tsx
List of all shortcuts
Custom shortcut editor
Conflict detection
Reset to defaults
Phase 9: Logs & Playground (7 tasks)
Task 86: Create components/logs/LogViewer.tsx
TanStack Virtual list
Level filter (debug, info, warn, error)
Search/filter input
Auto-scroll toggle
Clear logs button
Export logs button
Task 87: Create components/logs/LogEntry.tsx
Timestamp display
Level badge (color-coded)
Message with syntax highlighting (JSON)
Copy button
Expandable for long messages
Stack trace display (errors)
Task 88: Create components/playground/EmbeddingsPlayground.tsx
Text input area
Model selector
Generate embeddings button
Vector display (first 10 dims + ...)
Dimension count
Cosine similarity calculator (2 inputs)
Code snippet (curl, JS, Python)
Task 89: Create components/playground/CompletionPlayground.tsx
Prompt input area
Sampling controls
Generate button
Output display
Latency display (TTFT, total time)
Tokens/second display
Token count
Copy output button
Task 90: Create components/playground/VisionPlayground.tsx
Image upload area
Image preview
Prompt input
Model selector (vision-capable only)
Generate button
Output display
Examples gallery
Task 91: Create components/playground/CodeSnippet.tsx
Language tabs (curl, JavaScript, Python)
Syntax-highlighted code
Copy button
Line numbers toggle
Wrap lines toggle
Task 92: Create components/playground/PlaygroundLayout.tsx
Tabbed interface
Embeddings, Completion, Vision tabs
Quick action buttons
History sidebar (optional)
Phase 10: Application Shell (10 tasks)
Task 93: Create components/layout/Layout.tsx
Main app grid layout
Sidebar navigation (left)
Main content area (center)
Status bar (bottom)
Responsive breakpoints
Task 94: Create components/layout/Navigation.tsx
Tab navigation with icons
Active tab indicator
Keyboard shortcuts display
Tooltips for each tab
Badge for notifications (e.g., new logs)
Task 95: Create components/layout/StatusBar.tsx
Daemon status indicator (dot + text)
Current model display
Tokens/second (live)
P95 latency (live)
KV usage percentage
Connection status
Task 96: Create components/layout/CommandPalette.tsx using cmdk
⌘K trigger
Fuzzy search
Actions: navigate, import model, settings, etc.
Recent items
Keyboard navigation
Custom actions registry
Task 97: Create components/layout/Sidebar.tsx
Logo/branding
Navigation items
Collapsible design
User settings dropdown (future)
Help/docs link
Task 98: Create components/layout/Header.tsx
Page title
Breadcrumbs (optional)
Quick actions (page-specific)
Search input (global)
Task 99: Create components/layout/TrayPopover.tsx
Quick status view
Current model
Tokens/s, latency
Context used (%)
Quick actions: start/stop daemon, switch model
Open main window button
Task 100: Create components/layout/ErrorFallback.tsx
Full-page error display
Error message and stack
Retry button
Report issue button (opens GitHub)
Copy error details
Task 101: Create components/layout/LoadingFallback.tsx
Full-page loading spinner
Loading message
Progress indicator (if available)
Cancel button (optional)
Task 102: Create components/layout/NotificationProvider.tsx
Toast notification setup
Global notification state
Error notification on API failure
Success notification on actions
Phase 11: Pages (6 tasks)
Task 103: Create pages/ChatPage.tsx
ChatPane component integration
Layout with conversation list + messages
State management (active conversation)
URL params for conversation ID
New conversation action
Task 104: Create pages/ModelsPage.tsx
RegistryTable component integration
Search and filter controls
Import/pull buttons
Detail drawer integration
Bulk action bar
Task 105: Create pages/PlaygroundsPage.tsx
PlaygroundLayout integration
Tab routing
State persistence per playground
Quick examples
Task 106: Create pages/MetricsPage.tsx
StatCards at top
Charts in grid below
Filters sidebar
Export functionality
Real-time updates
Task 107: Create pages/SettingsPage.tsx
Tabbed settings layout
General, Performance, Paths, Updates, Privacy tabs
Form state management
Save/discard buttons
Unsaved changes warning
Task 108: Create pages/LogsPage.tsx
LogViewer integration
Filters sidebar
Full-height layout
Auto-scroll toggle
Phase 12: Custom Hooks (10 tasks)
Task 109: Create hooks/useChat.ts
Chat state management
Send message function
SSE streaming integration
Message history
Conversation CRUD operations
Task 110: Create hooks/useModels.ts
TanStack Query integration
List models query
Get model query
Delete model mutation
Load/unload model mutations
Task 111: Create hooks/useMetrics.ts
Real-time metrics polling (5s interval)
Metrics history state
Pause/resume polling
Query invalidation on page visibility
Task 112: Create hooks/useDaemon.ts
Daemon status query (1s polling)
Start daemon mutation
Stop daemon mutation
Restart daemon helper
Health check
Task 113: Create hooks/useConfig.ts
Load config query
Save config mutation
YAML parsing/serialization
Validation
Reset to defaults
Task 114: Create hooks/useConversations.ts
List conversations query
Get conversation query
Create conversation mutation
Update conversation mutation
Delete conversation mutation
Task 115: Create hooks/useDebounce.ts
Generic debounce hook
Configurable delay
Cancel on unmount
Task 116: Create hooks/useIntersectionObserver.ts
For lazy loading images
For infinite scroll (future)
Task 117: Create hooks/useLocalStorage.ts
Type-safe localStorage wrapper
SSR-safe
JSON serialization
Task 118: Create hooks/useMediaQuery.ts
Responsive design helper
Breakpoint constants
Phase 13: Styles (3 tasks)
Task 119: Create styles/index.css
Tailwind directives (@import "tailwindcss")
CSS reset/normalize
Global styles
Font imports (Inter, JetBrains Mono)
Focus-visible styles
Selection styles
Task 120: Create styles/theme-variables.css
CSS custom properties for colors
Light theme variables
Dark theme variables
Border radius values
Shadow values
Transition timings
Task 121: Create styles/animations.css
Fade in/out animations
Slide animations
Pulse animation
Spinner animation
Framer Motion variants
Phase 14: Root Files & Bootstrap (4 tasks)
Task 122: Create src/main.tsx
React.StrictMode wrapper
QueryClientProvider
Router setup
i18n initialization
Error boundary
Toaster provider
Theme provider
Global styles import
Task 123: Create src/App.tsx
BrowserRouter
Route definitions
Layout wrapper
Command palette
Keyboard shortcuts setup
404 page
Task 124: Update public/index.html
Meta tags (viewport, charset, description)
Title
Theme color
Apple touch icon
Manifest link
Task 125: Create src/vite-env.d.ts
Vite client types
Custom type augmentations
Window interface extensions
Phase 15: Testing (8 tasks)
Task 126: Create tests/setup.ts
Vitest global setup
Testing Library matchers
Mock bridge
Mock localStorage
Task 127: Write unit tests for lib/bridge.ts
Message sending
Response handling
Timeout behavior
Error cases
Dev fallback
Task 128: Write unit tests for lib/sse.ts
Connection establishment
Message parsing
Retry logic
Cleanup
Task 129: Write unit tests for lib/store.ts
State updates
Persistence
Selectors
Task 130: Write component tests for Chat components
Message rendering
Token streaming
Composer input
Model selector
Task 131: Write component tests for Models components
Table rendering
Filtering
Actions
Task 132: Write E2E tests with Playwright
App startup
Navigation
Chat session
Model import
Settings change
Task 133: Write accessibility tests
Keyboard navigation
Screen reader compatibility
Focus management
ARIA attributes
Phase 16: Build & Integration (5 tasks)
Task 134: Create build scripts
Production build optimization
Source maps
Asset optimization
Bundle size analysis
Task 135: Add Makefile targets
frontend-install:
 cd app/ui && npm install

frontend-dev:
 cd app/ui && npm run dev

frontend-build:
 cd app/ui && npm run build

frontend-test:
 cd app/ui && npm test

frontend-test-e2e:
 cd app/ui && npm run test:e2e

frontend-lint:
 cd app/ui && npm run lint

frontend-format:
 cd app/ui && npm run format
Task 136: Create deployment configuration
Production environment variables
Asset CDN configuration (if needed)
Error reporting setup
Analytics setup (opt-in)
Task 137: Create development documentation
README with setup instructions
Component documentation
API integration guide
Testing guide
Contributing guide
Task 138: Create GitHub Actions workflow
Run tests on PR
Build check
Lint check
Type check
Deliverables
✅ Production-ready React frontend in app/ui/
✅ Complete type system mirroring backend C++ structures
✅ Working WebView bridge with development fallback
✅ Real SSE streaming with exponential backoff retry
✅ Full UI component library based on shadcn/ui
✅ All pages implemented: Chat, Models, Playgrounds, Metrics, Settings, Logs
✅ Comprehensive test coverage (unit + E2E)
✅ Build artifacts in app/ui/dist/ ready for WebView embedding
✅ Documentation for developers
Key Success Criteria
Zero placeholders - every function fully implemented
Type safety - no any types except where truly necessary
Performance - <8ms frame budget, virtualized lists, code splitting
Accessibility - WCAG AA compliant, keyboard navigation
Error handling - graceful degradation, user-friendly messages
Testing - >80% code coverage
Developer experience - works standalone for development
Technical Guarantees
Bridge works both in WebView AND standalone dev mode
SSE streaming handles disconnects and reconnects automatically
State management cleanly separates UI state (Zustand) from server state (TanStack Query)
Virtualization handles 10,000+ messages without performance degradation
All components are accessible via keyboard
Dark mode works perfectly with system preferences
Build output is optimized for WebView embedding (<2MB initial bundle)
