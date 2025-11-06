/**
 * CommandPalette Component
 *
 * Command palette using cmdk with:
 * - ⌘K trigger
 * - Fuzzy search
 * - Actions: navigate, import model, settings, etc.
 * - Recent items
 * - Keyboard navigation
 */

import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Command } from 'cmdk'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import {
  MessageSquare,
  Package,
  Settings,
  BarChart3,
  ScrollText,
  FlaskConical,
  Download,
  Upload,
  Power,
  RefreshCw,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface CommandAction {
  id: string
  label: string
  description?: string
  icon: React.ComponentType<{ className?: string }>
  category: string
  action: () => void
  keywords?: string[]
}

export interface CommandPaletteProps {
  className?: string
}

export function CommandPalette({ className }: CommandPaletteProps) {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()

  // Handle keyboard shortcut
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((open) => !open)
      }
    }

    document.addEventListener('keydown', down)
    return () => document.removeEventListener('keydown', down)
  }, [])

  const commands: CommandAction[] = [
    {
      id: 'nav-chat',
      label: 'Go to Chat',
      icon: MessageSquare,
      category: 'Navigation',
      action: () => {
        navigate('/')
        setOpen(false)
      },
      keywords: ['chat', 'conversation', 'talk'],
    },
    {
      id: 'nav-models',
      label: 'Go to Models',
      icon: Package,
      category: 'Navigation',
      action: () => {
        navigate('/models')
        setOpen(false)
      },
      keywords: ['models', 'registry', 'import'],
    },
    {
      id: 'nav-playground',
      label: 'Go to Playground',
      icon: FlaskConical,
      category: 'Navigation',
      action: () => {
        navigate('/playground')
        setOpen(false)
      },
      keywords: ['playground', 'test', 'experiment'],
    },
    {
      id: 'nav-metrics',
      label: 'Go to Metrics',
      icon: BarChart3,
      category: 'Navigation',
      action: () => {
        navigate('/metrics')
        setOpen(false)
      },
      keywords: ['metrics', 'stats', 'performance'],
    },
    {
      id: 'nav-logs',
      label: 'Go to Logs',
      icon: ScrollText,
      category: 'Navigation',
      action: () => {
        navigate('/logs')
        setOpen(false)
      },
      keywords: ['logs', 'debug', 'errors'],
    },
    {
      id: 'nav-settings',
      label: 'Go to Settings',
      icon: Settings,
      category: 'Navigation',
      action: () => {
        navigate('/settings')
        setOpen(false)
      },
      keywords: ['settings', 'preferences', 'config'],
    },
    {
      id: 'model-import',
      label: 'Import Model',
      description: 'Import a local model file',
      icon: Upload,
      category: 'Models',
      action: () => {
        navigate('/models?action=import')
        setOpen(false)
      },
      keywords: ['import', 'add', 'upload'],
    },
    {
      id: 'model-pull',
      label: 'Pull Model',
      description: 'Download model from registry',
      icon: Download,
      category: 'Models',
      action: () => {
        navigate('/models?action=pull')
        setOpen(false)
      },
      keywords: ['pull', 'download', 'fetch'],
    },
    {
      id: 'daemon-restart',
      label: 'Restart Daemon',
      description: 'Restart the inference daemon',
      icon: RefreshCw,
      category: 'System',
      action: () => {
        // TODO: Implement daemon restart
        setOpen(false)
      },
      keywords: ['restart', 'reload', 'daemon'],
    },
    {
      id: 'daemon-stop',
      label: 'Stop Daemon',
      description: 'Stop the inference daemon',
      icon: Power,
      category: 'System',
      action: () => {
        // TODO: Implement daemon stop
        setOpen(false)
      },
      keywords: ['stop', 'shutdown', 'daemon'],
    },
  ]

  const categories = Array.from(new Set(commands.map((c) => c.category)))

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={cn(
          'inline-flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground',
          'border rounded-md hover:bg-accent hover:text-accent-foreground transition-colors',
          className
        )}
      >
        <span>Search...</span>
        <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded">⌘K</kbd>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="p-0 max-w-2xl">
          <Command className="rounded-lg border shadow-md">
            <Command.Input
              placeholder="Type a command or search..."
              className="border-none focus:ring-0 h-12 px-4"
            />
            <Command.List className="max-h-[400px] overflow-y-auto p-2">
              <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
                No results found.
              </Command.Empty>

              {categories.map((category) => (
                <Command.Group key={category} heading={category} className="mb-2">
                  {commands
                    .filter((cmd) => cmd.category === category)
                    .map((cmd) => {
                      const Icon = cmd.icon
                      return (
                        <Command.Item
                          key={cmd.id}
                          onSelect={() => cmd.action()}
                          className="flex items-center gap-3 px-3 py-2 rounded-md cursor-pointer hover:bg-accent"
                        >
                          <Icon className="h-4 w-4" />
                          <div className="flex-1">
                            <div className="text-sm font-medium">{cmd.label}</div>
                            {cmd.description && (
                              <div className="text-xs text-muted-foreground">{cmd.description}</div>
                            )}
                          </div>
                        </Command.Item>
                      )
                    })}
                </Command.Group>
              ))}
            </Command.List>
          </Command>
        </DialogContent>
      </Dialog>
    </>
  )
}
