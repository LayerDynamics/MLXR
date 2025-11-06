/**
 * KeyboardShortcuts Component
 *
 * Display and customize keyboard shortcuts:
 * - List of all available shortcuts
 * - Key combination display
 * - Optional customization
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Keyboard } from 'lucide-react'

export interface KeyboardShortcutsProps {
  className?: string
}

interface Shortcut {
  category: string
  shortcuts: Array<{
    keys: string[]
    description: string
  }>
}

const shortcuts: Shortcut[] = [
  {
    category: 'Global',
    shortcuts: [
      { keys: ['Cmd', 'K'], description: 'Open command palette' },
      { keys: ['Cmd', ','], description: 'Open settings' },
      { keys: ['Cmd', 'N'], description: 'New conversation' },
      { keys: ['Cmd', 'Q'], description: 'Quit app' },
    ],
  },
  {
    category: 'Chat',
    shortcuts: [
      { keys: ['Enter'], description: 'Send message' },
      { keys: ['Shift', 'Enter'], description: 'New line' },
      { keys: ['Esc'], description: 'Cancel streaming' },
      { keys: ['Cmd', 'L'], description: 'Clear conversation' },
    ],
  },
  {
    category: 'Navigation',
    shortcuts: [
      { keys: ['Cmd', '1'], description: 'Go to Chat' },
      { keys: ['Cmd', '2'], description: 'Go to Models' },
      { keys: ['Cmd', '3'], description: 'Go to Playground' },
      { keys: ['Cmd', '4'], description: 'Go to Metrics' },
      { keys: ['Cmd', '5'], description: 'Go to Logs' },
    ],
  },
]

export function KeyboardShortcuts({ className }: KeyboardShortcutsProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Keyboard className="h-5 w-5" />
          <CardTitle className="text-base">Keyboard Shortcuts</CardTitle>
        </div>
        <CardDescription>View all available keyboard shortcuts</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {shortcuts.map((section, index) => (
            <div key={section.category}>
              {index > 0 && <Separator className="my-4" />}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-muted-foreground">
                  {section.category}
                </h4>
                <div className="space-y-2">
                  {section.shortcuts.map((shortcut, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between gap-4 text-sm"
                    >
                      <span>{shortcut.description}</span>
                      <div className="flex gap-1">
                        {shortcut.keys.map((key, keyIndex) => (
                          <span key={keyIndex} className="flex items-center gap-1">
                            {keyIndex > 0 && (
                              <span className="text-muted-foreground">+</span>
                            )}
                            <Badge variant="outline" className="font-mono text-xs">
                              {key}
                            </Badge>
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
