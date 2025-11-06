/**
 * Navigation Component
 *
 * Tab navigation with:
 * - Tab navigation with icons
 * - Active tab indicator
 * - Keyboard shortcuts display
 * - Tooltips for each tab
 * - Badge for notifications
 */

import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import {
  MessageSquare,
  Package,
  Settings,
  BarChart3,
  ScrollText,
  FlaskConical,
} from 'lucide-react'

interface NavItem {
  name: string
  path: string
  icon: React.ComponentType<{ className?: string }>
  shortcut?: string
  badge?: number
}

const navItems: NavItem[] = [
  { name: 'Chat', path: '/', icon: MessageSquare, shortcut: '⌘1' },
  { name: 'Models', path: '/models', icon: Package, shortcut: '⌘2' },
  { name: 'Playground', path: '/playground', icon: FlaskConical, shortcut: '⌘3' },
  { name: 'Metrics', path: '/metrics', icon: BarChart3, shortcut: '⌘4' },
  { name: 'Logs', path: '/logs', icon: ScrollText, shortcut: '⌘5' },
  { name: 'Settings', path: '/settings', icon: Settings, shortcut: '⌘,' },
]

export interface NavigationProps {
  className?: string
}

export function Navigation({ className }: NavigationProps) {
  const location = useLocation()

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  return (
    <TooltipProvider delayDuration={300}>
      <nav className={cn('flex items-center gap-1', className)}>
        {navItems.map((item) => {
          const active = isActive(item.path)
          const Icon = item.icon

          return (
            <Tooltip key={item.path}>
              <TooltipTrigger asChild>
                <Link
                  to={item.path}
                  className={cn(
                    'relative flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
                    'hover:bg-accent hover:text-accent-foreground',
                    active
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground'
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                  {item.badge !== undefined && item.badge > 0 && (
                    <Badge
                      variant="destructive"
                      className="ml-auto h-5 w-5 flex items-center justify-center p-0 text-xs"
                    >
                      {item.badge > 99 ? '99+' : item.badge}
                    </Badge>
                  )}
                  {active && (
                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/2 h-0.5 bg-primary rounded-full" />
                  )}
                </Link>
              </TooltipTrigger>
              <TooltipContent>
                <div className="flex items-center gap-2">
                  <span>{item.name}</span>
                  {item.shortcut && (
                    <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded">{item.shortcut}</kbd>
                  )}
                </div>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </nav>
    </TooltipProvider>
  )
}
