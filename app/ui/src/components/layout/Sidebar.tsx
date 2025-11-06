/**
 * Sidebar Component
 *
 * Left sidebar with:
 * - Logo/branding
 * - Navigation items
 * - Collapsible design
 * - Help/docs link
 */

import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  MessageSquare,
  Database,
  Activity,
  Settings,
  FileText,
  Beaker,
  ChevronLeft,
  ChevronRight,
  HelpCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

interface NavItem {
  label: string
  path: string
  icon: React.ComponentType<{ className?: string }>
  badge?: number
}

const navItems: NavItem[] = [
  { label: 'Chat', path: '/', icon: MessageSquare },
  { label: 'Models', path: '/models', icon: Database },
  { label: 'Playground', path: '/playground', icon: Beaker },
  { label: 'Metrics', path: '/metrics', icon: Activity },
  { label: 'Logs', path: '/logs', icon: FileText },
  { label: 'Settings', path: '/settings', icon: Settings },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <aside
      className={cn(
        'flex flex-col border-r border-border bg-card transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo/Branding */}
      <div className="flex h-16 items-center justify-between border-b border-border px-4">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-md bg-primary" />
            <span className="text-xl font-bold">MLXR</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className={cn('h-8 w-8', collapsed && 'mx-auto')}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 space-y-1 p-2">
        <TooltipProvider>
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path

            const linkContent = (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  'hover:bg-accent hover:text-accent-foreground',
                  isActive && 'bg-accent text-accent-foreground',
                  collapsed && 'justify-center'
                )}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!collapsed && <span>{item.label}</span>}
                {!collapsed && item.badge !== undefined && (
                  <span className="ml-auto rounded-full bg-primary px-2 py-0.5 text-xs text-primary-foreground">
                    {item.badge}
                  </span>
                )}
              </Link>
            )

            if (collapsed) {
              return (
                <Tooltip key={item.path} delayDuration={0}>
                  <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                  <TooltipContent side="right">
                    <p>{item.label}</p>
                  </TooltipContent>
                </Tooltip>
              )
            }

            return linkContent
          })}
        </TooltipProvider>
      </nav>

      {/* Help/Docs Link */}
      <div className="border-t border-border p-2">
        <TooltipProvider>
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                className={cn(
                  'w-full justify-start gap-3',
                  collapsed && 'justify-center px-0'
                )}
                onClick={() => {
                  window.open('https://github.com/anthropics/mlxr/docs', '_blank')
                }}
              >
                <HelpCircle className="h-5 w-5 flex-shrink-0" />
                {!collapsed && <span>Help & Docs</span>}
              </Button>
            </TooltipTrigger>
            {collapsed && (
              <TooltipContent side="right">
                <p>Help & Docs</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>
    </aside>
  )
}
