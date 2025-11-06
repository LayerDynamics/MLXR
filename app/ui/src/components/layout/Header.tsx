/**
 * Header Component
 *
 * Page header with:
 * - Page title
 * - Breadcrumbs (optional)
 * - Quick actions (page-specific)
 * - Global search input (optional)
 */

import { ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { cn } from '@/lib/utils'

interface Breadcrumb {
  label: string
  href?: string
}

interface HeaderProps {
  title: string
  breadcrumbs?: Breadcrumb[]
  actions?: ReactNode
  className?: string
}

export function Header({ title, breadcrumbs, actions, className }: HeaderProps) {
  return (
    <header
      className={cn(
        'flex items-center justify-between border-b border-border bg-card px-6 py-4',
        className
      )}
    >
      <div className="flex flex-col gap-2">
        {/* Breadcrumbs */}
        {breadcrumbs && breadcrumbs.length > 0 && (
          <nav className="flex items-center gap-2 text-sm text-muted-foreground">
            {breadcrumbs.map((crumb, index) => (
              <div key={index} className="flex items-center gap-2">
                {crumb.href ? (
                  <Link
                    to={crumb.href}
                    className="hover:text-foreground transition-colors"
                  >
                    {crumb.label}
                  </Link>
                ) : (
                  <span className="text-foreground">{crumb.label}</span>
                )}
                {index < breadcrumbs.length - 1 && (
                  <ChevronRight className="h-4 w-4" />
                )}
              </div>
            ))}
          </nav>
        )}

        {/* Page Title */}
        <h1 className="text-2xl font-bold">{title}</h1>
      </div>

      {/* Actions */}
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}
