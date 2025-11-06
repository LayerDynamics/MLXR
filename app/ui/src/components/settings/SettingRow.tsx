/**
 * SettingRow Component
 *
 * Reusable row layout for settings:
 * - Label and description
 * - Control on the right (switch, select, input, etc.)
 * - Consistent spacing and alignment
 */

import { cn } from '@/lib/utils'

export interface SettingRowProps {
  label: string
  description?: string
  children: React.ReactNode
  className?: string
}

export function SettingRow({
  label,
  description,
  children,
  className,
}: SettingRowProps) {
  return (
    <div className={cn('flex items-center justify-between gap-4 py-4', className)}>
      <div className="flex-1 space-y-1">
        <div className="text-sm font-medium leading-none">{label}</div>
        {description && (
          <div className="text-sm text-muted-foreground">{description}</div>
        )}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  )
}
