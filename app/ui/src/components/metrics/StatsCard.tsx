/**
 * StatsCard Component
 *
 * Display a single metric/stat with:
 * - Label and value
 * - Optional icon
 * - Trend indicator
 * - Compact card layout
 */

import { Card, CardContent } from '@/components/ui/card'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'

export interface StatsCardProps {
  label: string
  value: string | number
  icon?: LucideIcon
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  className?: string
}

export function StatsCard({
  label,
  value,
  icon: Icon,
  trend,
  trendValue,
  className,
}: StatsCardProps) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-3 w-3 text-green-500" />
      case 'down':
        return <TrendingDown className="h-3 w-3 text-red-500" />
      case 'neutral':
        return <Minus className="h-3 w-3 text-muted-foreground" />
      default:
        return null
    }
  }

  return (
    <Card className={cn('', className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between space-x-4">
          <div className="flex-1 space-y-1">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold">{value}</p>
            {trendValue && (
              <div className="flex items-center gap-1 text-xs">
                {getTrendIcon()}
                <span className="text-muted-foreground">{trendValue}</span>
              </div>
            )}
          </div>
          {Icon && (
            <div className="rounded-full bg-primary/10 p-3">
              <Icon className="h-5 w-5 text-primary" />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
