/**
 * MetricsFilter Component
 *
 * Filter controls for metrics:
 * - Time range selector
 * - Model filter
 * - Metric type selector
 * - Refresh controls
 */

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { RotateCw } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface MetricsFilterProps {
  timeRange: string
  onTimeRangeChange: (range: string) => void
  onRefresh?: () => void
  className?: string
}

export function MetricsFilter({
  timeRange,
  onTimeRangeChange,
  onRefresh,
  className,
}: MetricsFilterProps) {
  return (
    <div className={cn('flex items-center gap-3', className)}>
      <Select value={timeRange} onValueChange={onTimeRangeChange}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select time range" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="1h">Last hour</SelectItem>
          <SelectItem value="6h">Last 6 hours</SelectItem>
          <SelectItem value="24h">Last 24 hours</SelectItem>
          <SelectItem value="7d">Last 7 days</SelectItem>
          <SelectItem value="30d">Last 30 days</SelectItem>
        </SelectContent>
      </Select>

      {onRefresh && (
        <Button variant="outline" size="icon" onClick={onRefresh}>
          <RotateCw className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
