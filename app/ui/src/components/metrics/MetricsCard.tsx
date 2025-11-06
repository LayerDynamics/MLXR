/**
 * MetricsCard Component
 *
 * Container for grouped metrics with:
 * - Title and description
 * - Multiple stat items
 * - Optional chart/visualization
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export interface MetricItem {
  label: string
  value: string | number
  unit?: string
}

export interface MetricsCardProps {
  title: string
  description?: string
  metrics: MetricItem[]
  chart?: React.ReactNode
  className?: string
}

export function MetricsCard({
  title,
  description,
  metrics,
  chart,
  className,
}: MetricsCardProps) {
  return (
    <Card className={cn('', className)}>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-4">
        {chart && <div className="mb-4">{chart}</div>}

        <div className="grid grid-cols-2 gap-4">
          {metrics.map((metric, index) => (
            <div key={index} className="space-y-1">
              <div className="text-xs text-muted-foreground">{metric.label}</div>
              <div className="text-xl font-semibold">
                {metric.value}
                {metric.unit && (
                  <span className="ml-1 text-sm font-normal text-muted-foreground">
                    {metric.unit}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
