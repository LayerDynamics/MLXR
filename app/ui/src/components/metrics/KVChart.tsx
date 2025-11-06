/**
 * KVChart Component
 *
 * Chart showing metrics (placeholder for visualization library)
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export interface KVChartProps {
  className?: string
}

export function KVChart({ className }: KVChartProps) {
  return (
    <Card className={cn('', className)}>
      <CardHeader>
        <CardTitle className="text-base">KV</CardTitle>
        <CardDescription>Metrics visualization</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex h-[200px] items-center justify-center rounded-md border border-dashed">
          <div className="text-center text-sm text-muted-foreground">
            <p>KV chart</p>
            <p className="text-xs">(Chart visualization pending)</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
