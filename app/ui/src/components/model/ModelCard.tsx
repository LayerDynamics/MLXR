/**
 * ModelCard Component
 *
 * Card display for a model with:
 * - Model name and architecture
 * - File size and quantization
 * - Status (loaded/unloaded)
 * - Action buttons
 * - Click to view details
 */

import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Play, Trash2, Download } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ModelInfo } from '@/types/backend'

export interface ModelCardProps {
  model: ModelInfo
  onSelect?: () => void
  onLoad?: () => void
  onDelete?: () => void
  className?: string
}

export function ModelCard({
  model,
  onSelect,
  onLoad,
  onDelete,
  className,
}: ModelCardProps) {
  const formatSize = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024)
    return `${gb.toFixed(1)} GB`
  }

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleDateString()
  }

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-md',
        className
      )}
      onClick={onSelect}
    >
      <div className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold truncate">{model.name}</h3>
            <p className="text-sm text-muted-foreground truncate">
              {model.architecture}
            </p>
          </div>
          {model.is_loaded && (
            <Badge variant="default" className="shrink-0">
              Loaded
            </Badge>
          )}
        </div>

        {/* Details */}
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Download className="h-3 w-3" />
            {formatSize(model.file_size)}
          </span>
          <span>"</span>
          <span>{model.quant_type}</span>
          <span>"</span>
          <span>{model.format}</span>
        </div>

        {/* Tags */}
        {model.tags && model.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {model.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
            {model.tags.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{model.tags.length - 3}
              </Badge>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t">
          <span className="text-xs text-muted-foreground">
            {model.last_used_timestamp > 0
              ? `Used ${formatDate(model.last_used_timestamp)}`
              : `Added ${formatDate(model.created_timestamp)}`}
          </span>
          <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
            {!model.is_loaded && onLoad && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onLoad}
                className="h-7 px-2"
              >
                <Play className="h-3 w-3" />
              </Button>
            )}
            {onDelete && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onDelete}
                className="h-7 px-2 text-destructive hover:text-destructive"
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  )
}
