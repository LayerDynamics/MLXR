/**
 * ModelDetailDrawer Component
 *
 * Drawer/sheet showing detailed model information:
 * - Architecture details
 * - Quantization info
 * - File paths and sizes
 * - Metadata and tags
 * - Usage statistics
 * - Actions (load, delete, export)
 */

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import {
  Play,
  Trash2,
  Download,
  HardDrive,
  Layers,
  Tag,
} from 'lucide-react'
import type { ModelInfo } from '@/types/backend'
import { cn } from '@/lib/utils'

export interface ModelDetailDrawerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  model: ModelInfo | null
  onLoad?: (modelId: string) => void
  onDelete?: (modelId: string) => void
  onExport?: (modelId: string) => void
  className?: string
}

export function ModelDetailDrawer({
  open,
  onOpenChange,
  model,
  onLoad,
  onDelete,
  onExport,
  className,
}: ModelDetailDrawerProps) {
  if (!model) {
    return null
  }

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let size = bytes
    let unitIndex = 0

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024
      unitIndex++
    }

    return `${size.toFixed(2)} ${units[unitIndex]}`
  }

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className={cn('w-[400px] sm:w-[540px]', className)}>
        <SheetHeader>
          <SheetTitle>{model.name}</SheetTitle>
          <SheetDescription>
            {model.architecture} - {model.quant_type}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Status and Actions */}
          <div className="flex items-center justify-between">
            <Badge variant={model.is_loaded ? 'default' : 'secondary'}>
              {model.is_loaded ? 'Loaded' : 'Available'}
            </Badge>
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={() => onLoad?.(model.model_id)}
                disabled={model.is_loaded}
              >
                <Play className="mr-2 h-4 w-4" />
                Load
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onExport?.(model.model_id)}
              >
                <Download className="mr-2 h-4 w-4" />
                Export
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => onDelete?.(model.model_id)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <Separator />

          {/* Architecture Details */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Layers className="h-4 w-4" />
              Architecture
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-muted-foreground">Type</div>
                <div className="font-medium">{model.architecture}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Quantization</div>
                <div className="font-medium">{model.quant_type}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Context Length</div>
                <div className="font-medium">{model.context_length.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Layers</div>
                <div className="font-medium">{model.num_layers}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Vocabulary</div>
                <div className="font-medium">{model.vocab_size.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Parameters</div>
                <div className="font-medium">{model.param_count.toLocaleString()}</div>
              </div>
            </div>
          </div>

          <Separator />

          {/* File Information */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <HardDrive className="h-4 w-4" />
              File Information
            </div>
            <div className="space-y-2 text-sm">
              <div>
                <div className="text-muted-foreground">Format</div>
                <div className="font-medium">{model.format}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Size</div>
                <div className="font-medium">{formatBytes(model.file_size)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Path</div>
                <div className="font-mono text-xs break-all">
                  {model.file_path}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">Created</div>
                <div className="font-medium">{formatDate(model.created_timestamp)}</div>
              </div>
            </div>
          </div>

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <>
              <Separator />
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Tag className="h-4 w-4" />
                  Tags
                </div>
                <div className="flex flex-wrap gap-2">
                  {model.tags.map((tag: string) => (
                    <Badge key={tag} variant="outline">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
