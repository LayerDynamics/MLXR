/**
 * RegistryTable Component
 *
 * Table view of all models with:
 * - Sortable columns
 * - Selection for bulk actions
 * - Status indicators
 * - Action buttons per row
 */

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ArrowUpDown, Play, Trash2, Eye } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ModelInfo } from '@/types/backend'

export interface RegistryTableProps {
  models: ModelInfo[]
  selectedIds?: Set<string>
  onSelectModel?: (modelId: string, selected: boolean) => void
  onSelectAll?: (selected: boolean) => void
  onViewDetails?: (modelId: string) => void
  onLoad?: (modelId: string) => void
  onDelete?: (modelId: string) => void
  sortBy?: keyof ModelInfo
  sortOrder?: 'asc' | 'desc'
  onSort?: (column: keyof ModelInfo) => void
  className?: string
}

export function RegistryTable({
  models,
  selectedIds = new Set(),
  onSelectModel,
  onSelectAll,
  onViewDetails,
  onLoad,
  onDelete,
  onSort,
  className,
}: RegistryTableProps) {
  const formatSize = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024)
    return `${gb.toFixed(1)} GB`
  }

  const formatDate = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleDateString()
  }

  const allSelected = models.length > 0 && models.every((m) => selectedIds.has(m.model_id))
  const allChecked = allSelected

  const handleSort = (column: keyof ModelInfo) => {
    onSort?.(column)
  }

  return (
    <div className={cn('rounded-md border', className)}>
      <Table>
        <TableHeader>
          <TableRow>
            {onSelectModel && (
              <TableHead className="w-12">
                <Checkbox
                  checked={allChecked}
                  onCheckedChange={(checked) => onSelectAll?.(checked === true)}
                />
              </TableHead>
            )}
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleSort('name')}
                className="h-8 px-2"
              >
                Name
                <ArrowUpDown className="ml-2 h-3 w-3" />
              </Button>
            </TableHead>
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleSort('architecture')}
                className="h-8 px-2"
              >
                Architecture
                <ArrowUpDown className="ml-2 h-3 w-3" />
              </Button>
            </TableHead>
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleSort('file_size')}
                className="h-8 px-2"
              >
                Size
                <ArrowUpDown className="ml-2 h-3 w-3" />
              </Button>
            </TableHead>
            <TableHead>Quantization</TableHead>
            <TableHead>Format</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Last Used</TableHead>
            <TableHead className="w-24">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={onSelectModel ? 9 : 8}
                className="h-24 text-center text-muted-foreground"
              >
                No models found
              </TableCell>
            </TableRow>
          ) : (
            models.map((model) => (
              <TableRow key={model.model_id}>
                {onSelectModel && (
                  <TableCell>
                    <Checkbox
                      checked={selectedIds.has(model.model_id)}
                      onCheckedChange={(checked) =>
                        onSelectModel(model.model_id, checked === true)
                      }
                    />
                  </TableCell>
                )}
                <TableCell className="font-medium">{model.name}</TableCell>
                <TableCell className="text-muted-foreground">
                  {model.architecture}
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {formatSize(model.file_size)}
                </TableCell>
                <TableCell>
                  <Badge variant="outline">{model.quant_type}</Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {model.format}
                </TableCell>
                <TableCell>
                  {model.is_loaded ? (
                    <Badge variant="default">Loaded</Badge>
                  ) : (
                    <Badge variant="outline">Available</Badge>
                  )}
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {model.last_used_timestamp > 0
                    ? formatDate(model.last_used_timestamp)
                    : '-'}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    {onViewDetails && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onViewDetails(model.model_id)}
                        className="h-7 px-2"
                      >
                        <Eye className="h-3 w-3" />
                      </Button>
                    )}
                    {!model.is_loaded && onLoad && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onLoad(model.model_id)}
                        className="h-7 px-2"
                      >
                        <Play className="h-3 w-3" />
                      </Button>
                    )}
                    {onDelete && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDelete(model.model_id)}
                        className="h-7 px-2 text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  )
}
