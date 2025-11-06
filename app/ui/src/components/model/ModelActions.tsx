/**
 * ModelActions Component
 *
 * Bulk action buttons for selected models:
 * - Load/unload multiple models
 * - Delete selected models
 * - Export selected models
 * - Tag management
 * - Batch operations with confirmation
 */

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  Play,
  Square,
  Trash2,
  Download,
  Tag,
  MoreVertical,
  Loader2,
} from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ModelActionsProps {
  selectedCount: number
  onLoadSelected?: () => Promise<void>
  onUnloadSelected?: () => Promise<void>
  onDeleteSelected?: () => Promise<void>
  onExportSelected?: () => Promise<void>
  onTagSelected?: () => void
  disabled?: boolean
  className?: string
}

export function ModelActions({
  selectedCount,
  onLoadSelected,
  onUnloadSelected,
  onDeleteSelected,
  onExportSelected,
  onTagSelected,
  disabled = false,
  className,
}: ModelActionsProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [actionInProgress, setActionInProgress] = useState<string | null>(null)

  const handleAction = async (
    action: () => Promise<void> | void,
    actionName: string
  ) => {
    setIsLoading(true)
    setActionInProgress(actionName)
    try {
      await action()
    } catch (error) {
      console.error(`${actionName} failed:`, error)
    } finally {
      setIsLoading(false)
      setActionInProgress(null)
    }
  }

  const handleDelete = async () => {
    if (onDeleteSelected) {
      await handleAction(onDeleteSelected, 'delete')
      setShowDeleteDialog(false)
    }
  }

  const isDisabled = disabled || isLoading || selectedCount === 0

  return (
    <>
      <div className={cn('flex items-center gap-2', className)}>
        {/* Selection Count */}
        {selectedCount > 0 && (
          <div className="text-sm text-muted-foreground">
            {selectedCount} selected
          </div>
        )}

        {/* Load Button */}
        <Button
          size="sm"
          variant="outline"
          onClick={() => onLoadSelected && handleAction(onLoadSelected, 'load')}
          disabled={isDisabled}
        >
          {actionInProgress === 'load' ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Play className="mr-2 h-4 w-4" />
          )}
          Load
        </Button>

        {/* Unload Button */}
        <Button
          size="sm"
          variant="outline"
          onClick={() =>
            onUnloadSelected && handleAction(onUnloadSelected, 'unload')
          }
          disabled={isDisabled}
        >
          {actionInProgress === 'unload' ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Square className="mr-2 h-4 w-4" />
          )}
          Unload
        </Button>

        {/* More Actions Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="outline" disabled={isDisabled}>
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={() =>
                onExportSelected && handleAction(onExportSelected, 'export')
              }
            >
              <Download className="mr-2 h-4 w-4" />
              Export Selected
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onTagSelected}>
              <Tag className="mr-2 h-4 w-4" />
              Manage Tags
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => setShowDeleteDialog(true)}
              className="text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Selected
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Selected Models?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete {selectedCount}{' '}
              {selectedCount === 1 ? 'model' : 'models'} from your system. This
              action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {actionInProgress === 'delete' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
