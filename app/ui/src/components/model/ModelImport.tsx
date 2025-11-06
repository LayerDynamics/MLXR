/**
 * ModelImport Component
 *
 * Dialog for importing local model files:
 * - File/folder picker
 * - Format detection (GGUF, safetensors, MLX)
 * - Optional conversion
 * - Progress tracking
 */

import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { FolderOpen, FileCode, AlertCircle } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

export interface ModelImportProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImport?: (options: ImportOptions) => Promise<void>
}

export interface ImportOptions {
  path: string
  format: 'gguf' | 'safetensors' | 'mlx'
  name?: string
  convert?: boolean
}

export function ModelImport({ open, onOpenChange, onImport }: ModelImportProps) {
  const [path, setPath] = useState('')
  const [format, setFormat] = useState<ImportOptions['format']>('gguf')
  const [name, setName] = useState('')
  const [isImporting, setIsImporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handleBrowse = async () => {
    // Use the host bridge to open file picker
    try {
      const result = await window.__HOST__?.openPathDialog('models')
      if (result) {
        setPath(result)
        // Try to extract name from path
        const fileName = result.split('/').pop()?.replace(/\.[^.]+$/, '') || ''
        setName(fileName)

        // Detect format from extension
        if (result.endsWith('.gguf')) {
          setFormat('gguf')
        } else if (result.endsWith('.safetensors')) {
          setFormat('safetensors')
        } else if (result.endsWith('.mlx')) {
          setFormat('mlx')
        }
      }
    } catch (err) {
      setError('Failed to open file picker')
    }
  }

  const handleImport = async () => {
    if (!path || !name) {
      setError('Please provide a path and name')
      return
    }

    setIsImporting(true)
    setError(null)
    setProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90))
      }, 500)

      await onImport?.({
        path,
        format,
        name,
      })

      clearInterval(progressInterval)
      setProgress(100)

      // Close dialog after success
      setTimeout(() => {
        onOpenChange(false)
        resetForm()
      }, 500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed')
    } finally {
      setIsImporting(false)
    }
  }

  const resetForm = () => {
    setPath('')
    setFormat('gguf')
    setName('')
    setProgress(0)
    setError(null)
  }

  const handleClose = () => {
    if (!isImporting) {
      onOpenChange(false)
      resetForm()
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Import Model</DialogTitle>
          <DialogDescription>
            Import a local model file or folder into the registry
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* File Path */}
          <div className="space-y-2">
            <Label htmlFor="path">Model Path</Label>
            <div className="flex gap-2">
              <Input
                id="path"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="/path/to/model"
                disabled={isImporting}
              />
              <Button
                variant="outline"
                size="icon"
                onClick={handleBrowse}
                disabled={isImporting}
              >
                <FolderOpen className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Model Name */}
          <div className="space-y-2">
            <Label htmlFor="name">Model Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="llama-2-7b-chat"
              disabled={isImporting}
            />
          </div>

          {/* Format */}
          <div className="space-y-2">
            <Label htmlFor="format">Format</Label>
            <Select
              value={format}
              onValueChange={(value) => setFormat(value as ImportOptions['format'])}
              disabled={isImporting}
            >
              <SelectTrigger id="format">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gguf">
                  <div className="flex items-center gap-2">
                    <FileCode className="h-4 w-4" />
                    GGUF
                  </div>
                </SelectItem>
                <SelectItem value="safetensors">
                  <div className="flex items-center gap-2">
                    <FileCode className="h-4 w-4" />
                    SafeTensors
                  </div>
                </SelectItem>
                <SelectItem value="mlx">
                  <div className="flex items-center gap-2">
                    <FileCode className="h-4 w-4" />
                    MLX Native
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Progress */}
          {isImporting && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Importing...</span>
                <span className="text-muted-foreground">{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}

          {/* Error */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={isImporting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            disabled={!path || !name || isImporting}
          >
            {isImporting ? 'Importing...' : 'Import'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
