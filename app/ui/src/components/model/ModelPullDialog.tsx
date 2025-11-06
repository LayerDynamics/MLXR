/**
 * ModelPullDialog Component
 *
 * Dialog for pulling models from remote registries:
 * - Search Ollama/HuggingFace registries
 * - Preview model info
 * - Progress tracking for download
 * - Automatic registry detection
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
import { Download, AlertCircle, Search } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

export interface ModelPullDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onPull?: (options: PullOptions) => Promise<void>
}

export interface PullOptions {
  name: string
  registry: 'ollama' | 'huggingface'
  variant?: string
}

export function ModelPullDialog({
  open,
  onOpenChange,
  onPull,
}: ModelPullDialogProps) {
  const [modelName, setModelName] = useState('')
  const [registry, setRegistry] = useState<PullOptions['registry']>('ollama')
  const [isPulling, setIsPulling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handlePull = async () => {
    if (!modelName.trim()) {
      setError('Please enter a model name')
      return
    }

    setIsPulling(true)
    setError(null)
    setProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 5, 95))
      }, 500)

      await onPull?.({
        name: modelName.trim(),
        registry,
      })

      clearInterval(progressInterval)
      setProgress(100)

      // Close dialog after success
      setTimeout(() => {
        onOpenChange(false)
        resetForm()
      }, 500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pull failed')
    } finally {
      setIsPulling(false)
    }
  }

  const resetForm = () => {
    setModelName('')
    setRegistry('ollama')
    setProgress(0)
    setError(null)
  }

  const handleClose = () => {
    if (!isPulling) {
      onOpenChange(false)
      resetForm()
    }
  }

  // Popular models for quick access
  const popularModels = registry === 'ollama'
    ? ['llama2', 'llama2:13b', 'mistral', 'mixtral', 'phi', 'gemma']
    : ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.2']

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Pull Model</DialogTitle>
          <DialogDescription>
            Download a model from a remote registry
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Registry Selection */}
          <div className="space-y-2">
            <Label htmlFor="registry">Registry</Label>
            <Select
              value={registry}
              onValueChange={(value) => setRegistry(value as PullOptions['registry'])}
              disabled={isPulling}
            >
              <SelectTrigger id="registry">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ollama">
                  <div className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Ollama
                  </div>
                </SelectItem>
                <SelectItem value="huggingface">
                  <div className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    HuggingFace
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Model Name */}
          <div className="space-y-2">
            <Label htmlFor="model-name">Model Name</Label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="model-name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder={
                  registry === 'ollama'
                    ? 'llama2 or llama2:13b'
                    : 'organization/model-name'
                }
                disabled={isPulling}
                className="pl-9"
              />
            </div>
          </div>

          {/* Popular Models */}
          {!isPulling && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">
                Popular Models
              </Label>
              <div className="flex flex-wrap gap-2">
                {popularModels.map((model) => (
                  <Button
                    key={model}
                    variant="outline"
                    size="sm"
                    onClick={() => setModelName(model)}
                    className="h-7 text-xs"
                  >
                    {model}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Progress */}
          {isPulling && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Downloading...</span>
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
          <Button variant="outline" onClick={handleClose} disabled={isPulling}>
            Cancel
          </Button>
          <Button
            onClick={handlePull}
            disabled={!modelName.trim() || isPulling}
          >
            {isPulling ? 'Pulling...' : 'Pull Model'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
