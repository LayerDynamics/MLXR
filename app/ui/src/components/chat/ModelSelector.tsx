/**
 * ModelSelector Component
 *
 * Dropdown to select active model with:
 * - Model list from registry
 * - Model info (size, quantization)
 * - Loading state
 * - Empty state
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useAppStore } from '@/lib/store'
import { useModels } from '@/hooks/useModels'
import { Loader2 } from 'lucide-react'

export interface ModelSelectorProps {
  className?: string
}

export function ModelSelector({ className }: ModelSelectorProps) {
  const { activeModelId, setActiveModel } = useAppStore()
  const { models, isLoading } = useModels()

  const handleValueChange = (value: string) => {
    setActiveModel(value)
  }

  // Helper to format file size to GB
  const formatSizeGB = (bytes: number): string => {
    return (bytes / (1024 * 1024 * 1024)).toFixed(1)
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading models...
      </div>
    )
  }

  if (!models || models.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No models available. Please import a model first.
      </div>
    )
  }

  return (
    <Select value={activeModelId || ''} onValueChange={handleValueChange}>
      <SelectTrigger className={className}>
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.model_id} value={model.model_id}>
            <div className="flex flex-col">
              <span className="font-medium">{model.name}</span>
              <span className="text-xs text-muted-foreground">
                {model.architecture} • {model.quant_type}
                {model.file_size > 0 && ` • ${formatSizeGB(model.file_size)}GB`}
              </span>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
