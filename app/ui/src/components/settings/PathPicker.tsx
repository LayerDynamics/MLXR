/**
 * PathPicker Component
 *
 * Path selection with file/folder browser:
 * - Input field showing current path
 * - Browse button to open native picker
 * - Validation and error display
 */

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { FolderOpen, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface PathPickerProps {
  value: string
  onChange: (path: string) => void
  placeholder?: string
  type?: 'models' | 'cache'
  disabled?: boolean
  error?: string
  className?: string
}

export function PathPicker({
  value,
  onChange,
  placeholder = '/path/to/directory',
  type = 'models',
  disabled = false,
  error,
  className,
}: PathPickerProps) {
  const [isLoading, setIsLoading] = useState(false)

  const handleBrowse = async () => {
    setIsLoading(true)
    try {
      const result = await window.__HOST__?.openPathDialog(type)
      if (result) {
        onChange(result)
      }
    } catch (err) {
      console.error('Failed to open path dialog:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          className={cn(error && 'border-destructive')}
        />
        <Button
          variant="outline"
          size="icon"
          onClick={handleBrowse}
          disabled={disabled || isLoading}
        >
          <FolderOpen className="h-4 w-4" />
        </Button>
      </div>
      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-3 w-3" />
          {error}
        </div>
      )}
    </div>
  )
}
