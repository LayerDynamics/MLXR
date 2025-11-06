/**
 * AttachmentButton Component
 *
 * Button for attaching files/images to chat messages:
 * - File picker for images
 * - Preview selected files
 * - Support for vision models
 */

import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Paperclip, X } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface AttachmentButtonProps {
  onAttachment?: (file: File) => void
  accept?: string
  disabled?: boolean
  className?: string
}

export function AttachmentButton({
  onAttachment,
  accept = 'image/*',
  disabled = false,
  className,
}: AttachmentButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      onAttachment?.(file)
    }
  }

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />
      <Button
        variant="ghost"
        size="sm"
        onClick={handleClick}
        disabled={disabled}
        className="h-8"
      >
        <Paperclip className="h-4 w-4" />
      </Button>
      {selectedFile && (
        <div className="flex items-center gap-2 rounded-md bg-muted px-2 py-1 text-xs">
          <span className="truncate max-w-[150px]">{selectedFile.name}</span>
          <button
            onClick={handleClear}
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  )
}
