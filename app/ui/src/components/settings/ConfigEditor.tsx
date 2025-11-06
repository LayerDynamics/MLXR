/**
 * ConfigEditor Component
 *
 * YAML/JSON config file editor:
 * - Syntax highlighting
 * - Validation
 * - Save/revert actions
 * - Error display
 */

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Save, RotateCcw, AlertCircle, FileCode } from 'lucide-react'
import { useConfig } from '@/hooks/useConfig'

export interface ConfigEditorProps {
  className?: string
}

export function ConfigEditor({ className }: ConfigEditorProps) {
  const { config, updateConfig, isLoading } = useConfig()
  const [editedContent, setEditedContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isDirty, setIsDirty] = useState(false)

  // Initialize edited content when config loads
  useState(() => {
    if (config) {
      setEditedContent(JSON.stringify(config, null, 2))
    }
  })

  const handleChange = (value: string) => {
    setEditedContent(value)
    setIsDirty(true)
    setError(null)
  }

  const handleSave = async () => {
    try {
      // Validate JSON
      const parsed = JSON.parse(editedContent)

      // Save config
      await updateConfig(parsed)
      setIsDirty(false)
      setError(null)
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError(`Invalid JSON: ${err.message}`)
      } else {
        setError(err instanceof Error ? err.message : 'Failed to save config')
      }
    }
  }

  const handleRevert = () => {
    if (config) {
      setEditedContent(JSON.stringify(config, null, 2))
      setIsDirty(false)
      setError(null)
    }
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileCode className="h-5 w-5" />
            <CardTitle className="text-base">Configuration Editor</CardTitle>
          </div>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleRevert}
              disabled={!isDirty || isLoading}
            >
              <RotateCcw className="mr-2 h-3 w-3" />
              Revert
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!isDirty || isLoading}
            >
              <Save className="mr-2 h-3 w-3" />
              Save
            </Button>
          </div>
        </div>
        <CardDescription>
          Edit server configuration (JSON format)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Textarea
            value={editedContent}
            onChange={(e) => handleChange(e.target.value)}
            className="min-h-[400px] font-mono text-sm"
            placeholder="Loading configuration..."
            disabled={isLoading}
          />

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isDirty && !error && (
            <Alert>
              <AlertDescription>
                You have unsaved changes. Click Save to apply them.
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
