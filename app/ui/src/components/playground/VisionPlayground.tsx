/**
 * VisionPlayground Component
 *
 * Test vision/multimodal models
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Loader2, Upload, Image as ImageIcon, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useModels } from '@/hooks/useModels'

interface VisionResult {
  text: string
  total_ms: number
  tokens_generated: number
}

export interface VisionPlaygroundProps {
  className?: string
}

export function VisionPlayground({ className }: VisionPlaygroundProps) {
  const [prompt, setPrompt] = useState('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState<VisionResult | null>(null)
  const { models } = useModels()

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setImageFile(file)
    const reader = new FileReader()
    reader.onloadend = () => {
      setImagePreview(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleClearImage = () => {
    setImageFile(null)
    setImagePreview(null)
  }

  const handleGenerate = async () => {
    if (!prompt.trim() || !selectedModel || !imageFile) return

    setIsGenerating(true)
    const startTime = Date.now()

    try {
      const formData = new FormData()
      formData.append('model', selectedModel)
      formData.append('prompt', prompt)
      formData.append('image', imageFile)

      const response = await fetch('/api/v1/chat/completions', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Generation failed')

      const data = await response.json()
      const totalTime = Date.now() - startTime

      setResult({
        text: data.choices[0].message.content,
        total_ms: totalTime,
        tokens_generated: data.usage.completion_tokens,
      })
    } catch (error) {
      console.error('Generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className={cn('grid gap-4 lg:grid-cols-2', className)}>
      <Card>
        <CardHeader>
          <CardTitle>Vision Playground</CardTitle>
          <CardDescription>Test multimodal models with images</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models?.map((model) => (
                  <SelectItem key={model.model_id} value={model.model_id}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="image">Image</Label>
            {imagePreview ? (
              <div className="relative">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-full h-48 object-contain rounded border"
                />
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2"
                  onClick={handleClearImage}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <label
                htmlFor="image"
                className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded cursor-pointer hover:bg-muted/50"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="h-10 w-10 mb-3 text-muted-foreground" />
                  <p className="mb-2 text-sm text-muted-foreground">
                    Click to upload an image
                  </p>
                </div>
                <input
                  id="image"
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageSelect}
                />
              </label>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="prompt">Prompt</Label>
            <Textarea
              id="prompt"
              placeholder="What would you like to know about this image?"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
            />
          </div>

          <Button
            onClick={handleGenerate}
            disabled={!prompt.trim() || !selectedModel || !imageFile || isGenerating}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <ImageIcon className="h-4 w-4 mr-2" />
                Analyze Image
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Output</CardTitle>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="space-y-4">
              <div className="flex gap-2 flex-wrap">
                <Badge variant="secondary">Total: {result.total_ms}ms</Badge>
                <Badge variant="secondary">{result.tokens_generated} tokens</Badge>
                <Badge variant="secondary">
                  {(result.tokens_generated / (result.total_ms / 1000)).toFixed(1)} tok/s
                </Badge>
              </div>
              <div className="bg-muted p-4 rounded text-sm whitespace-pre-wrap">
                {result.text}
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              Upload an image and enter a prompt to see results
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
