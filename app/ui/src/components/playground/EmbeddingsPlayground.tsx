/**
 * EmbeddingsPlayground Component
 *
 * Test embeddings API
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Loader2, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useModels } from '@/hooks/useModels'

interface EmbeddingResult {
  vector: number[]
  dimensions: number
}

export interface EmbeddingsPlaygroundProps {
  className?: string
}

export function EmbeddingsPlayground({ className }: EmbeddingsPlaygroundProps) {
  const [text1, setText1] = useState('')
  const [text2, setText2] = useState('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [embedding1, setEmbedding1] = useState<EmbeddingResult | null>(null)
  const [embedding2, setEmbedding2] = useState<EmbeddingResult | null>(null)
  const { models } = useModels()

  const handleGenerate = async (text: string, setEmbedding: (result: EmbeddingResult) => void) => {
    if (!text.trim() || !selectedModel) return

    setIsGenerating(true)
    try {
      const response = await fetch('/api/v1/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          input: text,
        }),
      })

      if (!response.ok) throw new Error('Generation failed')

      const data = await response.json()
      const vector = data.data[0].embedding

      setEmbedding({
        vector,
        dimensions: vector.length,
      })
    } catch (error) {
      console.error('Generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const calculateCosineSimilarity = (v1: number[], v2: number[]): number => {
    if (v1.length !== v2.length) return 0
    let dotProduct = 0
    let norm1 = 0
    let norm2 = 0
    for (let i = 0; i < v1.length; i++) {
      dotProduct += v1[i] * v2[i]
      norm1 += v1[i] * v1[i]
      norm2 += v2[i] * v2[i]
    }
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2))
  }

  const similarity = embedding1 && embedding2
    ? calculateCosineSimilarity(embedding1.vector, embedding2.vector)
    : null

  const renderVector = (vector: number[]) => {
    const preview = vector.slice(0, 10).map(v => v.toFixed(4)).join(', ')
    return `[${preview}, ... ${vector.length} dims]`
  }

  return (
    <div className={cn('space-y-4', className)}>
      <Card>
        <CardHeader>
          <CardTitle>Embeddings Playground</CardTitle>
          <CardDescription>Generate embeddings and calculate similarity</CardDescription>
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
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Text 1</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter first text..."
              value={text1}
              onChange={(e) => setText1(e.target.value)}
              rows={4}
            />
            <Button
              onClick={() => handleGenerate(text1, setEmbedding1)}
              disabled={!text1.trim() || !selectedModel || isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Embedding
                </>
              )}
            </Button>
            {embedding1 && (
              <div className="space-y-2">
                <Badge>{embedding1.dimensions} dimensions</Badge>
                <div className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto">
                  {renderVector(embedding1.vector)}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Text 2</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter second text..."
              value={text2}
              onChange={(e) => setText2(e.target.value)}
              rows={4}
            />
            <Button
              onClick={() => handleGenerate(text2, setEmbedding2)}
              disabled={!text2.trim() || !selectedModel || isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Embedding
                </>
              )}
            </Button>
            {embedding2 && (
              <div className="space-y-2">
                <Badge>{embedding2.dimensions} dimensions</Badge>
                <div className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto">
                  {renderVector(embedding2.vector)}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {similarity !== null && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Cosine Similarity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold">{similarity.toFixed(4)}</div>
              <div className="flex-1">
                <div className="h-4 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${((similarity + 1) / 2) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
