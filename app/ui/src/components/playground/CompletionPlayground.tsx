/**
 * CompletionPlayground Component
 *
 * Test completion API with custom parameters
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Loader2, Play } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useModels } from '@/hooks/useModels'

interface CompletionResult {
  text: string
  ttft_ms: number
  total_ms: number
  tokens_generated: number
}

export interface CompletionPlaygroundProps {
  className?: string
}

export function CompletionPlayground({ className }: CompletionPlaygroundProps) {
  const [prompt, setPrompt] = useState('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(40)
  const [maxTokens, setMaxTokens] = useState(256)
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState<CompletionResult | null>(null)
  const { models } = useModels()

  const handleGenerate = async () => {
    if (!prompt.trim() || !selectedModel) return

    setIsGenerating(true)
    const startTime = Date.now()

    try {
      const response = await fetch('/api/v1/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          prompt,
          temperature,
          top_p: topP,
          top_k: topK,
          max_tokens: maxTokens,
        }),
      })

      if (!response.ok) throw new Error('Generation failed')

      const data = await response.json()
      const totalTime = Date.now() - startTime

      setResult({
        text: data.choices[0].text,
        ttft_ms: totalTime,
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
          <CardTitle>Completion Playground</CardTitle>
          <CardDescription>Test text completion with custom parameters</CardDescription>
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
            <Label htmlFor="prompt">Prompt</Label>
            <Textarea
              id="prompt"
              placeholder="Enter your prompt..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Temperature: {temperature.toFixed(2)}</Label>
              <Slider
                value={[temperature]}
                onValueChange={(v) => setTemperature(v[0])}
                min={0}
                max={2}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <Label>Top P: {topP.toFixed(2)}</Label>
              <Slider
                value={[topP]}
                onValueChange={(v) => setTopP(v[0])}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="topK">Top K</Label>
              <Input
                id="topK"
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 0)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="maxTokens">Max Tokens</Label>
              <Input
                id="maxTokens"
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 0)}
              />
            </div>
          </div>

          <Button
            onClick={handleGenerate}
            disabled={!prompt.trim() || !selectedModel || isGenerating}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Generate
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
                <Badge variant="secondary">TTFT: {result.ttft_ms}ms</Badge>
                <Badge variant="secondary">Total: {result.total_ms}ms</Badge>
                <Badge variant="secondary">{result.tokens_generated} tokens</Badge>
                <Badge variant="secondary">
                  {(result.tokens_generated / (result.total_ms / 1000)).toFixed(1)} tok/s
                </Badge>
              </div>
              <pre className="bg-muted p-4 rounded text-sm whitespace-pre-wrap">
                {result.text}
              </pre>
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              Configure parameters and click Generate to see results
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
