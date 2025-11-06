/**
 * Performance Settings Panel
 *
 * Performance and resource settings:
 * - GPU memory limit
 * - Batch size
 * - Context length
 * - KV cache settings
 * - Speculative decoding
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import { SettingRow } from './SettingRow'
import { useState } from 'react'

export interface PerformanceProps {
  className?: string
}

export function Performance({ className }: PerformanceProps) {
  const [batchSize, setBatchSize] = useState(32)
  const [contextLength, setContextLength] = useState(4096)
  const [maxMemory, setMaxMemory] = useState(8)

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle>Performance Settings</CardTitle>
          <CardDescription>
            Configure resource usage and performance options
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-0 divide-y">
          <SettingRow
            label="GPU Memory Limit"
            description={`Maximum GPU memory to use: ${maxMemory} GB`}
          >
            <div className="w-[200px]">
              <Slider
                value={[maxMemory]}
                onValueChange={(value) => setMaxMemory(value[0])}
                min={2}
                max={64}
                step={2}
              />
            </div>
          </SettingRow>

          <SettingRow
            label="Max Batch Size"
            description="Maximum number of requests to batch together"
          >
            <Input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-[120px]"
              min={1}
              max={256}
            />
          </SettingRow>

          <SettingRow
            label="Context Length"
            description="Maximum context window size (tokens)"
          >
            <Input
              type="number"
              value={contextLength}
              onChange={(e) => setContextLength(Number(e.target.value))}
              className="w-[120px]"
              min={512}
              max={32768}
              step={512}
            />
          </SettingRow>

          <SettingRow
            label="Enable KV Cache"
            description="Cache key-value pairs for faster inference"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="KV Cache Persistence"
            description="Save KV cache to disk between sessions"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Speculative Decoding"
            description="Use draft model to speed up generation"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Continuous Batching"
            description="Dynamically batch requests at token boundaries"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Quantization"
            description="Use quantized models for faster inference"
          >
            <Switch defaultChecked={true} />
          </SettingRow>
        </CardContent>
      </Card>
    </div>
  )
}
