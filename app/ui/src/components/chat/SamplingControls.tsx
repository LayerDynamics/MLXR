/**
 * SamplingControls Component
 *
 * Controls for sampling parameters:
 * - Temperature slider
 * - Top-P slider
 * - Top-K input
 * - Max tokens input
 * - Repetition penalty slider
 * - Reset to defaults button
 */

import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { RotateCcw } from 'lucide-react'

export interface SamplingControlsProps {
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  repetitionPenalty: number
  onTemperatureChange: (value: number) => void
  onTopPChange: (value: number) => void
  onTopKChange: (value: number) => void
  onMaxTokensChange: (value: number) => void
  onRepetitionPenaltyChange: (value: number) => void
  onReset?: () => void
  className?: string
}

export function SamplingControls({
  temperature,
  topP,
  topK,
  maxTokens,
  repetitionPenalty,
  onTemperatureChange,
  onTopPChange,
  onTopKChange,
  onMaxTokensChange,
  onRepetitionPenaltyChange,
  onReset,
  className,
}: SamplingControlsProps) {
  return (
    <div className={className}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Sampling Parameters</h3>
          {onReset && (
            <Button variant="ghost" size="sm" onClick={onReset} className="h-8">
              <RotateCcw className="mr-2 h-3 w-3" />
              Reset
            </Button>
          )}
        </div>

        {/* Temperature */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="temperature" className="text-sm">
              Temperature
            </Label>
            <span className="text-sm text-muted-foreground">
              {temperature.toFixed(2)}
            </span>
          </div>
          <Slider
            id="temperature"
            min={0}
            max={2}
            step={0.01}
            value={[temperature]}
            onValueChange={(values) => onTemperatureChange(values[0])}
          />
          <p className="text-xs text-muted-foreground">
            Controls randomness: 0 = deterministic, 2 = very random
          </p>
        </div>

        {/* Top-P */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="top-p" className="text-sm">
              Top-P (Nucleus)
            </Label>
            <span className="text-sm text-muted-foreground">
              {topP.toFixed(2)}
            </span>
          </div>
          <Slider
            id="top-p"
            min={0}
            max={1}
            step={0.01}
            value={[topP]}
            onValueChange={(values) => onTopPChange(values[0])}
          />
          <p className="text-xs text-muted-foreground">
            Cumulative probability cutoff for token selection
          </p>
        </div>

        {/* Top-K */}
        <div className="space-y-2">
          <Label htmlFor="top-k" className="text-sm">
            Top-K
          </Label>
          <Input
            id="top-k"
            type="number"
            min={0}
            max={100}
            value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Number of top tokens to consider (0 = disabled)
          </p>
        </div>

        {/* Max Tokens */}
        <div className="space-y-2">
          <Label htmlFor="max-tokens" className="text-sm">
            Max Tokens
          </Label>
          <Input
            id="max-tokens"
            type="number"
            min={1}
            max={8192}
            value={maxTokens}
            onChange={(e) => onMaxTokensChange(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Maximum number of tokens to generate
          </p>
        </div>

        {/* Repetition Penalty */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="repetition-penalty" className="text-sm">
              Repetition Penalty
            </Label>
            <span className="text-sm text-muted-foreground">
              {repetitionPenalty.toFixed(2)}
            </span>
          </div>
          <Slider
            id="repetition-penalty"
            min={1}
            max={2}
            step={0.01}
            value={[repetitionPenalty]}
            onValueChange={(values) => onRepetitionPenaltyChange(values[0])}
          />
          <p className="text-xs text-muted-foreground">
            Penalize repeated tokens (1 = no penalty, 2 = strong penalty)
          </p>
        </div>
      </div>
    </div>
  )
}
