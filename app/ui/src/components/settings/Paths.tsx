/**
 * Paths Settings Panel
 *
 * Configure file system paths:
 * - Models directory
 * - Cache directory
 * - Logs directory
 * - Config file location
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { PathPicker } from './PathPicker'
import { useState } from 'react'

export interface PathsProps {
  className?: string
}

export function Paths({ className }: PathsProps) {
  const [modelsPath, setModelsPath] = useState('~/Library/Application Support/MLXRunner/models')
  const [cachePath, setCachePath] = useState('~/Library/Application Support/MLXRunner/cache')
  const [logsPath, setLogsPath] = useState('~/Library/Logs/mlxrunnerd.log')
  const [configPath, setConfigPath] = useState('~/Library/Application Support/MLXRunner/server.yaml')

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle>Paths</CardTitle>
          <CardDescription>
            Configure storage locations for models, cache, and logs
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">Models Directory</label>
            <p className="text-sm text-muted-foreground">
              Location where model files are stored
            </p>
            <PathPicker
              value={modelsPath}
              onChange={setModelsPath}
              type="models"
              placeholder="~/Library/Application Support/MLXRunner/models"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Cache Directory</label>
            <p className="text-sm text-muted-foreground">
              Location for KV cache and temporary files
            </p>
            <PathPicker
              value={cachePath}
              onChange={setCachePath}
              type="cache"
              placeholder="~/Library/Application Support/MLXRunner/cache"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Logs Path</label>
            <p className="text-sm text-muted-foreground">
              Location of daemon log files
            </p>
            <PathPicker
              value={logsPath}
              onChange={setLogsPath}
              type="models"
              placeholder="~/Library/Logs/mlxrunnerd.log"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Config File</label>
            <p className="text-sm text-muted-foreground">
              Server configuration file location
            </p>
            <PathPicker
              value={configPath}
              onChange={setConfigPath}
              type="models"
              placeholder="~/Library/Application Support/MLXRunner/server.yaml"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
