/**
 * Updates Settings Panel
 *
 * Software update settings:
 * - Auto-update configuration
 * - Update channel (stable/beta)
 * - Check for updates button
 * - Current version info
 * - Release notes
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { SettingRow } from './SettingRow'
import { Download, Check, Loader2, ExternalLink } from 'lucide-react'

export interface UpdatesProps {
  className?: string
}

export function Updates({ className }: UpdatesProps) {
  const [updateChannel, setUpdateChannel] = useState<'stable' | 'beta'>('stable')
  const [isChecking, setIsChecking] = useState(false)
  const [updateAvailable, setUpdateAvailable] = useState(false)

  const currentVersion = '0.1.0'

  const handleCheckUpdates = async () => {
    setIsChecking(true)
    try {
      // Simulate update check
      await new Promise((resolve) => setTimeout(resolve, 2000))
      setUpdateAvailable(false)
    } catch (error) {
      console.error('Failed to check for updates:', error)
    } finally {
      setIsChecking(false)
    }
  }

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle>Updates</CardTitle>
          <CardDescription>
            Manage application updates and versioning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Current Version */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div>
              <div className="text-sm font-medium">Current Version</div>
              <div className="text-sm text-muted-foreground">
                MLXR v{currentVersion}
              </div>
            </div>
            <Badge variant={updateAvailable ? 'default' : 'secondary'}>
              {updateAvailable ? 'Update Available' : 'Up to Date'}
            </Badge>
          </div>

          {/* Settings */}
          <div className="space-y-0 divide-y">
            <SettingRow
              label="Automatic Updates"
              description="Automatically download and install updates"
            >
              <Switch defaultChecked={true} />
            </SettingRow>

            <SettingRow
              label="Update Channel"
              description="Choose which updates to receive"
            >
              <Select value={updateChannel} onValueChange={(value: any) => setUpdateChannel(value)}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="stable">Stable</SelectItem>
                  <SelectItem value="beta">Beta</SelectItem>
                </SelectContent>
              </Select>
            </SettingRow>

            <SettingRow
              label="Check on Startup"
              description="Check for updates when the application starts"
            >
              <Switch defaultChecked={true} />
            </SettingRow>

            <SettingRow
              label="Show Release Notes"
              description="Display release notes after updating"
            >
              <Switch defaultChecked={true} />
            </SettingRow>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button
              onClick={handleCheckUpdates}
              disabled={isChecking}
            >
              {isChecking ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Checking...
                </>
              ) : updateAvailable ? (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Install Update
                </>
              ) : (
                <>
                  <Check className="mr-2 h-4 w-4" />
                  Check for Updates
                </>
              )}
            </Button>
            <Button variant="outline" asChild>
              <a
                href="https://github.com/mlxr/releases"
                target="_blank"
                rel="noopener noreferrer"
              >
                Release Notes
                <ExternalLink className="ml-2 h-3 w-3" />
              </a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
