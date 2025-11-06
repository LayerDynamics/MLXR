/**
 * Privacy Settings Panel
 *
 * Privacy and data handling:
 * - Telemetry opt-in/out
 * - Crash reporting
 * - Usage analytics
 * - Data retention
 * - Clear cache/history
 */

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { SettingRow } from './SettingRow'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Trash2, Shield, AlertCircle } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

export interface PrivacyProps {
  className?: string
}

export function Privacy({ className }: PrivacyProps) {
  const [showClearDialog, setShowClearDialog] = useState(false)
  const [clearType, setClearType] = useState<'cache' | 'history' | 'all'>('cache')

  const handleClearData = (type: 'cache' | 'history' | 'all') => {
    setClearType(type)
    setShowClearDialog(true)
  }

  const confirmClearData = async () => {
    try {
      // Implement clear data logic
      console.log(`Clearing ${clearType}`)
      setShowClearDialog(false)
    } catch (error) {
      console.error('Failed to clear data:', error)
    }
  }

  return (
    <>
      <div className={className}>
        <Card>
          <CardHeader>
            <CardTitle>Privacy & Data</CardTitle>
            <CardDescription>
              Control your data and privacy settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Privacy Notice */}
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription>
                MLXR processes all data locally. No personal information is sent to
                external servers unless you explicitly enable telemetry.
              </AlertDescription>
            </Alert>

            {/* Settings */}
            <div className="space-y-0 divide-y">
              <SettingRow
                label="Anonymous Telemetry"
                description="Help improve MLXR by sending anonymous usage data"
              >
                <Switch defaultChecked={false} />
              </SettingRow>

              <SettingRow
                label="Crash Reporting"
                description="Automatically send crash reports to help fix bugs"
              >
                <Switch defaultChecked={false} />
              </SettingRow>

              <SettingRow
                label="Usage Analytics"
                description="Collect anonymous performance and usage statistics"
              >
                <Switch defaultChecked={false} />
              </SettingRow>

              <SettingRow
                label="Conversation History"
                description="Save chat conversations locally"
              >
                <Switch defaultChecked={true} />
              </SettingRow>

              <SettingRow
                label="Model Download History"
                description="Track downloaded models and their sources"
              >
                <Switch defaultChecked={true} />
              </SettingRow>
            </div>

            {/* Data Management */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium">Data Management</h4>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleClearData('cache')}
                >
                  <Trash2 className="mr-2 h-3 w-3" />
                  Clear Cache
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleClearData('history')}
                >
                  <Trash2 className="mr-2 h-3 w-3" />
                  Clear History
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleClearData('all')}
                >
                  <Trash2 className="mr-2 h-3 w-3" />
                  Clear All Data
                </Button>
              </div>
            </div>

            {/* Warning */}
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Clearing data will remove cached models, conversation history, and
                settings. This action cannot be undone.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>

      {/* Confirmation Dialog */}
      <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear {clearType}?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete{' '}
              {clearType === 'all'
                ? 'all cached data, conversation history, and settings'
                : clearType === 'cache'
                ? 'cached model data and KV cache'
                : 'conversation history'}
              . This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmClearData}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Clear Data
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
