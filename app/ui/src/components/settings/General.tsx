/**
 * General Settings Panel
 *
 * General application settings:
 * - Theme selection (light/dark/system)
 * - Language
 * - Auto-start on login
 * - Show in menu bar
 * - Notifications
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { SettingRow } from './SettingRow'
import { useAppStore } from '@/lib/store'

export interface GeneralProps {
  className?: string
}

export function General({ className }: GeneralProps) {
  const { theme, setTheme } = useAppStore()

  return (
    <div className={className}>
      <Card>
        <CardHeader>
          <CardTitle>General Settings</CardTitle>
          <CardDescription>
            Configure general application behavior
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-0 divide-y">
          <SettingRow
            label="Theme"
            description="Choose your preferred color theme"
          >
            <Select value={theme} onValueChange={setTheme}>
              <SelectTrigger className="w-[180px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="system">System</SelectItem>
              </SelectContent>
            </Select>
          </SettingRow>

          <SettingRow
            label="Launch at Login"
            description="Automatically start MLXR when you log in"
          >
            <Switch defaultChecked={false} />
          </SettingRow>

          <SettingRow
            label="Show in Menu Bar"
            description="Display MLXR icon in the system menu bar"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Enable Notifications"
            description="Show desktop notifications for important events"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Minimize to Tray"
            description="Keep MLXR running in the background when closed"
          >
            <Switch defaultChecked={true} />
          </SettingRow>

          <SettingRow
            label="Confirm Before Quit"
            description="Ask for confirmation before quitting the application"
          >
            <Switch defaultChecked={false} />
          </SettingRow>
        </CardContent>
      </Card>
    </div>
  )
}
