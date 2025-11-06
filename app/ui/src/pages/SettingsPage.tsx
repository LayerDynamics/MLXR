/**
 * Settings Page
 *
 * Application settings with tabs:
 * - General (theme, language, launch at login)
 * - Performance (batching, KV cache, speculation)
 * - Paths (models directory, cache directory)
 * - Updates (auto-update, check for updates)
 * - Privacy (telemetry, data management)
 */

import { Header } from '@/components/layout/Header'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export default function SettingsPage() {
  return (
    <div className="flex h-full flex-col">
      <Header title="Settings" />
      <div className="flex-1 overflow-auto p-6">
        <Tabs defaultValue="general" className="w-full">
          <TabsList>
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="paths">Paths</TabsTrigger>
            <TabsTrigger value="updates">Updates</TabsTrigger>
            <TabsTrigger value="privacy">Privacy</TabsTrigger>
          </TabsList>
          <TabsContent value="general" className="space-y-4 py-4">
            <div className="text-center text-muted-foreground">
              General settings will be implemented here
            </div>
          </TabsContent>
          <TabsContent value="performance" className="space-y-4 py-4">
            <div className="text-center text-muted-foreground">
              Performance settings will be implemented here
            </div>
          </TabsContent>
          <TabsContent value="paths" className="space-y-4 py-4">
            <div className="text-center text-muted-foreground">
              Path settings will be implemented here
            </div>
          </TabsContent>
          <TabsContent value="updates" className="space-y-4 py-4">
            <div className="text-center text-muted-foreground">
              Update settings will be implemented here
            </div>
          </TabsContent>
          <TabsContent value="privacy" className="space-y-4 py-4">
            <div className="text-center text-muted-foreground">
              Privacy settings will be implemented here
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
