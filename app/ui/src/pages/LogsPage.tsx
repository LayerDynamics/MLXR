/**
 * Logs Page
 *
 * Daemon logs viewer with:
 * - Virtualized log list
 * - Level filtering
 * - Search/filter
 * - Auto-scroll toggle
 * - Export logs
 */

import { Header } from '@/components/layout/Header'
import { Button } from '@/components/ui/button'
import { Download, Trash2 } from 'lucide-react'

export default function LogsPage() {
  return (
    <div className="flex h-full flex-col">
      <Header
        title="Logs"
        actions={
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Trash2 className="mr-2 h-4 w-4" />
              Clear Logs
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export Logs
            </Button>
          </div>
        }
      />
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-muted-foreground">
            Daemon Logs
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Log viewer will be implemented here
          </p>
        </div>
      </div>
    </div>
  )
}
