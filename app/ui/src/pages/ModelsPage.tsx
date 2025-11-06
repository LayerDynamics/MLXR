/**
 * Models Page
 *
 * Model registry and management with:
 * - Model list/table
 * - Import/pull models
 * - Model details
 * - Adapter management
 */

import { Header } from '@/components/layout/Header'
import { Button } from '@/components/ui/button'
import { Download, FolderOpen } from 'lucide-react'

export default function ModelsPage() {
  return (
    <div className="flex h-full flex-col">
      <Header
        title="Models"
        actions={
          <div className="flex gap-2">
            <Button variant="outline">
              <FolderOpen className="mr-2 h-4 w-4" />
              Import Model
            </Button>
            <Button>
              <Download className="mr-2 h-4 w-4" />
              Pull Model
            </Button>
          </div>
        }
      />
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-muted-foreground">
            Models Registry
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Model management components will be implemented here
          </p>
        </div>
      </div>
    </div>
  )
}
