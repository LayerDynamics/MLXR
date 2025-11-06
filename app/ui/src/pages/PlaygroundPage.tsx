/**
 * Playground Page
 *
 * API testing playground with tabs for:
 * - Embeddings
 * - Completion
 * - Vision (future)
 */

import { Header } from '@/components/layout/Header'

export default function PlaygroundPage() {
  return (
    <div className="flex h-full flex-col">
      <Header title="Playground" />
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-muted-foreground">
            API Playground
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Playground components will be implemented here
          </p>
        </div>
      </div>
    </div>
  )
}
