/**
 * Metrics Page
 *
 * Real-time performance metrics with:
 * - Stat cards (tokens/s, latency, requests)
 * - Charts (latency histogram, throughput, KV usage)
 * - Time range selector
 * - Export functionality
 */

import { Header } from '@/components/layout/Header'

export default function MetricsPage() {
  return (
    <div className="flex h-full flex-col">
      <Header title="Metrics" />
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-muted-foreground">
            Performance Metrics
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Metrics visualization will be implemented here
          </p>
        </div>
      </div>
    </div>
  )
}
