/**
 * Layout Component
 *
 * Main application layout with:
 * - Sidebar navigation (left)
 * - Main content area (center)
 * - Status bar (bottom)
 * - Responsive breakpoints
 */

import { ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background">
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar Navigation */}
        <Sidebar />

        {/* Main Content Area */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {children}
        </main>
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  )
}
