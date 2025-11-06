/**
 * Chat Page
 *
 * Main chat interface with:
 * - Conversation list
 * - Message display
 * - Message composer
 * - Model selector
 */

import { Header } from '@/components/layout/Header'
import { Button } from '@/components/ui/button'
import { MessageSquarePlus } from 'lucide-react'

export default function ChatPage() {
  return (
    <div className="flex h-full flex-col">
      <Header
        title="Chat"
        actions={
          <Button>
            <MessageSquarePlus className="mr-2 h-4 w-4" />
            New Conversation
          </Button>
        }
      />
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-muted-foreground">
            Chat Interface
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Chat components will be implemented here
          </p>
        </div>
      </div>
    </div>
  )
}
