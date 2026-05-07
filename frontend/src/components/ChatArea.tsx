import { useEffect, useRef } from 'react'
import type { Message } from '../types'
import { MessageBubble } from './MessageBubble'

interface ChatAreaProps {
  messages: Message[]
  onSuggestion: (q: string) => void
}

const SUGGESTIONS = [
  'Summarize the main topics covered in this document',
  'What are the key findings or conclusions?',
  'List all important dates, names, or numbers mentioned',
  'What problem does this document address?',
]

export function ChatArea({ messages, onSuggestion }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <div style={{ fontWeight: 600, fontSize: 16 }}>Ask anything about your document</div>
        <div style={{ fontSize: 13, opacity: 0.5 }}>The AI will answer strictly from the content you uploaded</div>
        <div className="suggestions">
          {SUGGESTIONS.map((q, i) => (
            <button key={i} className="suggestion-btn" onClick={() => onSuggestion(q)}>
              {q}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="messages-container">
      {messages.map(msg => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
