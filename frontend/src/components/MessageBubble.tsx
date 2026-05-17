import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'
import { SourceChips } from './SourceChips'

interface MessageBubbleProps {
  message: Message
}

const CRAG_ACTION_LABEL: Record<string, { text: string; color: string }> = {
  use_retrieved: { text: 'Doc',          color: 'var(--accent)' },
  use_web:       { text: 'Web',          color: '#f59e0b' },
  use_both:      { text: 'Doc + Web',    color: 'var(--accent2)' },
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  const cragAction = message.metadata?.crag?.action
  const cragInfo = cragAction ? CRAG_ACTION_LABEL[cragAction] : null

  return (
    <div className={`message ${message.role}`}>
      <div className="message-bubble">
        {isUser ? (
          <span>{message.content}</span>
        ) : (
          <>
            {message.content ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            ) : (
              <span style={{ opacity: 0.4, fontStyle: 'italic' }}>Thinking...</span>
            )}
            {message.isStreaming && message.content && (
              <span className="typing-cursor" />
            )}
          </>
        )}
      </div>

      {!isUser && message.sources && message.sources.length > 0 && (
        <SourceChips sources={message.sources} />
      )}

      {!isUser && message.metadata && (
        <div className="msg-meta">
          <span>{message.metadata.model_used.split('-').slice(0, 3).join(' ')}</span>
          <span>·</span>
          <span>{message.metadata.chunks_retrieved} chunks</span>
          <span>·</span>
          <span>{message.metadata.latency_ms}ms</span>
          {cragInfo && (
            <>
              <span>·</span>
              <span style={{
                color: cragInfo.color,
                fontWeight: 600,
                fontSize: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}>
                CRAG: {cragInfo.text}
              </span>
              {message.metadata.crag?.web_search_used && (
                <span style={{ color: '#f59e0b', fontSize: 10 }}>🌐</span>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
