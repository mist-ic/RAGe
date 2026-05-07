import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'
import { SourceChips } from './SourceChips'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

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
          <span>{message.metadata.model_used.split('-').slice(0,3).join(' ')}</span>
          <span>·</span>
          <span>{message.metadata.chunks_retrieved} chunks</span>
          <span>·</span>
          <span>{message.metadata.latency_ms}ms</span>
        </div>
      )}
    </div>
  )
}
