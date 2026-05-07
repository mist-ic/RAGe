import { useRef, useState, useEffect } from 'react'

interface InputAreaProps {
  onSend: (q: string) => void
  onStop: () => void
  disabled: boolean
  sessionId: string | null
}

export function InputArea({ onSend, onStop, disabled, sessionId }: InputAreaProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 140)}px`
  }, [value])

  const handleSend = () => {
    const q = value.trim()
    if (!q || !sessionId) return
    setValue('')
    onSend(q)
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (disabled) onStop()
      else handleSend()
    }
  }

  return (
    <div className="input-area">
      <div className="input-row">
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={sessionId ? 'Ask about your document...' : 'Upload a document first'}
          disabled={!sessionId}
          rows={1}
        />
        <button
          className={`send-btn ${disabled ? 'stop' : ''}`}
          onClick={disabled ? onStop : handleSend}
          disabled={!sessionId && !disabled}
          title={disabled ? 'Stop generation' : 'Send (Enter)'}
        >
          {disabled ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="4" width="16" height="16" rx="2"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22,2 15,22 11,13 2,9"/>
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
