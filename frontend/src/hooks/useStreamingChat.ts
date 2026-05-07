import { useState, useRef } from 'react'
import type { Message, UploadResult } from '../types'

export function useStreamingChat(sessionId: string | null) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [conversationId, setConversationId] = useState<string | undefined>()
  const abortRef = useRef<AbortController | null>(null)

  const processSSE = (
    buffer: string,
    msgId: string,
    contentRef: { current: string }
  ): string => {
    const events = buffer.split('\n\n')
    const remainder = events.pop() ?? ''

    for (const event of events) {
      const line = event.trim()
      if (!line.startsWith('data: ')) continue
      const dataStr = line.slice(6)
      if (!dataStr) continue

      try {
        const data = JSON.parse(dataStr)

        if (data.token) {
          contentRef.current += data.token
          const snap = contentRef.current
          setMessages(prev =>
            prev.map(m => m.id === msgId ? { ...m, content: snap } : m)
          )
        } else if (data.done) {
          setMessages(prev =>
            prev.map(m =>
              m.id === msgId
                ? { ...m, isStreaming: false, metadata: data.metadata, sources: data.sources }
                : m
            )
          )
          if (data.conversation_id) setConversationId(data.conversation_id)
        } else if (data.error) {
          setMessages(prev =>
            prev.map(m =>
              m.id === msgId
                ? { ...m, content: contentRef.current + `\n\n**Error:** ${data.error}`, isStreaming: false }
                : m
            )
          )
        }
      } catch {
        // ignore parse errors
      }
    }

    return remainder
  }

  const sendMessage = async (question: string) => {
    if (!sessionId) return
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()

    const userMsgId = crypto.randomUUID()
    const assistantMsgId = crypto.randomUUID()

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: question },
      { id: assistantMsgId, role: 'assistant', content: '', isStreaming: true },
    ])
    setIsStreaming(true)

    try {
      const resp = await fetch('/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, session_id: sessionId, conversation_id: conversationId }),
        signal: abortRef.current.signal,
      })

      if (!resp.ok || !resp.body) throw new Error('stream-unavailable')

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      const contentRef = { current: '' }
      let buf = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        buf = processSSE(buf, assistantMsgId, contentRef)
      }

      if (buf.trim()) processSSE(buf + '\n\n', assistantMsgId, contentRef)

    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return

      // Fallback to non-streaming
      try {
        const resp = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, session_id: sessionId, conversation_id: conversationId }),
        })
        const data = await resp.json()
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId
              ? { ...m, content: data.answer, isStreaming: false, metadata: data.metadata, sources: data.sources }
              : m
          )
        )
        if (data.conversation_id) setConversationId(data.conversation_id)
      } catch {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMsgId
              ? { ...m, content: 'Connection error. Please check the server.', isStreaming: false }
              : m
          )
        )
      }
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }

  const stopStreaming = () => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
      setIsStreaming(false)
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last?.isStreaming) {
          return prev.map((m, i) =>
            i === prev.length - 1 ? { ...m, isStreaming: false, content: m.content + ' [stopped]' } : m
          )
        }
        return prev
      })
    }
  }

  const clearMessages = () => {
    setMessages([])
    setConversationId(undefined)
  }

  return { messages, isStreaming, sendMessage, stopStreaming, clearMessages }
}
