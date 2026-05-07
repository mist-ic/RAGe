import './index.css'
import { useState } from 'react'
import type { UploadResult } from './types'
import { UploadArea } from './components/UploadArea'
import { ChatArea } from './components/ChatArea'
import { InputArea } from './components/InputArea'
import { useStreamingChat } from './hooks/useStreamingChat'

function App() {
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const { messages, isStreaming, sendMessage, stopStreaming, clearMessages } =
    useStreamingChat(uploadResult?.session_id ?? null)

  const handleUploadSuccess = (result: UploadResult) => {
    setUploadResult(result)
    clearMessages()
  }

  const handleUploadAnother = () => {
    setUploadResult(null)
    clearMessages()
  }

  return (
    <>
      <div className="ambient-glow tl" />
      <div className="ambient-glow br" />

      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="header-brand">
            <div className="header-logo">R</div>
            <div>
              <div className="header-title">RAGe</div>
            </div>
          </div>

          {uploadResult && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {messages.length > 0 && `${Math.floor(messages.length / 2)} questions`}
              </span>
              <button className="btn-upload-another" onClick={handleUploadAnother}>
                New document
              </button>
            </div>
          )}
        </header>

        {/* Main content */}
        <div className="content">
          {!uploadResult ? (
            <UploadArea onUploadSuccess={handleUploadSuccess} />
          ) : (
            <div className="chat-section">
              {/* Document banner */}
              <div className="doc-banner">
                <div className="doc-banner-info">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                  </svg>
                  <span style={{ fontWeight: 500 }}>{uploadResult.filename}</span>
                  <span>{uploadResult.pages} pages</span>
                  <span>·</span>
                  <span>{uploadResult.chunks_count} chunks</span>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <span className="doc-badge">Text RAG</span>
                  {uploadResult.has_pdf_embeddings && (
                    <span className="doc-badge multimodal">+ PDF Embeddings</span>
                  )}
                </div>
              </div>

              <ChatArea
                messages={messages}
                onSuggestion={sendMessage}
              />
            </div>
          )}
        </div>

        {/* Input always at bottom when doc loaded */}
        {uploadResult && (
          <InputArea
            onSend={sendMessage}
            onStop={stopStreaming}
            disabled={isStreaming}
            sessionId={uploadResult.session_id}
          />
        )}
      </div>
    </>
  )
}

export default App
