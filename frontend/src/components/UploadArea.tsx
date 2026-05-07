import { useState, useCallback } from 'react'
import type { UploadResult } from '../types'

interface UploadAreaProps {
  onUploadSuccess: (result: UploadResult) => void
}

export function UploadArea({ onUploadSuccess }: UploadAreaProps) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const processFile = useCallback(async (file: File) => {
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const resp = await fetch('/upload', { method: 'POST', body: formData })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Upload failed')
      onUploadSuccess(data as UploadResult)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }, [onUploadSuccess])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processFile(file)
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) processFile(file)
  }

  return (
    <div className="upload-section">
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          className="upload-input"
          accept=".pdf,.txt,.md"
          onChange={onFileChange}
          disabled={loading}
        />
        <div className="upload-icon">
          {loading ? (
            <div className="spinner" />
          ) : (
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14,2 14,8 20,8"/>
              <line x1="12" y1="18" x2="12" y2="12"/>
              <polyline points="9,15 12,12 15,15"/>
            </svg>
          )}
        </div>
        <div className="upload-title">
          {loading ? 'Processing document...' : 'Drop your document here'}
        </div>
        <div className="upload-sub">
          {loading
            ? 'Chunking, embedding, and indexing into Qdrant'
            : 'Or click to browse. We\'ll chunk, embed, and index it automatically.'}
        </div>
        <div className="upload-types">
          <span className="upload-type-badge">.pdf</span>
          <span className="upload-type-badge">.txt</span>
          <span className="upload-type-badge">.md</span>
        </div>
      </div>

      {error && (
        <div className="upload-status error">
          <div className="upload-status-row">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ff4f6b" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  )
}
