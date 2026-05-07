export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  sources?: Source[]
  metadata?: QueryMetadata
}

export interface Source {
  document: string
  page: number | null
  section: string | null
  relevance_score: number
  text_preview: string
}

export interface QueryMetadata {
  model_used: string
  latency_ms: number
  chunks_retrieved: number
}

export interface UploadResult {
  session_id: string
  filename: string
  chunks_count: number
  pages: number
  has_pdf_embeddings: boolean
}
