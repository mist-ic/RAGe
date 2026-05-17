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
  crag_label?: 'correct' | 'incorrect' | 'ambiguous' | null
}

export interface CRAGMetadata {
  action: 'use_retrieved' | 'use_web' | 'use_both'
  chunks_graded: number
  chunks_kept: number
  web_search_used: boolean
}

export interface QueryMetadata {
  model_used: string
  latency_ms: number
  chunks_retrieved: number
  crag?: CRAGMetadata | null
}

export interface UploadResult {
  session_id: string
  filename: string
  chunks_count: number
  pages: number
  has_pdf_embeddings: boolean
}
