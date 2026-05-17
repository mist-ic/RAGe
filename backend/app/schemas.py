"""
RAGe — Pydantic Schemas

Request/response models for the API.
"""

from typing import List, Optional
from pydantic import BaseModel


# ── Upload ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after uploading and indexing a document."""
    session_id: str
    filename: str
    chunks_count: int
    pages: int
    has_pdf_embeddings: bool = False


# ── Query ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /query or /query/stream request body."""
    question: str
    session_id: str
    conversation_id: Optional[str] = None


class SourceInfo(BaseModel):
    """A retrieved chunk used as context."""
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    relevance_score: float
    text_preview: str = ""
    crag_label: Optional[str] = None    # correct / incorrect / ambiguous


class CRAGMetadata(BaseModel):
    """Corrective RAG step metadata."""
    action: str                         # use_retrieved / use_web / use_both
    chunks_graded: int
    chunks_kept: int
    web_search_used: bool


class QueryMetadata(BaseModel):
    """Metadata about query processing."""
    model_used: str
    latency_ms: int
    chunks_retrieved: int
    crag: Optional[CRAGMetadata] = None


class QueryResponse(BaseModel):
    """POST /query response."""
    answer: str
    metadata: QueryMetadata
    sources: List[SourceInfo]
    conversation_id: str
