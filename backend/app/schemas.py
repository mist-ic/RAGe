"""
RAGe — Pydantic Schemas

Request/response models for the API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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


class QueryMetadata(BaseModel):
    """Metadata about query processing."""
    model_used: str
    latency_ms: int
    chunks_retrieved: int


class QueryResponse(BaseModel):
    """POST /query response."""
    answer: str
    metadata: QueryMetadata
    sources: List[SourceInfo]
    conversation_id: str
