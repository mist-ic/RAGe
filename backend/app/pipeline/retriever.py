"""
RAGe — Qdrant Retriever

Wraps qdrant-client for in-memory per-session vector storage and search.
Each uploaded document gets its own Qdrant collection named by session_id.
"""

from typing import List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import get_settings
from app.pipeline.chunker import Chunk


# Module-level in-memory Qdrant client (shared across requests)
_qdrant: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    """Return the singleton in-memory Qdrant client."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(":memory:")
    return _qdrant


def create_collection(session_id: str) -> None:
    """
    Create a Qdrant collection for the given session.

    Collection name: session_id
    Vector config: cosine distance, EMBED_DIMENSIONS size.
    """
    settings = get_settings()
    client = get_qdrant()

    # Delete existing collection with same name (re-upload)
    if client.collection_exists(session_id):
        client.delete_collection(session_id)

    client.create_collection(
        collection_name=session_id,
        vectors_config=VectorParams(
            size=settings.EMBED_DIMENSIONS,
            distance=Distance.COSINE,
        ),
    )


def index_chunks(
    session_id: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    pdf_pages=None,
    pdf_embeddings: Optional[np.ndarray] = None,
) -> None:
    """
    Upsert chunk embeddings into the Qdrant collection.

    Each point stores the full chunk metadata as payload.
    If pdf_embeddings are provided, they are added as additional points
    tagged with source='pdf_page'.

    Args:
        session_id: Collection name.
        chunks: List of Chunk objects.
        embeddings: np.ndarray of shape (len(chunks), EMBED_DIMENSIONS).
        pdf_pages: Optional list of Page objects (for PDF page embeddings).
        pdf_embeddings: Optional np.ndarray of shape (N_pages, EMBED_DIMENSIONS).
    """
    client = get_qdrant()
    points: List[PointStruct] = []

    # Text chunk points
    for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=idx,
            vector=vec.tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "source": "text_chunk",
            },
        ))

    # PDF page points (if available)
    if pdf_pages and pdf_embeddings is not None:
        pages_with_bytes = [p for p in pdf_pages if p.pdf_bytes]
        for i, (page, vec) in enumerate(zip(pages_with_bytes, pdf_embeddings)):
            points.append(PointStruct(
                id=len(chunks) + i,
                vector=vec.tolist(),
                payload={
                    "chunk_id": f"pdf_page_{page.page_number}",
                    "document": page.filename,
                    "page": page.page_number,
                    "section": "PDF Page",
                    "text": page.text[:500],   # preview only
                    "token_count": len(page.text) // 4,
                    "source": "pdf_page",
                },
            ))

    client.upsert(collection_name=session_id, points=points)


def search(
    session_id: str,
    query_embedding: np.ndarray,
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Tuple[dict, float]]:
    """
    Search for the most relevant chunks.

    Args:
        session_id: Collection name.
        query_embedding: 1D normalized embedding vector.
        top_k: Max results to return.
        threshold: Min cosine similarity score.

    Returns:
        List of (payload_dict, score) tuples, sorted descending by score.
    """
    client = get_qdrant()

    if not client.collection_exists(session_id):
        return []

    results = client.query_points(
        collection_name=session_id,
        query=query_embedding.tolist(),
        limit=top_k * 2,   # fetch more, filter by threshold below
        with_payload=True,
    )

    filtered = [
        (hit.payload, hit.score)
        for hit in results.points
        if hit.score >= threshold
    ]

    # Deduplicate by page (prefer text_chunk over pdf_page for same page)
    seen_pages = set()
    deduped = []
    for payload, score in sorted(filtered, key=lambda x: -x[1]):
        page_key = (payload.get("document"), payload.get("page"))
        source = payload.get("source", "text_chunk")
        if page_key not in seen_pages or source == "text_chunk":
            if source == "text_chunk":
                seen_pages.add(page_key)
            deduped.append((payload, score))
        if len(deduped) >= top_k:
            break

    return deduped


def delete_collection(session_id: str) -> None:
    """Remove a session's collection."""
    client = get_qdrant()
    if client.collection_exists(session_id):
        client.delete_collection(session_id)
