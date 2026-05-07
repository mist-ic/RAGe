"""
RAGe — Gemini Embedding 2 Wrapper

Handles text chunk embedding and query embedding using gemini-embedding-2.
Supports PDF page binary embedding for multimodal capability.

Task prefixes per Gemini Embedding 2 docs:
- Documents: "title: {filename} | text: {content}"
- Queries:   "task: question answering | query: {question}"
"""

import base64
import time
from typing import List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types

from app.config import get_settings
from app.pipeline.chunker import Chunk
from app.pipeline.extractor import Page


def _get_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def embed_chunks(chunks: List[Chunk]) -> np.ndarray:
    """
    Embed text chunks using Gemini Embedding 2.

    Each chunk is formatted with the asymmetric retrieval task prefix:
      "title: {filename} | text: {chunk_text}"

    Returns:
        numpy array of shape (N, EMBED_DIMENSIONS), L2-normalized.
    """
    settings = get_settings()
    client = _get_client()

    texts = [
        f"title: {c.document} | text: {c.text}"
        for c in chunks
    ]

    # Gemini Embedding 2 needs separate Content objects for separate embeddings
    all_embeddings: List[List[float]] = []

    # Process in batches of 20 to stay within rate limits
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Wrap each in a Content object to get separate embeddings
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in batch
        ]

        result = client.models.embed_content(
            model=settings.EMBED_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=settings.EMBED_DIMENSIONS
            ),
        )

        for emb in result.embeddings:
            all_embeddings.append(emb.values)

        # Small delay to avoid rate limits
        if i + batch_size < len(texts):
            time.sleep(0.1)

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2 normalize (gemini-embedding-2 auto-normalizes truncated dims, but we normalize for safety)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return arr / norms


def embed_query(question: str) -> np.ndarray:
    """
    Embed a user query using Gemini Embedding 2.

    Formatted with the question-answering task prefix:
      "task: question answering | query: {question}"

    Returns:
        1D numpy array of shape (EMBED_DIMENSIONS,), L2-normalized.
    """
    settings = get_settings()
    client = _get_client()

    formatted = f"task: question answering | query: {question}"

    result = client.models.embed_content(
        model=settings.EMBED_MODEL,
        contents=formatted,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.EMBED_DIMENSIONS
        ),
    )

    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def embed_pdf_pages(pages: List[Page]) -> Optional[np.ndarray]:
    """
    Embed PDF pages directly as binary using Gemini Embedding 2 multimodal.

    Limited to pages with pdf_bytes. Each page is embedded as a standalone
    PDF file (visual + text content). Useful for documents with complex layouts,
    tables, or visual content that text extraction misses.

    Returns:
        numpy array of shape (N, EMBED_DIMENSIONS) or None if no pages with PDF bytes.
    """
    settings = get_settings()
    pages_with_bytes = [p for p in pages if p.pdf_bytes]

    if not pages_with_bytes:
        return None

    # Gemini Embedding 2 supports up to 6 pages per request (PDF limit)
    # We embed each page separately to get per-page vectors
    client = _get_client()
    all_embeddings: List[List[float]] = []

    for page in pages_with_bytes:
        try:
            contents = [
                types.Content(parts=[
                    types.Part.from_bytes(
                        data=page.pdf_bytes,
                        mime_type="application/pdf",
                    )
                ])
            ]
            result = client.models.embed_content(
                model=settings.EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.EMBED_DIMENSIONS
                ),
            )
            all_embeddings.append(result.embeddings[0].values)
            time.sleep(0.1)
        except Exception as e:
            print(f"[Embedder] PDF page embedding failed for page {page.page_number}: {e}")
            continue

    if not all_embeddings:
        return None

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return arr / norms
