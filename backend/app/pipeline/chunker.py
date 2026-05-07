"""
RAGe — Structure-Aware Chunker

Splits document pages into retrieval-ready chunks using paragraph boundaries,
section heading detection, and small-chunk merging.

Chunking strategy (documented for assignment):
1. Detect section headings (title-case lines, 2-10 words, no trailing punctuation)
2. Split body by double newlines into paragraphs
3. Merge small paragraphs up to CHUNK_SIZE tokens to avoid noisy tiny chunks
4. Split any remaining oversized paragraph by sentences
5. Apply CHUNK_OVERLAP token overlap between adjacent chunks in the same section
6. Post-merge final tiny chunks across section boundaries
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from app.pipeline.extractor import Page


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    chunk_id: str
    document: str
    page: int
    section: str
    text: str
    token_count: int


# ── Helpers ────────────────────────────────────────────────

def _tok(text: str) -> int:
    """Estimate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _is_heading(line: str) -> bool:
    """
    Conservative section heading detection.

    A heading is:
    - 2-10 words
    - Starts with uppercase
    - Does not end with sentence punctuation (. , ; ?)
    - Not all digits/symbols
    - Not too long (< 120 chars)
    """
    s = line.strip()
    if not s or len(s) > 120 or len(s) < 4:
        return False
    words = s.split()
    if not (2 <= len(words) <= 10):
        return False
    if re.match(r'^[\d\$\%\.\,\-\+]+$', s):
        return False
    if s[-1] in '.,:;?!':
        return False
    if s.isupper() and 3 < len(s) < 60:
        return True
    if s[0].isupper():
        articles = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by'}
        cap = sum(1 for w in words if w[0].isupper() or w.lower() in articles)
        if cap >= len(words) * 0.6:
            return True
    return False


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split text into [(heading, body)] pairs."""
    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_heading = "General"
    current_lines: List[str] = []

    for line in lines:
        if _is_heading(line) and current_lines:
            body = '\n'.join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = '\n'.join(current_lines).strip()
    if body:
        sections.append((current_heading, body))

    return sections or [("General", text)]


def _split_paragraphs(text: str) -> List[str]:
    """Split on double newlines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _merge_small(paragraphs: List[str], max_tok: int) -> List[str]:
    """Merge consecutive small paragraphs up to max_tok."""
    if not paragraphs:
        return []
    merged, buf = [], ""
    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if _tok(candidate) <= max_tok:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = para
    if buf:
        merged.append(buf)
    return merged


def _split_long(text: str, max_tok: int) -> List[str]:
    """Split a too-long text by sentences."""
    if _tok(text) <= max_tok:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for sent in sentences:
        candidate = (cur + " " + sent).strip() if cur else sent
        if _tok(candidate) <= max_tok:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _apply_overlap(chunks: List[str], overlap_tok: int) -> List[str]:
    """Prepend tail of previous chunk to next chunk for context continuity."""
    if len(chunks) <= 1 or overlap_tok <= 0:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        overlap_words = max(1, int(overlap_tok * 1.3))
        if overlap_words < len(prev_words):
            tail = ' '.join(prev_words[-overlap_words:])
            result.append(tail + " " + chunks[i])
        else:
            result.append(chunks[i])
    return result


def _post_merge(
    pairs: List[Tuple[str, str]],
    min_tok: int = 80,
    max_tok: int = 400,
) -> List[Tuple[str, str]]:
    """Final pass: merge tiny chunks with their neighbors."""
    if not pairs:
        return []
    merged, buf_text, buf_head = [], "", ""
    for text, heading in pairs:
        if not buf_text:
            buf_text, buf_head = text, heading
            continue
        combined = buf_text + "\n\n" + text
        if (_tok(buf_text) < min_tok or _tok(text) < min_tok) and _tok(combined) <= max_tok:
            buf_text = combined
            if heading != "General":
                buf_head = heading
        else:
            merged.append((buf_text, buf_head))
            buf_text, buf_head = text, heading
    if buf_text:
        if merged and _tok(buf_text) < min_tok:
            prev_text, prev_head = merged[-1]
            combined = prev_text + "\n\n" + buf_text
            if _tok(combined) <= max_tok:
                merged[-1] = (combined, prev_head)
            else:
                merged.append((buf_text, buf_head))
        else:
            merged.append((buf_text, buf_head))
    return merged


# ── Public API ─────────────────────────────────────────────

def chunk_pages(
    pages: List[Page],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_tokens: int = 80,
) -> List[Chunk]:
    """
    Chunk extracted document pages into retrieval-ready pieces.

    Strategy:
    1. Split each page text into sections by heading detection
    2. Split each section body into paragraphs
    3. Merge small paragraphs up to chunk_size tokens
    4. Split oversized paragraphs by sentences
    5. Apply overlap between adjacent chunks within a section
    6. Post-merge tiny chunks across boundaries

    Args:
        pages: List of Page objects from extractor.
        chunk_size: Max tokens per chunk (default 400, ~1600 chars).
        chunk_overlap: Overlap tokens between adjacent chunks (default 60).
        min_chunk_tokens: Min tokens before merging with neighbor (default 80).

    Returns:
        List of Chunk objects with text and metadata.
    """
    all_chunks: List[Chunk] = []

    for page in pages:
        base = page.filename.replace('.pdf', '').replace('.txt', '')
        sections = _split_sections(page.text)
        page_pairs: List[Tuple[str, str]] = []

        for heading, body in sections:
            paragraphs = _split_paragraphs(body)
            merged = _merge_small(paragraphs, chunk_size)
            texts: List[str] = []
            for m in merged:
                texts.extend(_split_long(m, chunk_size))
            texts = _apply_overlap(texts, chunk_overlap)
            for t in texts:
                page_pairs.append((t, heading))

        page_pairs = _post_merge(page_pairs, min_chunk_tokens, chunk_size)

        for idx, (text, heading) in enumerate(page_pairs):
            all_chunks.append(Chunk(
                chunk_id=f"{base}_p{page.page_number}_c{idx}",
                document=page.filename,
                page=page.page_number,
                section=heading,
                text=text,
                token_count=_tok(text),
            ))

    return all_chunks
