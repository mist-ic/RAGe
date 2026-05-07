"""
RAGe — FastAPI Application

Routes:
  POST /upload           — Upload PDF/TXT, chunk, embed, index into Qdrant
  POST /query/stream     — SSE streaming RAG query
  POST /query            — Non-streaming RAG query
  GET  /health           — Health check
  GET  /*                — Serve React SPA (production)
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import gemini_client
from app.config import get_settings
from app.memory.conversation import ConversationMemory
from app.pipeline import embedder, retriever
from app.pipeline.chunker import chunk_pages
from app.pipeline.extractor import extract_document
from app.pipeline.prompt import build_prompt
from app.schemas import (
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    UploadResponse,
)


# ── State ──────────────────────────────────────────────────

memory = ConversationMemory()

# Track session metadata: session_id -> {filename, chunks_count, pages, has_pdf_embeddings}
_sessions: dict = {}


# ── Lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    # Initialize Qdrant client on startup
    retriever.get_qdrant()
    print(f"[RAGe] Started. Qdrant ready. Upload dir: {settings.UPLOAD_DIR}")
    yield
    print("[RAGe] Shutting down.")


# ── App ────────────────────────────────────────────────────

app = FastAPI(
    title="RAGe",
    description="Upload any document and chat with it. Built with Gemini 3 Flash + Gemini Embedding 2 + Qdrant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "RAGe"}


# ── Upload ─────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF or TXT file, process it through the full RAG pipeline:
    1. Extract text (per-page)
    2. Structure-aware paragraph chunking
    3. Embed chunks with Gemini Embedding 2 (text + optional PDF page embeddings)
    4. Store in Qdrant (in-memory, per-session collection)
    Returns a session_id for subsequent queries.
    """
    settings = get_settings()

    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF, TXT, or MD file.",
        )

    # Save to disk temporarily
    session_id = str(uuid.uuid4())
    save_path = Path(settings.UPLOAD_DIR) / f"{session_id}_{filename}"
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    with open(save_path, "wb") as f:
        f.write(content)

    try:
        # 1. Extract pages
        pages = extract_document(str(save_path))
        if not pages:
            raise HTTPException(status_code=422, detail="No text content found in document.")

        # 2. Chunk
        chunks = chunk_pages(
            pages,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not produce any chunks from document.")

        # 3. Embed text chunks
        embeddings = embedder.embed_chunks(chunks)

        # 4. PDF page embeddings (bonus: multimodal, only for PDF files <= 6 pages)
        pdf_embeddings = None
        has_pdf_embeddings = False
        if ext == ".pdf" and len(pages) <= 6:
            pdf_embeddings = embedder.embed_pdf_pages(pages)
            if pdf_embeddings is not None:
                has_pdf_embeddings = True

        # 5. Create Qdrant collection and index
        retriever.create_collection(session_id)
        retriever.index_chunks(
            session_id,
            chunks,
            embeddings,
            pdf_pages=pages if has_pdf_embeddings else None,
            pdf_embeddings=pdf_embeddings,
        )

        # Store session metadata
        _sessions[session_id] = {
            "filename": filename,
            "chunks_count": len(chunks),
            "pages": len(pages),
            "has_pdf_embeddings": has_pdf_embeddings,
        }

        print(f"[Upload] {filename}: {len(pages)} pages, {len(chunks)} chunks, pdf_embed={has_pdf_embeddings}")

        return UploadResponse(
            session_id=session_id,
            filename=filename,
            chunks_count=len(chunks),
            pages=len(pages),
            has_pdf_embeddings=has_pdf_embeddings,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.remove(save_path)
        except OSError:
            pass


# ── Shared RAG logic ───────────────────────────────────────

def _run_rag(req: QueryRequest) -> tuple:
    """
    Core RAG pipeline: embed query, retrieve chunks, build prompt.

    Returns:
        (system_prompt, contents, retrieved_chunks, question)
    """
    settings = get_settings()

    if req.session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a document first.",
        )

    # 1. Embed query
    q_embedding = embedder.embed_query(req.question)

    # 2. Retrieve relevant chunks from Qdrant
    results = retriever.search(
        req.session_id,
        q_embedding,
        top_k=settings.TOP_K,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    # 3. Build grounded prompt
    history = None
    if req.conversation_id:
        history = memory.get_history(req.conversation_id)

    system_prompt, contents = build_prompt(results, history)
    return system_prompt, contents, results, req.question


# ── Non-streaming query ────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Non-streaming RAG query. Returns full answer JSON."""
    t_start = time.time()

    system_prompt, contents, results, question = _run_rag(req)
    answer = gemini_client.generate(system_prompt, contents, question)

    conv_id = req.conversation_id or str(uuid.uuid4())
    memory.add_turn(conv_id, question, answer)

    latency_ms = int((time.time() - t_start) * 1000)
    settings = get_settings()

    sources = [
        SourceInfo(
            document=p.get("document", ""),
            page=p.get("page"),
            section=p.get("section"),
            relevance_score=round(s, 4),
            text_preview=p.get("text", "")[:180],
        )
        for p, s in results
    ]

    return QueryResponse(
        answer=answer,
        metadata=QueryMetadata(
            model_used=settings.LLM_MODEL,
            latency_ms=latency_ms,
            chunks_retrieved=len(results),
        ),
        sources=sources,
        conversation_id=conv_id,
    )


# ── Streaming query ────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    SSE streaming RAG query.

    SSE events:
      data: {"token": "..."}     — incremental text token
      data: {"done": true, "metadata": {...}, "sources": [...]}  — final event
      data: {"error": "..."}     — error event
    """
    settings = get_settings()

    try:
        system_prompt, contents, results, question = _run_rag(req)
    except HTTPException as e:
        async def error_gen():
            yield f"data: {json.dumps({'error': e.detail})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    conv_id = req.conversation_id or str(uuid.uuid4())
    t_start = time.time()

    sources = [
        {
            "document": p.get("document", ""),
            "page": p.get("page"),
            "section": p.get("section"),
            "relevance_score": round(s, 4),
            "text_preview": p.get("text", "")[:180],
        }
        for p, s in results
    ]

    async def event_generator() -> AsyncGenerator[str, None]:
        full_answer = ""
        try:
            for token in gemini_client.generate_stream(system_prompt, contents, question):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            latency_ms = int((time.time() - t_start) * 1000)
            memory.add_turn(conv_id, question, full_answer)

            done_event = {
                "done": True,
                "metadata": {
                    "model_used": settings.LLM_MODEL,
                    "latency_ms": latency_ms,
                    "chunks_retrieved": len(results),
                },
                "sources": sources,
                "conversation_id": conv_id,
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Static frontend (production) ───────────────────────────

# In Docker: WORKDIR=/app, frontend_build is at /app/frontend_build
# In dev: frontend is served by Vite on :5173
_static_candidates = [
    Path("/app/frontend_build"),           # Docker / Cloud Run
    Path(__file__).resolve().parent.parent.parent / "frontend_build",  # local relative
]

for _candidate in _static_candidates:
    if _candidate.exists():
        app.mount("/", StaticFiles(directory=str(_candidate), html=True), name="static")
        break
