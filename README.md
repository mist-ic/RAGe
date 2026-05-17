# RAGe

Upload any document and chat with it. Built from scratch with Gemini 3 Flash, Gemini Embedding 2, and Qdrant.

**[Live Demo](https://rage-411746695116.asia-south1.run.app)**

---

## What It Does

Upload a PDF or text file. RAGe chunks it, embeds every chunk using Gemini Embedding 2, and stores the vectors in Qdrant. Ask a question -- the system retrieves the most relevant chunks, **grades each one for relevance**, decides whether to use retrieved context, fall back to web search, or combine both, then generates a grounded answer using Gemini 3 Flash that cites exact pages and sections.

The LLM is strictly forbidden from using its training knowledge. Every answer comes from your document (or a grounded web search when the document doesn't cover the question).

---

## Corrective RAG (CRAG)

Implements **"Corrective Retrieval Augmented Generation"** (Yan et al., 2024) in [`backend/app/pipeline/evaluator.py`](backend/app/pipeline/evaluator.py).

```
Retrieve → Grade each chunk → Decide action → [Web search] → Generate
```

After standard retrieval, each chunk is independently graded by the LLM:

| Label | Meaning |
|---|---|
| `correct` | Chunk is relevant to the query -- keep it |
| `ambiguous` | Partially relevant -- keep, but supplement |
| `incorrect` | Not relevant -- discard |

The grader's verdict drives one of three actions:

| Action | When | Knowledge source |
|---|---|---|
| `use_retrieved` | All/most chunks correct | Retrieved document chunks only |
| `use_web` | All chunks incorrect | Web search (Gemini `google_search` tool) |
| `use_both` | Mixed relevance | Kept chunks + web search supplement |

Irrelevant chunks are stripped before generation. The CRAG action, chunk labels, and whether web search fired are all surfaced in the API response and visible in the UI.

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 3 Flash (`gemini-3-flash-preview`) |
| Embeddings | Gemini Embedding 2 (`gemini-embedding-2`, 768 dims) |
| Vector DB | Qdrant (in-memory, per-session) |
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + TypeScript + Vite |
| Hosting | GCP Cloud Run (`asia-south1`) |

---

## Chunking Strategy

Structure-aware paragraph chunking in [`backend/app/pipeline/chunker.py`](backend/app/pipeline/chunker.py):

1. **Section heading detection** -- title-case lines (2-10 words, no trailing punctuation) mark new sections
2. **Paragraph splitting** -- body text split on double newlines
3. **Small-chunk merging** -- consecutive short paragraphs merged up to 400 tokens to avoid noisy tiny chunks
4. **Oversized splitting** -- paragraphs over 400 tokens split by sentence boundaries
5. **Overlap** -- 60-token tail of each chunk prepended to the next for context continuity
6. **Post-merge pass** -- final scan merges any remaining sub-80-token fragments

Embeddings use task-formatted prompts per Gemini Embedding 2 docs:
- Chunks: `title: {filename} | text: {content}`
- Queries: `task: question answering | query: {question}`

For PDFs of 6 pages or fewer, each page is also embedded as raw PDF binary via the Gemini Embedding 2 multimodal API, giving the model visual layout context alongside text chunks.

---

## Quick Start (Local)

**Prerequisites:** Python 3.11+, Node.js 20+

```bash
git clone https://github.com/mist-ic/RAGe.git
cd RAGe

# Backend
cp .env.example .env       # add GEMINI_API_KEY
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                 # proxies API to :8000
```

Open http://localhost:5173

---

## Deploy

```bash
# One-command deploy to Cloud Run
npm run deploy:gcloud
```

Requires `gcloud` CLI authenticated and `GEMINI_API_KEY` in `.env`.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Multipart upload PDF/TXT. Returns `session_id`. |
| `POST` | `/query/stream` | SSE streaming RAG query. Body: `{question, session_id}`. |
| `POST` | `/query` | Non-streaming RAG query. Same body. |
| `GET` | `/health` | Health check. |

---

## Project Structure

```
RAGe/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes
│   │   ├── config.py            # Settings
│   │   ├── gemini_client.py     # Gemini 3 Flash wrapper
│   │   ├── pipeline/
│   │   │   ├── extractor.py     # PDF + text extraction (PyMuPDF)
│   │   │   ├── chunker.py       # Structure-aware chunking
│   │   │   ├── embedder.py      # Gemini Embedding 2 API
│   │   │   ├── retriever.py     # Qdrant vector search
│   │   │   └── prompt.py        # Grounded prompt builder
│   │   └── memory/
│   │       └── conversation.py  # Multi-turn memory
│   └── requirements.txt
├── frontend/                    # React 18 + TypeScript
├── Dockerfile                   # Multi-stage build
└── scripts/write-gcloud-env.mjs
```

---

## Author

Praveen Kumar - 24bcs10048
