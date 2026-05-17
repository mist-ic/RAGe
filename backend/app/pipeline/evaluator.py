"""
RAGe — Corrective RAG (CRAG) Evaluator

Implements the evaluation step from:
  "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

Pipeline:
  1. Retrieve candidate chunks from Qdrant (existing step)
  2. Evaluate each chunk's relevance to the query via LLM grader
  3. Decide action based on aggregate score:
       CORRECT   — all/most chunks relevant   → use retrieved context
       INCORRECT — all/most chunks irrelevant → discard, fall back to web search
       AMBIGUOUS — mixed relevance            → keep relevant chunks + web search
  4. Refine the knowledge set accordingly
  5. Generate answer from corrected context (handled in main pipeline)
"""

import json
from enum import Enum
from typing import List, Tuple

from google import genai
from google.genai import types

from app.config import get_settings


# ── Relevance labels ───────────────────────────────────────

class RelevanceLabel(str, Enum):
    CORRECT   = "correct"    # chunk is relevant to the query
    INCORRECT = "incorrect"  # chunk is not relevant
    AMBIGUOUS = "ambiguous"  # partially relevant / uncertain


class CRAGAction(str, Enum):
    """Which knowledge source(s) to use for generation."""
    USE_RETRIEVED    = "use_retrieved"    # standard RAG path
    USE_WEB          = "use_web"          # all retrieved irrelevant, use web only
    USE_BOTH         = "use_both"         # mixed, use relevant chunks + web


# ── Grader prompt ──────────────────────────────────────────

_GRADER_SYSTEM = (
    "You are a strict document relevance grader. "
    "You determine whether a retrieved text chunk is useful for answering a user query. "
    "Reply with a single JSON object and nothing else."
)

_GRADER_TEMPLATE = """Query: {query}

Retrieved chunk:
\"\"\"{chunk}\"\"\"

Is this chunk relevant to answering the query?
Reply ONLY with this JSON (no markdown):
{{"label": "<correct|incorrect|ambiguous>", "reason": "<one short sentence>"}}"""


# ── Core functions ─────────────────────────────────────────

def grade_chunk(query: str, chunk_text: str) -> Tuple[RelevanceLabel, str]:
    """
    Grade a single retrieved chunk against the user query.

    Uses Gemini Flash in non-streaming mode with low thinking to keep latency
    down. Returns (label, reason).
    """
    settings = get_settings()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    prompt = _GRADER_TEMPLATE.format(
        query=query,
        chunk=chunk_text[:800],   # cap to avoid token waste on huge chunks
    )

    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_GRADER_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level="none"),
        ),
    )

    raw = (response.text or "").strip()

    # Strip markdown fences if model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        label_str = parsed.get("label", "ambiguous").lower()
        reason = parsed.get("reason", "")
        label = RelevanceLabel(label_str) if label_str in RelevanceLabel._value2member_map_ else RelevanceLabel.AMBIGUOUS
        return label, reason
    except (json.JSONDecodeError, KeyError):
        return RelevanceLabel.AMBIGUOUS, "Could not parse grader response"


def evaluate_chunks(
    query: str,
    chunks: List[Tuple[dict, float]],
) -> List[Tuple[dict, float, RelevanceLabel, str]]:
    """
    Grade all retrieved chunks against the query.

    Returns list of (payload, score, label, reason) tuples.
    Grades up to 5 chunks to keep latency reasonable.
    """
    graded = []
    for payload, score in chunks[:5]:
        chunk_text = payload.get("text", "")
        label, reason = grade_chunk(query, chunk_text)
        graded.append((payload, score, label, reason))
    return graded


def decide_action(
    graded: List[Tuple[dict, float, RelevanceLabel, str]],
) -> Tuple[CRAGAction, List[Tuple[dict, float]]]:
    """
    Given graded chunks, decide which CRAG action to take and which
    chunks to keep.

    Decision logic (Yan et al., 2024):
      - If ANY chunk is CORRECT  → at least use retrieved
      - If ALL chunks INCORRECT  → USE_WEB only
      - If mix of correct + incorrect/ambiguous → USE_BOTH

    Returns:
        (action, kept_chunks)  where kept_chunks are the (payload, score)
        pairs that are CORRECT or AMBIGUOUS (irrelevant ones stripped).
    """
    if not graded:
        return CRAGAction.USE_WEB, []

    correct_chunks   = [(p, s) for p, s, l, _ in graded if l == RelevanceLabel.CORRECT]
    ambiguous_chunks = [(p, s) for p, s, l, _ in graded if l == RelevanceLabel.AMBIGUOUS]
    incorrect_chunks = [(p, s) for p, s, l, _ in graded if l == RelevanceLabel.INCORRECT]

    n_correct   = len(correct_chunks)
    n_ambiguous = len(ambiguous_chunks)
    n_incorrect = len(incorrect_chunks)

    # All irrelevant → web only
    if n_correct == 0 and n_ambiguous == 0:
        return CRAGAction.USE_WEB, []

    # All (or mostly) correct → standard RAG
    if n_incorrect == 0 and n_ambiguous <= 1:
        return CRAGAction.USE_RETRIEVED, correct_chunks + ambiguous_chunks

    # Mixed → use what we have + supplement with web
    kept = correct_chunks + ambiguous_chunks
    return CRAGAction.USE_BOTH, kept


def web_search(query: str) -> str:
    """
    Perform a grounded web search using Gemini's built-in google_search tool.

    Returns a concise summary string to be injected as supplementary context.
    This does not replace the LLM answer — it feeds web knowledge into the
    context block alongside any kept retrieved chunks.
    """
    settings = get_settings()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    search_prompt = (
        f"Search the web and provide a concise factual summary (3-5 sentences) "
        f"that directly answers or provides context for this question: {query}"
    )

    try:
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(thinking_level="none"),
            ),
        )
        return response.text or ""
    except Exception as e:
        return f"[Web search unavailable: {e}]"
