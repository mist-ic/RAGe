"""
RAGe — Gemini 3 Flash Client

Wrapper around google-genai for streaming and non-streaming generation.
"""

import time
from typing import Generator, List, Optional, Dict

from google import genai
from google.genai import types

from app.config import get_settings


def _client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def generate(
    system_prompt: str,
    contents: List[Dict],
    question: str,
    max_retries: int = 3,
) -> str:
    """
    Non-streaming Gemini 3 Flash generation.

    Args:
        system_prompt: The grounded system instruction.
        contents: Prior conversation turns (Gemini format).
        question: The current user question.
        max_retries: Retry count on transient errors.

    Returns:
        The assistant's text response.
    """
    settings = get_settings()
    client = _client()

    # Append current question
    full_contents = contents + [{
        "role": "user",
        "parts": [{"text": question}],
    }]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=full_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    # Per models.md: do NOT set temperature for Gemini 3 (default=1.0)
                ),
            )
            return response.text or ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def generate_stream(
    system_prompt: str,
    contents: List[Dict],
    question: str,
) -> Generator[str, None, None]:
    """
    Streaming Gemini 3 Flash generation — yields text chunks.

    Args:
        system_prompt: The grounded system instruction.
        contents: Prior conversation turns (Gemini format).
        question: The current user question.

    Yields:
        Text delta strings as they arrive.
    """
    settings = get_settings()
    client = _client()

    full_contents = contents + [{
        "role": "user",
        "parts": [{"text": question}],
    }]

    response = client.models.generate_content_stream(
        model=settings.LLM_MODEL,
        contents=full_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text
