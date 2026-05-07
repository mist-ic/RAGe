"""
RAGe — Configuration

Pydantic settings loaded from environment variables.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings."""

    # --- API Keys ---
    GEMINI_API_KEY: str

    # --- Server ---
    PORT: int = 8000

    # --- Models ---
    LLM_MODEL: str = "gemini-3-flash-preview"
    EMBED_MODEL: str = "gemini-embedding-2"
    EMBED_DIMENSIONS: int = 768

    # --- RAG Pipeline ---
    CHUNK_SIZE: int = 400       # max tokens per chunk (~4 chars/token)
    CHUNK_OVERLAP: int = 60     # overlap tokens between adjacent chunks
    TOP_K: int = 5              # top chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.3  # min cosine similarity

    # --- Paths ---
    UPLOAD_DIR: str = str(PROJECT_ROOT / "backend" / "uploads")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Returns cached Settings instance."""
    return Settings()
