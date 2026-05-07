"""
RAGe — PDF and Text Extractor

Extracts text from PDF files (per-page) using PyMuPDF and handles plain text.
Also returns raw PDF bytes per page for multimodal embedding.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class Page:
    """One page of extracted text from a document."""
    filename: str
    page_number: int     # 1-indexed
    text: str
    pdf_bytes: Optional[bytes] = None  # raw page PDF for multimodal embedding


def extract_pdf(filepath: str) -> List[Page]:
    """
    Extract text from a PDF file, one Page object per PDF page.

    Also captures each page as a standalone PDF (bytes) for Gemini Embedding 2
    multimodal embedding.

    Args:
        filepath: Absolute path to the PDF file.

    Returns:
        List of Page objects sorted by page number.
    """
    pages: List[Page] = []
    filename = Path(filepath).name

    doc = fitz.open(filepath)
    total_pages = len(doc)

    for i in range(total_pages):
        page = doc[i]
        text = page.get_text().strip()
        if not text:
            continue

        # Extract this page as a standalone single-page PDF (bytes)
        # Used for multimodal Gemini Embedding 2 input
        page_pdf_bytes: Optional[bytes] = None
        try:
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=i, to_page=i)
            page_pdf_bytes = single_page_doc.tobytes()
            single_page_doc.close()
        except Exception:
            pass  # not critical — text embedding will still work

        pages.append(Page(
            filename=filename,
            page_number=i + 1,
            text=text,
            pdf_bytes=page_pdf_bytes,
        ))

    doc.close()
    return pages


def extract_text_file(filepath: str) -> List[Page]:
    """
    Read a plain text file as a single 'page'.

    Args:
        filepath: Absolute path to the .txt file.

    Returns:
        List with a single Page object.
    """
    filename = Path(filepath).name
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if not text:
        return []

    return [Page(filename=filename, page_number=1, text=text)]


def extract_document(filepath: str) -> List[Page]:
    """
    Dispatch to the correct extractor based on file extension.

    Args:
        filepath: Absolute path to a PDF or TXT file.

    Returns:
        List of Page objects.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in (".txt", ".md"):
        return extract_text_file(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")
