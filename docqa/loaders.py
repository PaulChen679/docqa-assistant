"""Load text from uploaded documents (PDF, DOCX, TXT, MD), keeping page numbers."""
from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class Page:
    source: str  # file name
    page: int    # 1-based page number (DOCX/TXT are treated as a single page)
    text: str


def load_document(name: str, data: bytes) -> list[Page]:
    """Extract text from a file given its name and raw bytes."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "pdf":
        return _load_pdf(name, data)
    if ext == "docx":
        return _load_docx(name, data)
    if ext in {"txt", "md"}:
        return [Page(name, 1, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {name}")


def _load_pdf(name: str, data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = (p.extract_text() or "").strip()
        if text:
            pages.append(Page(name, i, text))
    return pages


def _load_docx(name: str, data: bytes) -> list[Page]:
    import docx

    document = docx.Document(io.BytesIO(data))
    text = "\n".join(par.text for par in document.paragraphs if par.text.strip())
    return [Page(name, 1, text)] if text else []
