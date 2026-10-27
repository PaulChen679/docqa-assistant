"""Split pages into overlapping chunks while keeping source/page metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Page


@dataclass
class Chunk:
    id: int
    source: str
    page: int
    text: str


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split each page into ~chunk_size character chunks, breaking on sentence boundaries."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = re.sub(r"\s+", " ", page.text).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer to cut at the last sentence end inside the window.
                cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
                if cut > start + chunk_size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(len(chunks), page.source, page.page, piece))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
