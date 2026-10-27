"""Retrievers: OpenAI embeddings (semantic) or BM25 (keyword, works offline with no API key)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


_STOPWORDS = set(
    "a an and are as at be by do does for from how i in is it my of on or the to was what when where "
    "which who why with you your".split()
)
_SUFFIXES = ("ements", "ement", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "s")


def _stem(word: str) -> str:
    """Very light suffix stripping so 'meals'/'meal' and 'reimbursed'/'reimbursement' match."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Classic Okapi BM25. No external services needed."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                if t not in doc:
                    continue
                tf = doc[t]
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                s += self.idf[t] * tf * (self.k1 + 1) / denom
            scores.append(s)
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


class EmbeddingRetriever:
    """Semantic search with OpenAI embeddings and cosine similarity (numpy)."""

    def __init__(self, chunks: list[Chunk], client, model: str = "text-embedding-3-small", batch: int = 100):
        self.chunks = chunks
        self.client, self.model = client, model
        vectors = []
        for i in range(0, len(chunks), batch):
            vectors.extend(self._embed([c.text for c in chunks[i:i + batch]]))
        self.matrix = self._normalize(np.array(vectors, dtype=np.float32))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self._normalize(np.array(self._embed([query])[0], dtype=np.float32))
        sims = self.matrix @ q
        order = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order]
