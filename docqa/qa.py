"""Build a grounded prompt from retrieved chunks and ask an LLM (OpenAI or Anthropic Claude)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .chunking import Chunk

SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY using the provided context. "
    "Cite sources inline like [1], [2] matching the context numbers. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Answer in the same language as the question."
)


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(hits, start=1)
    )


def build_user_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    return f"Context:\n{build_context(hits)}\n\nQuestion: {question}"


class LLM:
    """Thin wrapper so the app can switch providers with one setting."""

    def __init__(self, provider: str, api_key: str, model: str | None = None):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, system: str, user: str, history: list[dict] | None = None) -> str:
        messages = list(history or []) + [{"role": "user", "content": user}]
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        resp = self.client.messages.create(
            model=self.model, system=system, messages=messages, max_tokens=1024, temperature=0.1
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def answer_question(question: str, retriever, llm: LLM, k: int = 4, history: list[dict] | None = None) -> Answer:
    hits = retriever.search(question, k=k)
    if not hits:
        return Answer("I could not find anything relevant to that question in the uploaded documents.", [])
    text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits), history)
    return Answer(text, [c for c, _ in hits])
