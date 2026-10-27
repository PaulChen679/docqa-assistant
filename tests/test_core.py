"""Unit tests that run offline (no API keys needed). Run: pytest"""
import io

import docx
import pytest

from docqa import BM25Retriever, Page, answer_question, chunk_pages, load_document
from docqa.qa import build_user_prompt


def make_docx(text: str) -> bytes:
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_load_txt_and_docx():
    assert load_document("a.txt", b"hello world")[0].text == "hello world"
    pages = load_document("b.docx", make_docx("Line one\nLine two"))
    assert "Line two" in pages[0].text


def test_unsupported_type():
    with pytest.raises(ValueError):
        load_document("x.exe", b"")


def test_chunking_keeps_metadata_and_overlaps():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_pages([Page("doc.pdf", 3, text)], chunk_size=300, overlap=50)
    assert len(chunks) > 5
    assert all(c.source == "doc.pdf" and c.page == 3 for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_bm25_finds_relevant_chunk():
    pages = [
        Page("policy.pdf", 1, "Employees receive 20 days of paid vacation per year."),
        Page("policy.pdf", 2, "The office is closed on public holidays."),
        Page("policy.pdf", 3, "Expense reports must be submitted within 30 days."),
    ]
    r = BM25Retriever(chunk_pages(pages))
    top, _ = r.search("How many vacation days do employees get?", k=1)[0]
    assert top.page == 1


class FakeLLM:
    def __init__(self):
        self.user = None

    def complete(self, system, user, history=None):
        self.user = user
        return "Employees get 20 days [1]."


def test_answer_question_with_fake_llm():
    pages = [Page("policy.pdf", 1, "Employees receive 20 days of paid vacation per year.")]
    llm = FakeLLM()
    ans = answer_question("vacation days?", BM25Retriever(chunk_pages(pages)), llm)
    assert ans.text.endswith("[1].")
    assert "(source: policy.pdf, page 1)" in llm.user
    assert ans.sources[0].page == 1


def test_no_hits_returns_message_without_calling_llm():
    pages = [Page("a.txt", 1, "Completely unrelated text about cooking pasta.")]
    llm = FakeLLM()
    ans = answer_question("quantum entanglement", BM25Retriever(chunk_pages(pages)), llm)
    assert ans.sources == [] and llm.user is None


def test_prompt_numbers_sources():
    pages = [Page("a.pdf", 1, "alpha"), Page("b.pdf", 2, "beta")]
    chunks = chunk_pages(pages)
    prompt = build_user_prompt("q", [(chunks[0], 1.0), (chunks[1], 0.5)])
    assert "[1] (source: a.pdf, page 1)" in prompt and "[2] (source: b.pdf, page 2)" in prompt


def test_bm25_matches_word_variants():
    pages = [
        Page("h.pdf", 1, "Employees receive 20 days of paid vacation."),
        Page("h.pdf", 2, "Meals during travel are reimbursed up to 60 USD per day."),
    ]
    top, _ = BM25Retriever(chunk_pages(pages)).search("meal reimbursement limit", k=1)[0]
    assert top.page == 2


def test_embedding_retriever_with_fake_client():
    from types import SimpleNamespace

    from docqa import EmbeddingRetriever

    vocab = ["vacation", "expense", "remote"]

    class FakeEmbeddings:
        def create(self, model, input):
            vec = lambda t: [float(w in t.lower()) for w in vocab]
            return SimpleNamespace(data=[SimpleNamespace(embedding=vec(t)) for t in input])

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    pages = [Page("a", 1, "vacation policy"), Page("a", 2, "expense policy"), Page("a", 3, "remote work")]
    r = EmbeddingRetriever(chunk_pages(pages), client)
    assert r.search("expense rules", k=1)[0][0].page == 2
