# 📄 DocQA Assistant — Chat With Your Documents (RAG)

Upload PDFs, Word files or text documents and ask questions in plain language.
DocQA finds the most relevant passages and uses an LLM (OpenAI GPT or Anthropic Claude)
to write an answer **grounded in your documents, with citations to the exact file and page**.

> Screenshot: `docs/screenshot.png` *(add after running locally)*

## ✨ Features

- **Multi-format upload**: PDF, DOCX, TXT, Markdown, several files at once
- **Cited answers**: every answer references `[1]`, `[2]` and shows the source file + page
- **Two LLM providers**: OpenAI or Anthropic Claude, switchable from the sidebar
- **Two retrieval modes**
  - *Semantic*: OpenAI embeddings + cosine similarity (finds meaning, not just keywords)
  - *Keyword*: built-in BM25 search that runs fully offline, no extra API key
- **Search-only mode**: try it with no API key at all; it returns the best-matching passages
- **Hallucination guard**: the model is told to answer only from the context and to say so when the answer isn't there
- **Multi-turn chat** with short conversation memory
- **Tested core**: unit tests cover loading, chunking, retrieval and prompting, with no network needed

## 🏗️ How it works

```
Upload ──► Extract text (per page) ──► Split into overlapping chunks
                                              │
                                              ▼
Question ──► Retrieve top-k chunks (embeddings or BM25)
                                              │
                                              ▼
            Prompt = numbered context + question ──► LLM ──► Answer with [n] citations
```

| Module | Responsibility |
|---|---|
| `docqa/loaders.py` | Extracts text from PDF / DOCX / TXT and keeps page numbers |
| `docqa/chunking.py` | Sentence-aware chunking with overlap |
| `docqa/retriever.py` | `EmbeddingRetriever` (OpenAI) and `BM25Retriever` (offline) |
| `docqa/qa.py` | Prompt building and a provider-agnostic `LLM` wrapper |
| `app.py` | Streamlit chat UI |

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/docqa-assistant.git
cd docqa-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY (optional)
streamlit run app.py
```

Open http://localhost:8501, upload `samples/employee_handbook.pdf`, and try:

- *How many vacation days do full-time employees get?*
- *What's the home office stipend?*
- *Do flights need to be economy class?*

## 🧪 Tests

```bash
pip install pytest
pytest
```

## 🔧 Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI chat + embeddings |
| `ANTHROPIC_API_KEY` | — | Claude chat |
| `OPENAI_MODEL` | `gpt-4o-mini` | Override the OpenAI chat model |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Override the Claude model |

Chunk size and passages-per-answer can be tuned live in the sidebar.

## 🛣️ Possible extensions

- Persistent vector store (FAISS, Chroma, pgvector) for large document sets
- OCR for scanned PDFs (Tesseract)
- Hybrid search (BM25 + embeddings) with re-ranking
- FastAPI backend + Docker for deployment, user authentication
- Slack / website chat widget front end

## 📜 License

MIT
