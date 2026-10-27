"""Streamlit UI: upload documents, ask questions, get cited answers.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from docqa import BM25Retriever, EmbeddingRetriever, LLM, answer_question, chunk_pages, load_document

load_dotenv()
st.set_page_config(page_title="DocQA Assistant", page_icon="📄", layout="wide")

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "LLM provider",
        ["openai", "anthropic", "none (search only)"],
        help="'search only' needs no API key: it shows the most relevant passages.",
    )
    key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    api_key = ""
    if key_env:
        api_key = st.text_input(f"{key_env}", value=os.getenv(key_env, ""), type="password")

    retrieval_options = ["Keyword (BM25, offline)"]
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        retrieval_options.insert(0, "Semantic (OpenAI embeddings)")
    retrieval = st.selectbox("Retrieval", retrieval_options)

    top_k = st.slider("Passages per answer", 2, 8, 4)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, step=100)

    files = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("📄 DocQA Assistant")
st.caption("Ask questions about your documents. Answers cite the exact file and page.")

if not files:
    st.info("Upload one or more PDF, DOCX, TXT or MD files in the sidebar to get started.")
    st.stop()


# ---------------- Build the index (cached per file set + settings) ----------------
@st.cache_resource(show_spinner="Reading and indexing documents...")
def build_index(fingerprint: str, _files, chunk_size: int, retrieval: str, openai_key: str):
    pages = []
    for f in _files:
        pages.extend(load_document(f.name, f.getvalue()))
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=min(150, chunk_size // 4))
    if not chunks:
        return None, 0, 0
    if retrieval.startswith("Semantic"):
        from openai import OpenAI

        retriever = EmbeddingRetriever(chunks, OpenAI(api_key=openai_key))
    else:
        retriever = BM25Retriever(chunks)
    return retriever, len(pages), len(chunks)


fingerprint = hashlib.sha256(b"".join(hashlib.sha256(f.getvalue()).digest() for f in files)).hexdigest()
openai_key = api_key if provider == "openai" else os.getenv("OPENAI_API_KEY", "")
try:
    retriever, n_pages, n_chunks = build_index(fingerprint, files, chunk_size, retrieval, openai_key)
except Exception as e:  # noqa: BLE001 - show any indexing problem to the user
    st.error(f"Could not index the documents: {e}")
    st.stop()

if retriever is None:
    st.warning("No readable text was found. Scanned PDFs need OCR first.")
    st.stop()
st.success(f"Indexed {len(files)} file(s): {n_pages} page(s), {n_chunks} chunks.")

# ---------------- Chat ----------------
st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for i, src in enumerate(msg.get("sources", []), start=1):
            with st.expander(f"[{i}] {src['source']} — page {src['page']}"):
                st.write(src["text"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if key_env and not api_key:
                raise RuntimeError(f"Please enter your {key_env} in the sidebar.")
            if key_env:
                llm = LLM(provider, api_key)
                history = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
                ][-6:]
                with st.spinner("Thinking..."):
                    result = answer_question(question, retriever, llm, k=top_k, history=history)
                text, sources = result.text, result.sources
            else:
                hits = retriever.search(question, k=top_k)
                sources = [c for c, _ in hits]
                text = "Most relevant passages:" if sources else "No matching passages found."
        except Exception as e:  # noqa: BLE001
            text, sources = f"⚠️ {e}", []

        st.markdown(text)
        src_dicts = [{"source": c.source, "page": c.page, "text": c.text} for c in sources]
        for i, src in enumerate(src_dicts, start=1):
            with st.expander(f"[{i}] {src['source']} — page {src['page']}"):
                st.write(src["text"])
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": src_dicts})
