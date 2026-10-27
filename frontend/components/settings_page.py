"""Retrieval Settings page."""

from __future__ import annotations

import streamlit as st

from frontend import api_client
from frontend.api_client import APIError

EMBEDDING_MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"]


def page() -> None:
    st.title("Retrieval Settings")
    try:
        current = api_client.get("/settings")
    except APIError as exc:
        st.error(exc.detail)
        return
    s = current["settings"]
    if current["reindex_required"]:
        st.error("The vector index was built with a different embedding model. Rebuild the index below before searching.")

    with st.form("settings"):
        st.subheader("Retrieval")
        c1, c2, c3 = st.columns(3)
        top_k = c1.slider("Passages retrieved (top-k)", 1, 20, s["top_k"])
        strategy = c2.selectbox("Strategy", ["hybrid", "dense"], index=["hybrid", "dense"].index(s["strategy"]),
                                help="hybrid = dense vectors + BM25 fused with reciprocal rank fusion")
        rerank = c3.toggle("Cross-encoder reranking", s["rerank"],
                           help="Downloads cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB) on first use; slower on CPU.")
        st.subheader("Chunking (applies to new uploads, or after a re-parse rebuild)")
        c1, c2 = st.columns(2)
        chunk_size = c1.slider("Chunk size (tokens)", 100, 500, s["chunk_size"], step=10)
        chunk_overlap = c2.slider("Chunk overlap (tokens)", 0, 150, s["chunk_overlap"], step=5)
        st.subheader("Models")
        options = EMBEDDING_MODELS if s["embedding_model"] in EMBEDDING_MODELS else EMBEDDING_MODELS + [s["embedding_model"]]
        embedding = st.selectbox("Embedding model", options, index=options.index(s["embedding_model"]),
                                 help="Changing it makes the existing index incompatible; a rebuild is required.")
        c1, c2 = st.columns(2)
        providers = ["extractive", "ollama", "openai"]
        provider = c1.selectbox("LLM provider", providers, index=providers.index(s["llm_provider"]),
                                help="extractive works offline; ollama needs a local Ollama server; openai needs REMOIR_LLM_API_KEY on the backend.")
        model = c2.text_input("LLM model identifier", s["llm_model"])
        if embedding != s["embedding_model"]:
            st.warning("Changing the embedding model will require re-indexing every document.")
        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            try:
                r = api_client.put("/settings", json={
                    "top_k": top_k, "strategy": strategy, "rerank": rerank, "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "embedding_model": embedding, "llm_provider": provider,
                    "llm_model": model,
                })
                st.success("Settings saved.")
                for m in r["messages"]:
                    st.warning(m)
            except APIError as exc:
                st.error(exc.detail)

    health = api_client.health() or {}
    if health.get("llm"):
        llm = health["llm"]
        (st.success if llm.get("available") else st.warning)(
            f"LLM provider **{llm['provider']}** ({llm['model']}): {'available' if llm.get('available') else llm.get('detail') or 'unavailable'}"
        )

    st.subheader("Index maintenance")
    st.caption(f"Vectors: {health.get('vectors', '?')} · embedding model: `{health.get('embedding_model', '?')}` · "
               f"consistent with metadata: {health.get('index_consistent')}")
    c1, c2 = st.columns(2)
    if c1.button("Rebuild index (re-embed stored chunks)"):
        with st.spinner("Re-embedding…"):
            st.json(api_client.post("/index/rebuild", json={"reparse": False}))
    if c2.button("Rebuild with re-parsing (apply new chunking)"):
        with st.spinner("Re-extracting and re-indexing every PDF…"):
            st.json(api_client.post("/index/rebuild", json={"reparse": True}))
