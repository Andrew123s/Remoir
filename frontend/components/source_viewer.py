"""Source Explorer page and shared passage rendering."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.api_client import APIError


def page_label(p: dict) -> str:
    label = p["pages"]
    if p.get("printed_page_labels"):
        label += f" (printed: {', '.join(p['printed_page_labels'])})"
    return label


def quote_passage(text: str) -> str:
    """Blockquote the passage verbatim: escape Markdown so headings like '2. Results' are not turned into lists."""
    escaped = re.sub(r"([\\`*_{}\[\]<>#|~])", r"\\\1", text)
    lines = [re.sub(r"^(\d+)\.", r"\1\\.", ln) for ln in escaped.split("\n")]
    return "> " + "  \n> ".join(lines)


def render_passage(p: dict, show_scores: bool = True) -> None:
    """Show filename, page, section, metadata and the passage, with an optional page preview."""
    meta = [f"**{p['filename']}**", page_label(p)]
    if p.get("section"):
        meta.append(f"section: _{p['section']}_")
    st.markdown(" · ".join(meta))
    details = []
    if p.get("title"):
        details.append(f"Title: {p['title']}")
    if p.get("authors"):
        details.append(f"Authors: {p['authors']}")
    if p.get("extraction_source") == "ocr":
        details.append("Text source: OCR")
    if details:
        st.caption(" | ".join(details))
    st.markdown(quote_passage(p["text"]))
    if show_scores:
        scores = f"score {p['score']:.4f} · dense {p['dense_score']:.3f} · lexical {p['lexical_score']:.2f}"
        if p.get("rerank_score") is not None:
            scores += f" · rerank {p['rerank_score']:.2f}"
        st.caption(f"{scores} · chunk `{p['chunk_id']}` · page numbers are PDF page indices")
    # an expander opens client-side, so it does not rerun the script and close a surrounding popover
    with st.expander("View original PDF page"):
        for n in range(p["page_start"], p["page_end"] + 1):
            try:
                st.image(page_png(p["document_id"], n), caption=f"{p['filename']} — PDF page {n}")
            except APIError as exc:
                st.info(f"Page preview unavailable: {exc.detail}")


@st.cache_data(max_entries=64, show_spinner=False)
def page_png(document_id: str, page: int) -> bytes:
    return api_client.get(f"/documents/{document_id}/pages/{page}/image")


def render_answer_with_citations(answer: str) -> str:
    """Make [S1] markers visually distinct in markdown."""
    return re.sub(r"\[(S\d+)\]", r" `\1`", answer)


def page() -> None:
    st.title("Source Explorer")
    st.caption("Retrieve passages directly, without generating an answer. Useful for checking what the assistant sees.")
    try:
        docs = [d for d in api_client.get("/documents") if d["status"] == "indexed"]
    except APIError as exc:
        st.error(exc.detail)
        return

    with st.form("search"):
        query = st.text_input("Search query", placeholder="e.g. limitations of the study")
        col1, col2, col3 = st.columns([3, 1, 1])
        selected = col1.multiselect("Restrict to documents", docs, format_func=lambda d: d["filename"])
        top_k = col2.number_input("Passages", 1, 20, 6)
        strategy = col3.selectbox("Strategy", ["hybrid", "dense"])
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and query.strip():
        try:
            st.session_state["explorer_results"] = api_client.post("/search", json={
                "query": query, "document_ids": [d["id"] for d in selected] or None,
                "retrieval": {"top_k": int(top_k), "strategy": strategy},
            })
        except APIError as exc:
            st.error(exc.detail)

    result = st.session_state.get("explorer_results")
    last_answer = st.session_state.get("last_response")
    if result:
        st.subheader(f"{len(result['results'])} passages for “{result['query']}”")
        st.caption(" · ".join(f"{k}: {v:.0f} ms" for k, v in result["timings_ms"].items()))
        if result["results"]:
            table = pd.DataFrame([
                {"id": r["source_id"], "document": r["filename"], "pages": page_label(r), "section": r["section"],
                 "score": r["score"], "dense": r["dense_score"], "lexical": r["lexical_score"]}
                for r in result["results"]
            ])
            st.dataframe(table, hide_index=True, use_container_width=True)
        for r in result["results"]:
            with st.container(border=True):
                st.markdown(f"**[{r['source_id']}]**")
                render_passage(r)
    elif last_answer:
        st.subheader("Evidence retrieved for the last chat question")
        for r in last_answer["evidence"]:
            with st.container(border=True):
                render_passage(r)
    else:
        st.info("Run a search, or ask a question in Research Chat to inspect its evidence here.")
