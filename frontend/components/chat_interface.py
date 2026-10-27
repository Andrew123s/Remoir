"""Research Chat page."""

from __future__ import annotations

import streamlit as st

from frontend import api_client
from frontend.api_client import APIError
from frontend.components.source_viewer import page_label, render_answer_with_citations, render_passage

GROUNDING_LABELS = {
    "grounded": ("All cited claims are supported by their passages", "success"),
    "partially_grounded": ("Some claims are weakly supported or had invalid citations", "warning"),
    "ungrounded": ("No valid citations — not supported by the sources", "error"),
    "no_claims": ("No factual claims detected", "info"),
}


def _reset_conversation() -> None:
    cid = st.session_state.get("conversation_id")
    if cid:
        try:
            api_client.delete(f"/chat/conversations/{cid}")
        except APIError:
            pass
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["last_response"] = None


def _render_response(resp: dict, idx: int) -> None:
    if resp["status"] == "insufficient_evidence":
        st.warning(resp["answer"])
    elif resp["status"] == "generation_failed":
        st.error(resp["answer"])
    else:
        st.markdown(render_answer_with_citations(resp["answer"]))

    if resp["citations"]:
        st.markdown("**Sources**")
        cols = st.columns(min(len(resp["citations"]), 4))
        for i, cit in enumerate(resp["citations"]):
            with cols[i % len(cols)].popover(f"[{cit['source_id']}] {cit['filename'][:28]} · {cit['pages']}", use_container_width=True):
                render_passage(cit)
        for cit in resp["citations"]:
            st.caption(f"[{cit['source_id']}] Document: {cit['filename']}, {page_label(cit)}"
                       + (f" — {cit['title']}" if cit.get("title") else ""))

    g = resp.get("grounding")
    if g:
        text, kind = GROUNDING_LABELS.get(g["status"], (g["status"], "info"))
        getattr(st, kind)(f"Citation check: {text}.", icon=None)
        flagged = [c for c in g["claims"] if not c["supported"]]
        if flagged or g["invalid_citations"]:
            with st.expander("Citation verification details"):
                if g["invalid_citations"]:
                    st.write("Removed citations to sources that were not retrieved:", ", ".join(g["invalid_citations"]))
                for c in flagged:
                    st.write(f"- _{c['issue']}_ (support {c['support_score']}): {c['sentence']}")

    with st.expander(f"Retrieved evidence ({len(resp['evidence'])} passages)"):
        a = resp["evidence_assessment"]
        st.caption(f"Evidence check: {a['reason']} (top similarity {a['top_dense_score']:.3f}, "
                   f"term coverage {a['term_coverage']:.0%})")
        if resp["retrieval_query"] and resp["retrieval_query"] != st.session_state.get("_last_q"):
            st.caption(f"Retrieval query: {resp['retrieval_query']}")
        for i, e in enumerate(resp["evidence"]):
            st.markdown(f"**{e['filename']}** — {page_label(e)} · dense {e['dense_score']:.3f}")
            st.caption(e["text"][:400] + ("…" if len(e["text"]) > 400 else ""))
    t = resp["timings_ms"]
    st.caption(f"{resp['provider']} · {resp['model']} · total {t.get('total_ms', 0):.0f} ms"
               + (f" · generation {t['generation_ms']:.0f} ms" if "generation_ms" in t else ""))


def page() -> None:
    st.title("Research Chat")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)

    try:
        docs = [d for d in api_client.get("/documents") if d["status"] == "indexed"]
        collections = api_client.get("/collections")
    except APIError as exc:
        st.error(f"Backend error: {exc.detail}")
        return

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])
        selected = col1.multiselect(
            "Answer using only these documents (empty = all)", docs, format_func=lambda d: d["filename"], key="chat_docs"
        )
        coll = col2.selectbox("Collection", [None] + collections, format_func=lambda c: "— none —" if c is None else c["name"])
        col3.write("")
        if col3.button("New conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

    if not docs:
        st.info("No indexed documents yet. Upload PDFs in the Document Library first.")

    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_response(msg["response"], idx)

    question = st.chat_input("Ask a question about your documents…", disabled=not docs)
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Retrieving passages and generating a grounded answer…", expanded=False) as status:
                try:
                    resp = api_client.post("/chat/ask", json={
                        "question": question,
                        "document_ids": [d["id"] for d in selected] or None,
                        "collection_id": coll["id"] if coll else None,
                        "conversation_id": st.session_state["conversation_id"],
                    })
                    status.update(label="Done", state="complete")
                except APIError as exc:
                    status.update(label="Failed", state="error")
                    st.error(exc.detail)
                    st.session_state["messages"].pop()
                    return
            st.session_state["conversation_id"] = resp["conversation_id"]
            st.session_state["last_response"] = resp
            st.session_state["_last_q"] = question
            _render_response(resp, len(st.session_state["messages"]))
        st.session_state["messages"].append({"role": "assistant", "response": resp})
