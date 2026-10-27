"""Document Library page: upload, inspect, delete, and organise collections."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.api_client import APIError

STATUS_ICON = {"indexed": "✅", "duplicate": "🔁", "failed": "❌", "ocr_required": "🖼️", "processing": "⏳"}


def page() -> None:
    st.title("Document Library")

    with st.container(border=True):
        st.subheader("Upload PDFs")
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True,
                                 key=f"uploader-{st.session_state.get('upload_round', 0)}")
        if files and st.button("Process and index", type="primary"):
            for f in files:
                with st.spinner(f"Processing {f.name}…"):
                    try:
                        r = api_client.post("/documents/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
                        icon = STATUS_ICON.get(r["status"], "")
                        msg = f"{icon} **{f.name}**: {r['message']}"
                        {"indexed": st.success, "duplicate": st.info}.get(r["status"], st.warning)(msg)
                    except APIError as exc:
                        st.error(f"❌ **{f.name}**: {exc.detail}")
            st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
            st.button("Done")

    try:
        docs = api_client.get("/documents")
    except APIError as exc:
        st.error(exc.detail)
        return

    st.subheader(f"Documents ({len(docs)})")
    if not docs:
        st.info("No documents uploaded yet.")
        return
    table = pd.DataFrame([
        {"status": f"{STATUS_ICON.get(d['status'], '')} {d['status']}", "filename": d["filename"],
         "title": d["title"] or "—", "pages": d["page_count"], "chunks": d["chunk_count"],
         "size KB": round(d["size_bytes"] / 1024, 1), "uploaded": d["uploaded_at"][:19].replace("T", " "),
         "seconds": d["processing_seconds"]}
        for d in docs
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)

    failures = [d for d in docs if d["status"] in {"failed", "ocr_required"}]
    if failures:
        st.warning("Some documents could not be indexed:")
        for d in failures:
            st.write(f"- **{d['filename']}** — {d['error']}")

    for d in docs:
        with st.expander(f"{STATUS_ICON.get(d['status'], '')} {d['filename']}"):
            detail = api_client.get(f"/documents/{d['id']}")
            c1, c2 = st.columns(2)
            c1.markdown(
                f"**Document ID:** `{d['id']}`  \n"
                f"**Title:** {d['title'] or '_not available_'}"
                + (f" _(from {d['title_source'].replace('_', ' ')})_" if d.get("title_source") else "")
                + f"  \n**Authors:** {d['authors'] or '_not available_'}  \n"
                f"**Year:** {d['year'] or '_not available_'}"
                + (" _(PDF creation date, not necessarily publication year)_" if d.get("year_source") else "")
            )
            c2.markdown(
                f"**Status:** {d['status']}  \n**Embedding model:** `{d['embedding_model'] or '—'}`  \n"
                f"**SHA-256:** `{d['content_hash'][:16]}…`  \n**OCR applied:** {'yes' if d['ocr_applied'] else 'no'}"
            )
            for w in d["warnings"]:
                st.caption(f"⚠️ {w}")
            if detail["pages"]:
                pages = pd.DataFrame(detail["pages"])
                st.caption("Per-page extraction")
                st.dataframe(pages, hide_index=True, use_container_width=True, height=min(35 * len(pages) + 40, 300))
            if st.button("Delete document", key=f"del-{d['id']}", type="secondary"):
                st.session_state["confirm_delete"] = d["id"]
            if st.session_state.get("confirm_delete") == d["id"]:
                st.error("Delete this document, its chunks and vectors? This cannot be undone.")
                a, b = st.columns(2)
                if a.button("Yes, delete", key=f"yes-{d['id']}", type="primary"):
                    api_client.delete(f"/documents/{d['id']}")
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
                if b.button("Cancel", key=f"no-{d['id']}"):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

    st.subheader("Collections")
    indexed = [d for d in docs if d["status"] == "indexed"]
    with st.form("new_collection", clear_on_submit=True):
        name = st.text_input("Collection name")
        members = st.multiselect("Documents", indexed, format_func=lambda d: d["filename"])
        if st.form_submit_button("Create collection") and name.strip():
            try:
                api_client.post("/collections", json={"name": name, "document_ids": [m["id"] for m in members]})
                st.rerun()
            except APIError as exc:
                st.error(exc.detail)
    names = {d["id"]: d["filename"] for d in docs}
    for coll in api_client.get("/collections"):
        c1, c2 = st.columns([5, 1])
        c1.write(f"**{coll['name']}** — {', '.join(names.get(i, i) for i in coll['document_ids']) or 'empty'}")
        if c2.button("Delete", key=f"coll-{coll['id']}"):
            api_client.delete(f"/collections/{coll['id']}")
            st.rerun()
