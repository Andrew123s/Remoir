"""Remoir Streamlit frontend: `streamlit run frontend/streamlit_app.py`.

Talks to the FastAPI backend at REMOIR_API_URL (default http://127.0.0.1:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend import api_client  # noqa: E402
from frontend.components import (  # noqa: E402
    chat_interface,
    document_manager,
    evaluation_dashboard,
    settings_page,
    source_viewer,
)

st.set_page_config(page_title="Remoir — Research Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      code {font-size: 0.85em;}
      [data-testid="stSidebar"] {min-width: 260px;}
    </style>
    """,
    unsafe_allow_html=True,
)

health = api_client.health()
with st.sidebar:
    st.markdown("## 📚 Remoir")
    st.caption("Research assistant with document retrieval and verified citations")
    if health is None:
        st.error(f"Backend not reachable at {api_client.API_URL}. Start it with `uvicorn app.main:app`.")
    else:
        st.success(f"Backend online · {health['documents']} documents · {health['vectors']} vectors")
        llm = health["llm"]
        st.caption(f"Answers: **{llm['provider']}** ({llm['model']})" + ("" if llm.get("available") else " — unavailable"))
        st.caption(f"Embeddings: `{health['embedding_model']}`")
        if not health["index_compatible"]:
            st.warning("Index built with a different embedding model. Rebuild it in Retrieval Settings.")
    st.divider()
    st.caption("Local single-user application. Do not expose it to the internet without authentication.")

pages = [
    st.Page(chat_interface.page, title="Research Chat", icon="💬", url_path="chat", default=True),
    st.Page(document_manager.page, title="Document Library", icon="📄", url_path="library"),
    st.Page(source_viewer.page, title="Source Explorer", icon="🔎", url_path="sources"),
    st.Page(settings_page.page, title="Retrieval Settings", icon="⚙️", url_path="settings"),
    st.Page(evaluation_dashboard.page, title="Evaluation Dashboard", icon="📊", url_path="evaluation"),
]
if health is None:
    st.title("Remoir")
    st.error("The backend API is not running. Start it first:")
    st.code(".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000", language="powershell")
    st.stop()
st.navigation(pages).run()
