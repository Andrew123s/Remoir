"""Prepare a fresh checkout and report what is available.

    python scripts/initialize.py              # create data folders, check environment
    python scripts/initialize.py --download   # also pre-download the embedding model (~130 MB)
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings  # noqa: E402

REQUIRED = ["fastapi", "uvicorn", "pymupdf", "pypdf", "sentence_transformers", "faiss", "streamlit", "numpy", "pandas"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download and cache the configured embedding model")
    args = parser.parse_args()
    settings = load_settings()
    ok = True

    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  ! Python 3.11 or newer is required")
        ok = False

    for sub in ("uploads", "processed", "vector_store"):
        (settings.data_path / sub).mkdir(parents=True, exist_ok=True)
    print("Data folders ready under the configured data directory")

    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    print("Packages: " + ("all installed" if not missing else f"MISSING {missing} -> pip install -r requirements.txt"))
    ok &= not missing

    print(f"Embedding model: {settings.embedding.model_name}")
    print(f"LLM provider: {settings.llm.provider} (model {settings.llm.model})")
    try:
        tags = httpx.get(f"{settings.llm.ollama_url}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        print(f"Ollama: running, models: {names or 'none pulled'}")
    except Exception:
        print("Ollama: not reachable (optional). Install from https://ollama.com, then `ollama pull llama3.2:3b` (~2 GB).")
    print("Tesseract OCR: " + ("found" if shutil.which("tesseract") else "not found (optional, needed only for scanned PDFs)"))

    if args.download:
        print(f"Downloading {settings.embedding.model_name} (about 130 MB for bge-small)...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding.model_name, device="cpu")
        print(f"  cached; dimension {model.get_sentence_embedding_dimension()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
