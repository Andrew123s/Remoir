"""Bulk-ingest every PDF in a folder into the local index (no server needed).

    python scripts/ingest_documents.py path\\to\\papers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.core.security import UploadRejected  # noqa: E402
from app.dependencies import Container  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    setup_logging("WARNING")
    if not args.folder.is_dir():
        print(f"Not a folder: {args.folder}")
        return 1
    pdfs = sorted(args.folder.rglob("*.pdf") if args.recursive else args.folder.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found.")
        return 1
    container = Container(load_settings())
    counts: dict[str, int] = {}
    for path in pdfs:
        try:
            outcome = container.ingestion.ingest_bytes(path.name, path.read_bytes())
            status, message = outcome.status, outcome.message
        except UploadRejected as exc:
            status, message = "rejected", str(exc)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status:12s}] {path.name}: {message}")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in counts.items()), f"| vectors in index: {container.vectors.size}")
    container.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
