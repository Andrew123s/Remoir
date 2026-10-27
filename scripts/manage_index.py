"""Back up, restore, rebuild or inspect the local index.

    python scripts/manage_index.py status
    python scripts/manage_index.py backup                 # -> backups/<timestamp>/
    python scripts/manage_index.py restore backups/<timestamp>
    python scripts/manage_index.py rebuild [--reparse]    # re-embed (or re-extract) everything

Stop the API server before `restore` so files are not replaced while in use.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.dependencies import Container  # noqa: E402

DB = Path("processed") / "remoir.db"
VECTORS = Path("vector_store")


def backup(data: Path, target_root: Path) -> Path:
    target = target_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    (target / DB.parent).mkdir(parents=True)
    # SQLite online backup API gives a consistent copy even with WAL enabled
    src = sqlite3.connect(data / DB)
    dst = sqlite3.connect(target / DB)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    shutil.copytree(data / VECTORS, target / VECTORS)
    shutil.copytree(data / "uploads", target / "uploads")
    return target


def restore(data: Path, source: Path) -> None:
    for sub in (DB.parent, VECTORS, Path("uploads")):
        if not (source / sub).exists():
            raise SystemExit(f"Backup is incomplete: missing {sub}")
    for sub in (DB.parent, VECTORS, Path("uploads")):
        shutil.rmtree(data / sub, ignore_errors=True)
        shutil.copytree(source / sub, data / sub)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    b = sub.add_parser("backup")
    b.add_argument("--to", type=Path, default=ROOT / "backups")
    r = sub.add_parser("restore")
    r.add_argument("source", type=Path)
    rb = sub.add_parser("rebuild")
    rb.add_argument("--reparse", action="store_true", help="re-extract and re-chunk the stored PDFs")
    args = parser.parse_args()
    setup_logging("WARNING")
    settings = load_settings()
    data = settings.data_path

    if args.cmd == "backup":
        print(f"Backup written to {backup(data, args.to)}")
    elif args.cmd == "restore":
        restore(data, args.source)
        print("Restored. Start the API and check /health (index_consistent should be true).")
    else:
        container = Container(settings)
        if args.cmd == "rebuild":
            print(container.ingestion.rebuild_index(reparse=args.reparse))
        docs = container.store.list_documents()
        print(f"Documents: {len(docs)} ({sum(d['status'] == 'indexed' for d in docs)} indexed)")
        print(f"Vectors: {container.vectors.size} | chunks: {container.store.chunk_count()} | consistent: {container.index_consistent()}")
        print(f"Index model: {container.vectors.manifest.get('model_id')} | current model: {container.embedder.model_id}"
              f" | compatible: {container.vectors.compatible}")
        container.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
