"""SQLite store for documents, chunks, collections and conversations.

Chunk rows use an INTEGER primary key that doubles as the FAISS vector id,
which keeps vectors and metadata associated across restarts and deletions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_name TEXT,
    title TEXT,
    title_source TEXT,
    authors TEXT,
    year INTEGER,
    year_source TEXT,
    page_count INTEGER DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    error_code TEXT,
    error TEXT,
    warnings TEXT DEFAULT '[]',
    pages TEXT DEFAULT '[]',
    chunk_count INTEGER DEFAULT 0,
    embedding_model TEXT,
    ocr_applied INTEGER DEFAULT 0,
    ocr_required INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    processing_seconds REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section TEXT,
    token_count INTEGER,
    page_spans TEXT NOT NULL,
    extraction_source TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collection_documents (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (collection_id, document_id)
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    document_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);
"""

_JSON_FIELDS = {"warnings", "pages", "page_spans", "document_ids"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key in _JSON_FIELDS & out.keys():
        out[key] = json.loads(out[key]) if out[key] else []
    return out


class DocumentStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        self._conn.close()

    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_row(r) for r in self._conn.execute(sql, params).fetchall()]

    # Documents -----------------------------------------------------------
    def create_document(self, filename: str, content_hash: str, size_bytes: int, stored_name: str | None) -> str:
        doc_id = uuid.uuid4().hex[:16]
        self._exec(
            "INSERT INTO documents (id, filename, stored_name, content_hash, size_bytes, status, uploaded_at)"
            " VALUES (?, ?, ?, ?, ?, 'processing', ?)",
            (doc_id, filename, stored_name, content_hash, size_bytes, now_iso()),
        )
        return doc_id

    def update_document(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        values = [json.dumps(v) if k in _JSON_FIELDS else v for k, v in fields.items()]
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE documents SET {assignments} WHERE id = ?", [*values, doc_id])

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return rows[0] if rows else None

    def find_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM documents WHERE content_hash = ?", (content_hash,))
        return rows[0] if rows else None

    def list_documents(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM documents ORDER BY uploaded_at DESC, filename")

    def delete_document(self, doc_id: str) -> list[int]:
        """Delete a document and its chunks; returns the vector ids that must be removed."""
        with self._lock, self._conn:
            ids = [r[0] for r in self._conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return ids

    # Chunks --------------------------------------------------------------
    def add_chunks(self, doc_id: str, chunks: list[dict[str, Any]]) -> list[int]:
        ids = []
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            for c in chunks:
                cur = self._conn.execute(
                    "INSERT INTO chunks (chunk_id, document_id, chunk_index, text, page_start, page_end, section,"
                    " token_count, page_spans, extraction_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{doc_id}:{c['chunk_index']:04d}", doc_id, c["chunk_index"], c["text"], c["page_start"],
                        c["page_end"], c.get("section"), c.get("token_count"), json.dumps(c["page_spans"]),
                        c.get("extraction_source", "native"),
                    ),
                )
                ids.append(cur.lastrowid)
        return ids

    def remove_chunks(self, doc_id: str) -> list[int]:
        with self._lock, self._conn:
            ids = [r[0] for r in self._conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        return ids

    def get_chunks(self, vector_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not vector_ids:
            return {}
        marks = ",".join("?" * len(vector_ids))
        rows = self._all(
            f"SELECT c.*, d.filename, d.title, d.authors, d.year, d.year_source FROM chunks c"
            f" JOIN documents d ON d.id = c.document_id WHERE c.id IN ({marks})",
            list(vector_ids),
        )
        return {r["id"]: r for r in rows}

    def chunk_ids_for_documents(self, doc_ids: list[str]) -> list[int]:
        marks = ",".join("?" * len(doc_ids))
        return [r["id"] for r in self._all(f"SELECT id FROM chunks WHERE document_id IN ({marks})", doc_ids)]

    def all_chunks(self) -> list[dict[str, Any]]:
        return self._all("SELECT id, document_id, text FROM chunks ORDER BY id")

    def document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,))

    def chunk_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # Collections ---------------------------------------------------------
    def create_collection(self, name: str) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)", (cid, name, now_iso()))
        return self.get_collection(cid)

    def get_collection(self, cid: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM collections WHERE id = ?", (cid,))
        if not rows:
            return None
        rows[0]["document_ids"] = [
            r["document_id"] for r in self._all("SELECT document_id FROM collection_documents WHERE collection_id = ?", (cid,))
        ]
        return rows[0]

    def list_collections(self) -> list[dict[str, Any]]:
        return [self.get_collection(r["id"]) for r in self._all("SELECT id FROM collections ORDER BY name")]

    def set_collection_documents(self, cid: str, doc_ids: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM collection_documents WHERE collection_id = ?", (cid,))
            self._conn.executemany(
                "INSERT INTO collection_documents (collection_id, document_id) VALUES (?, ?)",
                [(cid, d) for d in dict.fromkeys(doc_ids)],
            )

    def delete_collection(self, cid: str) -> bool:
        return self._exec("DELETE FROM collections WHERE id = ?", (cid,)).rowcount > 0

    # Conversations -------------------------------------------------------
    def create_conversation(self) -> str:
        cid = uuid.uuid4().hex
        self._exec("INSERT INTO conversations (id, created_at) VALUES (?, ?)", (cid, now_iso()))
        return cid

    def conversation_exists(self, cid: str) -> bool:
        return bool(self._all("SELECT id FROM conversations WHERE id = ?", (cid,)))

    def add_turn(self, cid: str, role: str, content: str, document_ids: list[str] | None = None) -> None:
        self._exec(
            "INSERT INTO turns (conversation_id, role, content, document_ids, created_at) VALUES (?, ?, ?, ?, ?)",
            (cid, role, content, json.dumps(document_ids or []), now_iso()),
        )

    def get_turns(self, cid: str, limit: int) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?", (cid, limit))
        return list(reversed(rows))

    def delete_conversation(self, cid: str) -> bool:
        return self._exec("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0
