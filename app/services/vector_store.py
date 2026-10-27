"""Persistent FAISS index of normalized embeddings (inner product == cosine).

Vector ids are the SQLite chunk row ids. A manifest stored next to the index
records the embedding model and dimension so an index built by one model is
never queried with another.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import faiss
import numpy as np

from app.services.document_store import now_iso

logger = logging.getLogger(__name__)


class IncompatibleIndexError(RuntimeError):
    pass


class FaissVectorStore:
    INDEX_FILE = "index.faiss"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, directory: Path, model_id: str, dimension: int | None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._lock = threading.RLock()
        self.manifest: dict = {}
        self.index: faiss.Index | None = None
        self._dimension = dimension
        self._load()

    @property
    def index_path(self) -> Path:
        return self.directory / self.INDEX_FILE

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST_FILE

    def _load(self) -> None:
        if self.manifest_path.exists() and self.index_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded FAISS index with %d vectors (%s)", self.index.ntotal, self.manifest.get("model_id"))

    @property
    def compatible(self) -> bool:
        return self.index is None or self.manifest.get("model_id") == self.model_id

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            self.manifest = {"model_id": self.model_id, "dimension": dimension, "created_at": now_iso()}
        elif not self.compatible:
            raise IncompatibleIndexError(
                f"Index was built with {self.manifest.get('model_id')}; current model is {self.model_id}. "
                "Rebuild the index before adding or searching."
            )

    def _save(self) -> None:
        self.manifest.update(count=self.size, updated_at=now_iso())
        tmp_index = self.index_path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp_index))
        os.replace(tmp_index, self.index_path)
        tmp_manifest = self.manifest_path.with_suffix(".tmp")
        tmp_manifest.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp_manifest, self.manifest_path)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        with self._lock:
            self._ensure_index(vectors.shape[1])
            id_array = np.asarray(ids, dtype="int64")
            self.index.remove_ids(id_array)  # idempotent re-adds never duplicate
            self.index.add_with_ids(vectors, id_array)
            self._save()

    def remove(self, ids: list[int]) -> int:
        if self.index is None or not ids:
            return 0
        with self._lock:
            removed = int(self.index.remove_ids(np.asarray(ids, dtype="int64")))
            self._save()
            return removed

    def search(self, query: np.ndarray, k: int, allowed_ids: list[int] | None = None) -> list[tuple[int, float]]:
        if self.index is None or self.size == 0 or k <= 0:
            return []
        if not self.compatible:
            raise IncompatibleIndexError(
                f"Index was built with {self.manifest.get('model_id')}; current model is {self.model_id}."
            )
        query = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        params = None
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(allowed_ids, dtype="int64")))
            k = min(k, len(allowed_ids))
        k = min(k, self.size)
        with self._lock:
            scores, ids = self.index.search(query, k, params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def ids(self) -> set[int]:
        if self.index is None:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).tolist())

    def reset(self, model_id: str | None = None) -> None:
        with self._lock:
            self.model_id = model_id or self.model_id
            self.index, self.manifest = None, {}
            for path in (self.index_path, self.manifest_path):
                path.unlink(missing_ok=True)
