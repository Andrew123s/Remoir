"""End-to-end ingestion: validate -> extract -> chunk -> embed -> store."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.security import internal_filename, safe_join, validate_upload
from app.services import chunker, pdf_processor
from app.services.document_store import DocumentStore
from app.services.embedding_service import Embedder
from app.services.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionOutcome:
    document_id: str
    status: str  # "indexed", "duplicate", "failed", "ocr_required"
    message: str
    duplicate_of: str | None = None


class IngestionService:
    def __init__(self, settings: Settings, store: DocumentStore, vectors: FaissVectorStore, embedder: Embedder):
        self.settings = settings
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.upload_dir = settings.data_path / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_bytes(self) -> int:
        return int(self.settings.ingestion.max_upload_mb * 1024 * 1024)

    def ingest_bytes(self, filename: str | None, data: bytes, content_type: str | None = "application/pdf") -> IngestionOutcome:
        """Ingest one PDF. Raises UploadRejected for invalid uploads; processing failures are recorded."""
        display = validate_upload(filename, content_type, data, self.max_bytes)
        content_hash = pdf_processor.file_hash(data)
        existing = self.store.find_by_hash(content_hash)
        if existing:
            return IngestionOutcome(
                existing["id"], "duplicate",
                f"This file was already uploaded as '{existing['filename']}'.", duplicate_of=existing["id"],
            )
        stored_name = internal_filename()
        safe_join(self.upload_dir, stored_name).write_bytes(data)
        doc_id = self.store.create_document(display, content_hash, len(data), stored_name)
        return self._process(doc_id, data)

    def _process(self, doc_id: str, data: bytes) -> IngestionOutcome:
        started = time.perf_counter()
        try:
            extraction = pdf_processor.extract_pdf(data, self.settings.ingestion, self.settings.ocr)
        except pdf_processor.PDFProcessingError as exc:
            self.store.update_document(doc_id, status="failed", error_code=exc.code, error=exc.message)
            return IngestionOutcome(doc_id, "failed", exc.message)
        except Exception:
            logger.exception("Unexpected extraction failure for %s", doc_id)
            msg = "Unexpected error while reading the PDF."
            self.store.update_document(doc_id, status="failed", error_code="extraction_error", error=msg)
            return IngestionOutcome(doc_id, "failed", msg)

        text_hash = extraction.text_hash
        if text_hash:
            twin = self.store.find_by_text_hash(text_hash, doc_id)
            if twin:
                self._discard(doc_id)
                return IngestionOutcome(
                    twin["id"], "duplicate",
                    f"The same text is already indexed as '{twin['filename']}' (different file bytes).",
                    duplicate_of=twin["id"],
                )
        meta = extraction.metadata
        pages_summary = [
            {"page": p.number, "label": p.label, "status": p.status, "chars": p.char_count, "source": p.source}
            for p in extraction.pages
        ]
        self.store.update_document(
            doc_id, text_hash=text_hash, title=meta.title, title_source=meta.title_source, authors=meta.authors, year=meta.year,
            year_source=meta.year_source, page_count=meta.page_count, warnings=extraction.warnings,
            pages=pages_summary, ocr_applied=int(extraction.ocr_applied), ocr_required=int(extraction.ocr_required),
        )
        if extraction.ocr_required:
            msg = (
                "The PDF has little or no extractable text (likely scanned). OCR is required; enable it in "
                "config (ocr.enabled) after installing Tesseract."
            )
            self.store.update_document(doc_id, status="ocr_required", error_code="ocr_required", error=msg)
            return IngestionOutcome(doc_id, "ocr_required", msg)

        try:
            count = self._token_counter()
            chunks = chunker.chunk_pages(extraction.pages, self.settings.chunking, count)
            if not chunks:
                msg = "No indexable text was found in the PDF."
                self.store.update_document(doc_id, status="failed", error_code="no_text", error=msg)
                return IngestionOutcome(doc_id, "failed", msg)
            sources = {p.number: p.source for p in extraction.pages}
            rows = [
                {
                    "chunk_index": c.index, "text": c.text, "page_start": c.page_start, "page_end": c.page_end,
                    "section": c.section, "token_count": c.token_count, "page_spans": c.page_spans,
                    "extraction_source": "ocr" if any(sources[p] == "ocr" for p in c.pages) else "native",
                }
                for c in chunks
            ]
            ids = self.store.add_chunks(doc_id, rows)
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
            self.vectors.add(ids, embeddings)
        except Exception:
            logger.exception("Indexing failed for %s", doc_id)
            self.vectors.remove(self.store.remove_chunks(doc_id))
            msg = "Indexing failed while chunking or embedding the document."
            self.store.update_document(doc_id, status="failed", error_code="indexing_error", error=msg)
            return IngestionOutcome(doc_id, "failed", msg)

        elapsed = time.perf_counter() - started
        self.store.update_document(
            doc_id, status="indexed", chunk_count=len(chunks), embedding_model=self.embedder.model_id,
            processing_seconds=round(elapsed, 3), error_code=None, error=None,
        )
        logger.info("Indexed document %s: %d pages, %d chunks in %.2fs", doc_id, meta.page_count, len(chunks), elapsed)
        return IngestionOutcome(doc_id, "indexed", f"Indexed {len(chunks)} chunks from {meta.page_count} pages.")

    def _token_counter(self):
        def count(text: str) -> int:
            try:
                return self.embedder.count_tokens(text)
            except Exception:
                return chunker.approx_token_count(text)

        return count

    def delete(self, doc_id: str) -> bool:
        if not self.store.get_document(doc_id):
            return False
        self.vectors.remove(self.store.remove_chunks(doc_id))
        self._discard(doc_id)
        return True

    def _discard(self, doc_id: str) -> None:
        """Remove the document row and its stored file."""
        doc = self.store.get_document(doc_id)
        self.store.delete_document(doc_id)
        if doc and doc.get("stored_name"):
            safe_join(self.upload_dir, doc["stored_name"]).unlink(missing_ok=True)

    def stored_pdf_path(self, doc_id: str) -> Path | None:
        doc = self.store.get_document(doc_id)
        if not doc or not doc.get("stored_name"):
            return None
        path = safe_join(self.upload_dir, doc["stored_name"])
        return path if path.exists() else None

    def rebuild_index(self, reparse: bool = False) -> dict:
        """Rebuild all vectors with the current embedder.

        With `reparse=True` documents are re-extracted and re-chunked from the stored
        PDFs (needed after chunking changes); otherwise stored chunk text is re-embedded.
        """
        started = time.perf_counter()
        self.vectors.reset(self.embedder.model_id)
        docs = self.store.list_documents()
        reindexed, failed = 0, 0
        for doc in docs:
            if reparse:
                path = self.stored_pdf_path(doc["id"])
                if not path:
                    failed += 1
                    continue
                self.store.remove_chunks(doc["id"])
                outcome = self._process(doc["id"], path.read_bytes())
                reindexed += outcome.status == "indexed"
                failed += outcome.status == "failed"
            else:
                chunks = self.store.document_chunks(doc["id"])
                if chunks:
                    self.vectors.add([c["id"] for c in chunks], self.embedder.embed_documents([c["text"] for c in chunks]))
                    self.store.update_document(doc["id"], embedding_model=self.embedder.model_id)
                    reindexed += 1
        return {
            "documents_reindexed": reindexed, "documents_failed": failed,
            "vectors": self.vectors.size, "seconds": round(time.perf_counter() - started, 2),
        }
