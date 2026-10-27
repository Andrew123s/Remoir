"""Document upload, listing, inspection and deletion."""

from __future__ import annotations

import pymupdf as fitz
from fastapi import APIRouter, Depends, File, HTTPException, Path, Request, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.core.security import UploadRejected, require_api_key
from app.dependencies import Container, get_container
from app.schemas import DocumentDetail, DocumentInfo, MessageResponse, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])

DOC_ID = Path(pattern=r"^[a-f0-9]{16}$")


def _doc_or_404(c: Container, document_id: str) -> dict:
    doc = c.store.get_document(document_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return doc


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise UploadRejected(f"File exceeds the {limit // (1024 * 1024)} MB upload limit.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...), c: Container = Depends(get_container)):
    """Upload a PDF; it is validated, extracted, chunked, embedded and indexed synchronously."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > c.ingestion.max_bytes + 1024 * 1024:
        raise UploadRejected(f"File exceeds the {c.ingestion.max_bytes // (1024 * 1024)} MB upload limit.", 413)
    data = await _read_limited(file, c.ingestion.max_bytes)
    with c.lock:
        outcome = c.ingestion.ingest_bytes(file.filename, data, file.content_type)
        c.retriever.invalidate()
    doc = c.store.get_document(outcome.document_id)
    return UploadResponse(
        document_id=outcome.document_id, status=outcome.status, message=outcome.message,
        duplicate_of=outcome.duplicate_of, document=DocumentInfo(**doc) if doc else None,
    )


@router.get("", response_model=list[DocumentInfo])
def list_documents(c: Container = Depends(get_container)):
    return [DocumentInfo(**d) for d in c.store.list_documents()]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str = DOC_ID, c: Container = Depends(get_container)):
    return DocumentDetail(**_doc_or_404(c, document_id))


@router.delete("/{document_id}", response_model=MessageResponse)
def delete_document(document_id: str = DOC_ID, c: Container = Depends(get_container)):
    _doc_or_404(c, document_id)
    with c.lock:
        c.ingestion.delete(document_id)
        c.retriever.invalidate()
    return MessageResponse(message="Document, chunks and vectors deleted.")


@router.get("/{document_id}/chunks")
def get_document_chunks(document_id: str = DOC_ID, c: Container = Depends(get_container)):
    _doc_or_404(c, document_id)
    return [
        {k: ch[k] for k in ("chunk_id", "chunk_index", "page_start", "page_end", "section", "token_count", "text")}
        for ch in c.store.document_chunks(document_id)
    ]


@router.get("/{document_id}/file")
def download_document(document_id: str = DOC_ID, c: Container = Depends(get_container)):
    doc = _doc_or_404(c, document_id)
    path = c.ingestion.stored_pdf_path(document_id)
    if not path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Original file is not available.")
    return FileResponse(path, media_type="application/pdf", filename=doc["filename"])


@router.get("/{document_id}/pages/{page}/image")
def page_image(document_id: str = DOC_ID, page: int = Path(ge=1, le=10000), c: Container = Depends(get_container)):
    """Render one PDF page (1-based physical index) as PNG for the source viewer."""
    _doc_or_404(c, document_id)
    path = c.ingestion.stored_pdf_path(document_id)
    if not path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Original file is not available.")
    with fitz.open(path) as pdf:
        if page > pdf.page_count:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Page out of range.")
        png = pdf[page - 1].get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).tobytes("png")
    return Response(content=png, media_type="image/png")
