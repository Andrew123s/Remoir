"""Named document collections for multi-document research."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.core.security import require_api_key
from app.dependencies import Container, get_container
from app.schemas import CollectionCreate, CollectionOut, CollectionUpdate, MessageResponse

router = APIRouter(prefix="/collections", tags=["collections"], dependencies=[Depends(require_api_key)])
COLL_ID = Path(pattern=r"^[a-f0-9]{12}$")


def _validate_docs(c: Container, doc_ids: list[str]) -> None:
    unknown = [d for d in doc_ids if not c.store.get_document(d)]
    if unknown:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown document ids: {', '.join(unknown[:5])}")


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(req: CollectionCreate, c: Container = Depends(get_container)):
    _validate_docs(c, req.document_ids)
    try:
        coll = c.store.create_collection(req.name.strip())
    except sqlite3.IntegrityError:
        raise HTTPException(status.HTTP_409_CONFLICT, "A collection with this name already exists.")
    c.store.set_collection_documents(coll["id"], req.document_ids)
    return CollectionOut(**c.store.get_collection(coll["id"]))


@router.get("", response_model=list[CollectionOut])
def list_collections(c: Container = Depends(get_container)):
    return [CollectionOut(**coll) for coll in c.store.list_collections()]


@router.put("/{collection_id}", response_model=CollectionOut)
def update_collection(req: CollectionUpdate, collection_id: str = COLL_ID, c: Container = Depends(get_container)):
    if not c.store.get_collection(collection_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found.")
    _validate_docs(c, req.document_ids)
    c.store.set_collection_documents(collection_id, req.document_ids)
    return CollectionOut(**c.store.get_collection(collection_id))


@router.delete("/{collection_id}", response_model=MessageResponse)
def delete_collection(collection_id: str = COLL_ID, c: Container = Depends(get_container)):
    if not c.store.delete_collection(collection_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found.")
    return MessageResponse(message="Collection deleted (documents are kept).")
