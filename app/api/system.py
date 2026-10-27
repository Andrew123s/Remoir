"""Health, runtime settings, index maintenance and evaluation results."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status

from app import __version__
from app.core.config import PROJECT_ROOT
from app.core.security import require_api_key
from app.dependencies import ALLOWED_EMBEDDING_MODELS, Container, get_container
from app.schemas import HealthResponse, RebuildRequest, RuntimeSettings, SettingsResponse

router = APIRouter(tags=["system"])
protected = [Depends(require_api_key)]
EVALUATION_RESULTS = PROJECT_ROOT / "reports" / "evaluation_results.json"


@router.get("/health", response_model=HealthResponse)
def health(c: Container = Depends(get_container)):
    """Liveness and dependency status. Never includes secrets or filesystem paths."""
    return HealthResponse(
        status="ok",
        version=__version__,
        documents=len(c.store.list_documents()),
        vectors=c.vectors.size,
        embedding_model=c.embedder.model_id,
        index_compatible=c.vectors.compatible,
        index_consistent=c.index_consistent(),
        llm=c.provider.status(),
        ocr_enabled=c.settings.ocr.enabled,
    )


@router.get("/settings", response_model=SettingsResponse, dependencies=protected)
def get_runtime_settings(c: Container = Depends(get_container)):
    return SettingsResponse(settings=RuntimeSettings(**c.runtime_settings()), reindex_required=not c.vectors.compatible,
                            messages=[f"Allowed embedding models: {', '.join(ALLOWED_EMBEDDING_MODELS)}"])


@router.put("/settings", response_model=SettingsResponse, dependencies=protected)
def put_runtime_settings(req: RuntimeSettings, c: Container = Depends(get_container)):
    try:
        reindex, messages = c.update_runtime_settings(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))
    return SettingsResponse(settings=RuntimeSettings(**c.runtime_settings()), reindex_required=reindex, messages=messages)


@router.post("/index/rebuild", dependencies=protected)
def rebuild_index(req: RebuildRequest, c: Container = Depends(get_container)):
    """Re-embed all chunks with the current model, or re-parse every stored PDF when `reparse` is true."""
    with c.lock:
        result = c.ingestion.rebuild_index(reparse=req.reparse)
        c.retriever.invalidate()
    return result


@router.get("/evaluation/latest", dependencies=protected)
def latest_evaluation():
    if not EVALUATION_RESULTS.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No evaluation has been run yet.")
    return json.loads(EVALUATION_RESULTS.read_text(encoding="utf-8"))
