"""FastAPI application entry point: `uvicorn app.main:app`."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api import chat, collections, documents, system
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.security import UploadRejected
from app.services.retrieval_service import InvalidQuestionError
from app.services.vector_store import IncompatibleIndexError

logger = logging.getLogger("remoir")


def create_app() -> FastAPI:
    setup_logging(get_settings().app.log_level)
    app = FastAPI(
        title="Remoir API",
        version=__version__,
        description="Research assistant with document retrieval (RAG), grounded answers and verified citations.",
    )
    for module in (system, documents, chat, collections):
        app.include_router(module.router)

    @app.exception_handler(UploadRejected)
    async def upload_rejected(_: Request, exc: UploadRejected):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(InvalidQuestionError)
    async def invalid_question(_: Request, exc: InvalidQuestionError):
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)})

    @app.exception_handler(IncompatibleIndexError)
    async def incompatible_index(_: Request, exc: IncompatibleIndexError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "The vector index was built with a different embedding model. Rebuild the index."},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
