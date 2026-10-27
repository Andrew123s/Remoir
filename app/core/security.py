"""Upload validation, safe filenames and optional API-key authentication."""

from __future__ import annotations

import hmac
import os
import re
import unicodedata
import uuid
from pathlib import Path

from fastapi import Header, HTTPException, status

PDF_MAGIC = b"%PDF-"
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", None, ""}


class UploadRejected(ValueError):
    """Raised when an upload fails validation. The message is safe to show users."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def sanitize_display_name(filename: str | None, max_length: int = 150) -> str:
    """Return a filename safe for display and metadata: no directories, control or odd characters."""
    name = unicodedata.normalize("NFKC", filename or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\- ()\[\]]", "_", name).strip(" .")
    name = re.sub(r"\.{2,}", ".", name)
    if not name:
        name = "document.pdf"
    if len(name) > max_length:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_length - len(ext) - 1] + "." + ext) if dot else name[:max_length]
    return name


def validate_upload(filename: str | None, content_type: str | None, data: bytes, max_bytes: int) -> str:
    """Validate an uploaded file and return its sanitized display name."""
    display = sanitize_display_name(filename)
    if Path(display).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadRejected("Only PDF files are supported.", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise UploadRejected("Only PDF files are supported.", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    if not data:
        raise UploadRejected("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadRejected(
            f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )
    if not data[:1024].lstrip().startswith(PDF_MAGIC):
        raise UploadRejected("The file is not a valid PDF (missing PDF header).", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    return display


def internal_filename() -> str:
    """Storage name that never derives from user input."""
    return f"{uuid.uuid4().hex}.pdf"


def safe_join(base: Path, name: str) -> Path:
    """Join `name` to `base`, refusing anything that escapes `base`."""
    base = base.resolve()
    target = (base / name).resolve()
    if base != target.parent and base not in target.parents:
        raise UploadRejected("Invalid path.")
    return target


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If REMOIR_API_KEY is set, every protected endpoint requires a matching X-API-Key header."""
    expected = os.environ.get("REMOIR_API_KEY")
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")
