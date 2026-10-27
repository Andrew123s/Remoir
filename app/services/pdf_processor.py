"""PDF text and metadata extraction with page-level provenance.

PyMuPDF is the primary extractor. If PyMuPDF fails on an individual page,
pypdf is tried for that page so one damaged page does not sink the document.
Optional OCR (PyMuPDF + local Tesseract) is used for pages without a text layer.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import statistics
from dataclasses import dataclass, field

import pymupdf as fitz

from app.core.config import IngestionSettings, OCRSettings
from app.services import text_cleaner

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Document-level failure with a user-safe message and machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Block:
    text: str
    is_heading: bool = False


@dataclass
class PageContent:
    number: int  # 1-based physical PDF page index
    blocks: list[Block]
    source: str = "native"  # "native", "pypdf", or "ocr"
    label: str | None = None  # printed page label when the PDF defines one
    status: str = "ok"  # "ok", "low_text", "failed"
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class DocumentMetadata:
    title: str | None = None
    title_source: str | None = None  # "pdf_metadata" or "first_page_heading"
    authors: str | None = None
    year: int | None = None
    # The PDF info dictionary has no publication year; we only know the file's
    # creation date, which is reported with this provenance label.
    year_source: str | None = None
    page_count: int = 0
    producer: str | None = None


@dataclass
class ExtractionResult:
    pages: list[PageContent]
    metadata: DocumentMetadata
    content_hash: str
    is_scanned: bool = False
    ocr_required: bool = False
    ocr_applied: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def text_pages(self) -> list[PageContent]:
        return [p for p in self.pages if p.status == "ok"]

    @property
    def text_hash(self) -> str | None:
        """Hash of the normalised extracted text: identifies the same document saved as different bytes."""
        text = " ".join(" ".join(p.text.split()) for p in self.pages).lower()
        return hashlib.sha256(text.encode("utf-8")).hexdigest() if text.strip() else None


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _year_from_pdf_date(value: str | None) -> int | None:
    match = re.match(r"(?:D:)?((?:19|20)\d{2})", value or "")
    return int(match.group(1)) if match else None


def _blocks_from_page(page: fitz.Page) -> list[tuple[str, float, bool]]:
    """Return (text, max_font_size, is_bold) per text block in reading order."""
    out = []
    data = page.get_text("dict", sort=True)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines, sizes, bold = [], [], True
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans)
            if text.strip():
                lines.append(text)
                sizes.extend(s.get("size", 0) for s in spans if s.get("text", "").strip())
                bold = bold and all(
                    (s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()
                    for s in spans if s.get("text", "").strip()
                )
        if lines:
            out.append(("\n".join(lines), max(sizes) if sizes else 0.0, bold))
    return out


def _looks_like_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    words = text.split()
    if not words or len(words) > 14 or len(text) > 120 or text.rstrip().endswith((".", ",", ";")):
        return False
    numbered = bool(re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+[A-Z]", text))
    return size >= body_size * 1.15 or (bold and (numbered or text[:1].isupper())) or numbered and size >= body_size


def _ocr_page(page: fitz.Page, settings: OCRSettings) -> list[tuple[str, float, bool]]:
    textpage = page.get_textpage_ocr(language=settings.language, dpi=settings.dpi, full=True)
    text = page.get_text("text", textpage=textpage)
    return [(para, 0.0, False) for para in re.split(r"\n\s*\n", text) if para.strip()]


def _pypdf_page_text(data: bytes, index: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return reader.pages[index].extract_text() or ""


def extract_pdf(data: bytes, ingestion: IngestionSettings, ocr: OCRSettings) -> ExtractionResult:
    """Extract page-level text and metadata. Raises PDFProcessingError for unusable documents."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for broken files
        raise PDFProcessingError("corrupted", "The PDF could not be opened; it may be corrupted.") from exc

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFProcessingError("encrypted", "The PDF is password-protected and cannot be processed.")
        if doc.page_count == 0:
            raise PDFProcessingError("empty", "The PDF contains no pages (it may be empty or corrupted).")

        raw_pages: list[list[tuple[str, float, bool]]] = []
        sources, labels, errors = [], [], []
        for page in doc:
            label = None
            try:
                label = page.get_label() or None
            except Exception:
                pass
            labels.append(label)
            try:
                raw_pages.append(_blocks_from_page(page))
                sources.append("native")
                errors.append(None)
            except Exception as exc:
                logger.warning("PyMuPDF failed on page %d, trying pypdf: %s", page.number + 1, exc)
                try:
                    text = _pypdf_page_text(data, page.number)
                    raw_pages.append([(p, 0.0, False) for p in re.split(r"\n\s*\n", text) if p.strip()])
                    sources.append("pypdf")
                    errors.append(None)
                except Exception as exc2:
                    raw_pages.append([])
                    sources.append("native")
                    errors.append(f"Text extraction failed: {type(exc2).__name__}")

        meta = doc.metadata or {}
        low_text = [sum(len(t) for t, _, _ in blocks) < ingestion.min_page_chars for blocks in raw_pages]
        is_scanned = sum(low_text) / len(raw_pages) >= ingestion.scanned_page_ratio
        warnings: list[str] = []
        ocr_applied = False
        if any(low_text) and ocr.enabled:
            for i, page in enumerate(doc):
                if not low_text[i]:
                    continue
                try:
                    raw_pages[i] = _ocr_page(page, ocr)
                    sources[i] = "ocr"
                    ocr_applied = True
                except Exception as exc:
                    warnings.append(f"OCR failed on page {i + 1}: {type(exc).__name__}. Is Tesseract installed?")
                    logger.warning("OCR failed on page %d: %s", i + 1, exc)
                    break  # a missing Tesseract fails identically on every page
            low_text = [sum(len(t) for t, _, _ in blocks) < ingestion.min_page_chars for blocks in raw_pages]

        sizes = [s for blocks in raw_pages for t, s, _ in blocks if s > 0 for _ in range(max(1, len(t) // 50))]
        body_size = statistics.median(sizes) if sizes else 10.0
        repeated = text_cleaner.find_repeated_lines([[t for t, _, _ in b] for b in raw_pages])

        pages: list[PageContent] = []
        for i, blocks in enumerate(raw_pages):
            cleaned: list[Block] = []
            for text, size, bold in blocks:
                if text.strip().lower() in repeated:
                    continue
                c = text_cleaner.clean_block(text)
                if c:
                    heading = sources[i] == "native" and _looks_like_heading(c, size, bold, body_size)
                    cleaned.append(Block(c, heading))
            page = PageContent(number=i + 1, blocks=cleaned, source=sources[i], label=labels[i])
            if errors[i]:
                page.status, page.error = "failed", errors[i]
            elif low_text[i] or text_cleaner.is_low_quality(page.text):
                page.status = "low_text"
            pages.append(page)

        metadata = DocumentMetadata(
            title=(meta.get("title") or "").strip() or None,
            authors=(meta.get("author") or "").strip() or None,
            year=_year_from_pdf_date(meta.get("creationDate")),
            page_count=doc.page_count,
            producer=(meta.get("producer") or "").strip() or None,
        )
        if metadata.year:
            metadata.year_source = "pdf_creation_date"
        if metadata.title:
            metadata.title_source = "pdf_metadata"
        else:
            first_heading = next((b.text for b in pages[0].blocks if b.is_heading), None)
            if first_heading and len(first_heading) > 8:
                metadata.title, metadata.title_source = first_heading, "first_page_heading"

    result = ExtractionResult(
        pages=pages,
        metadata=metadata,
        content_hash=file_hash(data),
        is_scanned=is_scanned and not ocr_applied,
        ocr_required=is_scanned and not ocr_applied,
        ocr_applied=ocr_applied,
        warnings=warnings,
    )
    failed = [p.number for p in pages if p.status == "failed"]
    if failed:
        result.warnings.append(f"Extraction failed on pages: {failed}")
    sparse = [p.number for p in pages if p.status == "low_text"]
    if sparse and not result.ocr_required:
        result.warnings.append(f"Little or no extractable text on pages: {sparse}")
    return result
