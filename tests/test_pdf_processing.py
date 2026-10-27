import pytest

from app.core.config import IngestionSettings, OCRSettings
from app.services import pdf_processor
from app.services.pdf_processor import PDFProcessingError, extract_pdf
from app.services.text_cleaner import clean_block, find_repeated_lines

ING, OCR = IngestionSettings(), OCRSettings()


def test_valid_pdf_extracts_pages_in_order(fixtures_dir):
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), ING, OCR)
    assert [p.number for p in result.pages] == [1, 2, 3]
    assert "panel fixed-effects" in result.pages[1].text
    assert "Limitations" in result.pages[2].text
    assert all(p.status == "ok" for p in result.pages)
    assert not result.is_scanned


def test_metadata_extraction(fixtures_dir):
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), ING, OCR)
    meta = result.metadata
    assert meta.title == "Rainfall Variability and Maize Yields in the Volta Highlands"
    assert meta.title_source == "pdf_metadata"
    assert "Mensah" in meta.authors
    assert meta.year == 2024 and meta.year_source == "pdf_creation_date"
    assert meta.page_count == 3


def test_missing_metadata_is_not_fabricated(fixtures_dir):
    result = extract_pdf((fixtures_dir / "ecosystem_services_review.pdf").read_bytes(), ING, OCR)
    assert result.metadata.authors is None
    assert result.metadata.year is None
    # title falls back to the first-page heading and is labelled as such
    assert result.metadata.title_source == "first_page_heading"
    assert result.metadata.title.startswith("Ecosystem Services")


def test_headings_detected(fixtures_dir):
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), ING, OCR)
    headings = [b.text for p in result.pages for b in p.blocks if b.is_heading]
    assert "2. Methodology" in headings and "4. Limitations" in headings


def test_running_page_numbers_removed(fixtures_dir):
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), ING, OCR)
    assert all(b.text.strip() not in {"1", "2", "3"} for p in result.pages for b in p.blocks)


def test_corrupted_pdf_raises_clear_error(fixtures_dir):
    with pytest.raises(PDFProcessingError) as exc:
        extract_pdf((fixtures_dir / "corrupted.pdf").read_bytes(), ING, OCR)
    assert exc.value.code in {"corrupted", "empty"}


def test_encrypted_pdf_detected(fixtures_dir):
    with pytest.raises(PDFProcessingError) as exc:
        extract_pdf((fixtures_dir / "encrypted.pdf").read_bytes(), ING, OCR)
    assert exc.value.code == "encrypted"


def test_blank_pdf_flagged_as_needing_ocr(fixtures_dir):
    result = extract_pdf((fixtures_dir / "blank.pdf").read_bytes(), ING, OCR)
    assert result.ocr_required and result.pages[0].status == "low_text"


def test_scanned_pdf_detected(fixtures_dir):
    result = extract_pdf((fixtures_dir / "scanned_page.pdf").read_bytes(), ING, OCR)
    assert result.is_scanned and result.ocr_required and not result.ocr_applied


def test_ocr_enabled_without_tesseract_reports_warning(fixtures_dir, monkeypatch):
    def broken_ocr(page, settings):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(pdf_processor, "_ocr_page", broken_ocr)
    result = extract_pdf((fixtures_dir / "scanned_page.pdf").read_bytes(), ING, OCRSettings(enabled=True))
    assert result.ocr_required
    assert any("OCR failed" in w for w in result.warnings)


def test_ocr_text_is_marked_with_provenance(fixtures_dir, monkeypatch):
    monkeypatch.setattr(
        pdf_processor, "_ocr_page",
        lambda page, s: [("Recognised text from a scanned page about soil moisture and crops.", 0.0, False)],
    )
    result = extract_pdf((fixtures_dir / "scanned_page.pdf").read_bytes(), ING, OCRSettings(enabled=True))
    assert result.ocr_applied and not result.ocr_required
    assert result.pages[0].source == "ocr"


def test_partial_page_failure_falls_back(fixtures_dir, monkeypatch):
    original = pdf_processor._blocks_from_page

    def flaky(page):
        if page.number == 1:
            raise RuntimeError("damaged page")
        return original(page)

    monkeypatch.setattr(pdf_processor, "_blocks_from_page", flaky)
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), ING, OCR)
    assert result.pages[1].source == "pypdf"
    assert "fixed-effects" in result.pages[1].text


def test_clean_block_repairs_hyphenation_and_wrapping():
    assert clean_block("agri-\ncultural yields were\nlower") == "agricultural yields were lower"
    assert clean_block("Self-\nReported") == "Self- Reported"  # capitalised: keep hyphen


def test_clean_block_keeps_table_rows():
    table = "Model   RMSE   R2\nGBM   0.42   0.81\nRF   0.47   0.77"
    assert clean_block(table).count("\n") == 2


def test_repeated_header_detection():
    pages = [["Journal of Tests", f"content {i}"] for i in range(4)]
    assert "journal of tests" in find_repeated_lines(pages)


def test_file_hash_is_stable(fixtures_dir):
    data = (fixtures_dir / "rainfall_maize_study.pdf").read_bytes()
    assert pdf_processor.file_hash(data) == pdf_processor.file_hash(bytes(data))
