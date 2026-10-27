import pytest

from app.core.config import ChunkingSettings, IngestionSettings, OCRSettings
from app.services.chunker import approx_token_count, chunk_pages, split_sentences
from app.services.pdf_processor import Block, PageContent, extract_pdf


def page(n, *paras, heading=None):
    blocks = ([Block(heading, True)] if heading else []) + [Block(p) for p in paras]
    return PageContent(number=n, blocks=blocks)


SENT = "The study measured rainfall variability across many farming districts in the region. "


def test_empty_input_returns_no_chunks():
    assert chunk_pages([], ChunkingSettings()) == []
    assert chunk_pages([PageContent(1, [])], ChunkingSettings()) == []


def test_chunks_respect_size_limit():
    settings = ChunkingSettings(chunk_size=60, chunk_overlap=10, min_chunk_tokens=5, max_chunk_tokens=120)
    chunks = chunk_pages([page(1, SENT * 30)], settings)
    assert len(chunks) > 3
    assert all(c.token_count <= settings.max_chunk_tokens for c in chunks)
    assert all(c.token_count <= settings.chunk_size + approx_token_count(SENT) for c in chunks)


def test_overlap_repeats_trailing_sentences():
    text = " ".join(f"Sentence number {i} describes a separate finding about crops." for i in range(40))
    settings = ChunkingSettings(chunk_size=60, chunk_overlap=20, min_chunk_tokens=5, max_chunk_tokens=120)
    chunks = chunk_pages([page(1, text)], settings)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(a.text)[-1]
        assert last_sentence in b.text


def test_zero_overlap_has_no_shared_sentences():
    text = " ".join(f"Sentence number {i} describes a separate finding about crops." for i in range(40))
    settings = ChunkingSettings(chunk_size=60, chunk_overlap=0, min_chunk_tokens=5, max_chunk_tokens=120)
    chunks = chunk_pages([page(1, text)], settings)
    sentences = [s for c in chunks for s in split_sentences(c.text)]
    assert len(sentences) == len(set(sentences))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([page(1, SENT)], ChunkingSettings(chunk_size=50, chunk_overlap=50))


def test_sentences_are_not_split_mid_sentence():
    text = " ".join(f"Finding {i} shows that yields declined in dry years." for i in range(30))
    chunks = chunk_pages([page(1, text)], ChunkingSettings(chunk_size=60, chunk_overlap=0, min_chunk_tokens=5))
    for c in chunks:
        assert c.text.startswith("Finding") and c.text.endswith(".")


def test_abbreviations_do_not_split_sentences():
    assert len(split_sentences("Results by Smith et al. were strong. Next sentence here.")) == 2
    assert len(split_sentences("Crops (e.g. maize) failed. Then rain came.")) == 2


def test_page_provenance_and_cross_page_chunks():
    pages = [page(1, "Alpha finding on page one about soils."), page(2, "Beta finding on page two about rain.")]
    chunks = chunk_pages(pages, ChunkingSettings(chunk_size=200, chunk_overlap=0, min_chunk_tokens=1))
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.page_start, c.page_end) == (1, 2)
    spans = {p: c.text[s:e] for p, s, e in c.page_spans}
    assert spans[1] == "Alpha finding on page one about soils."
    assert spans[2] == "Beta finding on page two about rain."


def test_heading_starts_new_chunk_and_sets_section():
    pages = [
        page(1, SENT * 4, heading="1. Introduction"),
        page(2, "The limitations include sparse weather stations and self-reported data.", heading="2. Limitations"),
    ]
    chunks = chunk_pages(pages, ChunkingSettings(chunk_size=200, chunk_overlap=20, min_chunk_tokens=10))
    assert chunks[-1].section == "2. Limitations"
    assert chunks[-1].text.startswith("2. Limitations")
    assert chunks[0].section == "1. Introduction"


def test_overlong_sentence_is_split():
    long_sentence = " ".join(["word"] * 900) + "."
    settings = ChunkingSettings(chunk_size=100, chunk_overlap=0, min_chunk_tokens=5, max_chunk_tokens=200)
    chunks = chunk_pages([page(1, long_sentence)], settings)
    assert len(chunks) > 1 and all(c.token_count <= 200 for c in chunks)


def test_small_tail_merged_into_previous_chunk():
    text = SENT * 6 + "Short tail."
    settings = ChunkingSettings(chunk_size=60, chunk_overlap=0, min_chunk_tokens=10, max_chunk_tokens=200)
    chunks = chunk_pages([page(1, text)], settings)
    assert chunks[-1].text.endswith("Short tail.")
    assert chunks[-1].token_count > 10


def test_failed_pages_skipped():
    bad = page(2, "should not appear")
    bad.status = "failed"
    chunks = chunk_pages([page(1, SENT), bad], ChunkingSettings(chunk_size=100, chunk_overlap=0, min_chunk_tokens=1))
    assert all("should not appear" not in c.text for c in chunks)


def test_real_fixture_chunks_cover_all_pages(fixtures_dir):
    result = extract_pdf((fixtures_dir / "rainfall_maize_study.pdf").read_bytes(), IngestionSettings(), OCRSettings())
    chunks = chunk_pages(result.pages, ChunkingSettings(chunk_size=120, chunk_overlap=20, min_chunk_tokens=10))
    covered = {p for c in chunks for p in c.pages}
    assert covered == {1, 2, 3}
    assert any(c.page_start != c.page_end for c in chunks)  # results section spans pages 2-3
