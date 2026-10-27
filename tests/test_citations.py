from app.services.citation_service import (
    CitationVerifier,
    extract_citations,
    format_pages,
    strip_invalid_citations,
)
from app.services.embedding_service import HashingEmbedder

SOURCES = {
    "S1": "Rainfall variability was associated with an 11.4 percent decline in average maize yield.",
    "S2": "Wild pollinators depend on hedgerows and field margins.",
}


def verifier():
    return CitationVerifier(HashingEmbedder(), min_support=0.35)


def test_extract_citation_formats():
    assert extract_citations("A [S1] and B [S2, S3] and C [4]") == ["S1", "S2", "S3", "S4"]


def test_invalid_citations_removed_and_reported():
    cleaned, invalid = strip_invalid_citations("Claim [S1][S9]. Other [S2, S7].", {"S1", "S2"})
    assert cleaned == "Claim [S1]. Other [S2]."
    assert invalid == ["S9", "S7"]


def test_supported_claim_passes():
    report = verifier().verify(
        "Rainfall variability was associated with an 11.4 percent decline in maize yield [S1].", SOURCES
    )
    assert report.grounding_status == "grounded"
    assert report.used_source_ids == ["S1"]
    assert report.claims[0].supported and report.claims[0].support_score > 0.35


def test_claim_citing_wrong_source_is_flagged():
    report = verifier().verify("Rainfall variability caused an 11.4 percent decline in maize yield [S2].", SOURCES)
    assert report.grounding_status == "partially_grounded"
    assert report.unsupported_claims[0].issue == "weak_support"


def test_nonexistent_source_is_not_presented_as_sourced():
    report = verifier().verify("Pesticides have no environmental impact [S7].", SOURCES)
    assert report.invalid_citations == ["S7"]
    assert report.used_source_ids == []
    assert report.grounding_status == "ungrounded"


def test_uncited_factual_sentence_is_flagged():
    report = verifier().verify(
        "Maize yields doubled across all districts after the introduction of new irrigation subsidies in 2019.", SOURCES
    )
    assert report.claims[0].issue == "uncited"
    assert report.grounding_status == "ungrounded"


def test_meta_lines_are_not_treated_as_claims():
    text = "The following statements are quoted from the retrieved passages (extractive mode):\n- \"Wild pollinators depend on hedgerows and field margins.\" [S2]"
    report = verifier().verify(text, SOURCES)
    assert len(report.claims) == 1 and report.grounding_status == "grounded"


def test_page_formatting():
    assert format_pages(3, 3) == "p. 3"
    assert format_pages(7, 8) == "pp. 7–8"
