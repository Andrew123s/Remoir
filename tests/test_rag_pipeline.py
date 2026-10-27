import re

import httpx
import pytest

from app.core.config import LLMSettings
from app.services.llm_service import (
    INSUFFICIENT_EVIDENCE,
    ExtractiveProvider,
    GenerationRequest,
    LLMError,
    OllamaProvider,
    OpenAICompatibleProvider,
    SourceForPrompt,
    is_abstention,
)
from tests.conftest import FakeLLM


def use_llm(container, llm):
    container.pipeline.provider = llm
    return llm


def cite_first_source(request):
    first = request.sources[0]
    sentence = re.split(r"(?<=\.)\s", first.text)[0]
    return f"{sentence} [{first.source_id}]"


def test_context_construction_and_source_attribution(indexed):
    c, ids = indexed
    llm = use_llm(c, FakeLLM(cite_first_source))
    result = c.pipeline.answer("What limitations did the maize rainfall study identify?")
    req = llm.requests[0]
    assert [s.source_id for s in req.sources] == [f"S{i + 1}" for i in range(len(req.sources))]
    prompt = req.render_user_prompt()
    assert '<source id="S1"' in prompt and "untrusted" in prompt.lower()
    assert result.status == "answered"
    assert result.sources[0]["source_id"] == "S1"
    assert result.sources[0]["filename"] in ids
    assert result.sources[0]["pages"].startswith("p")


def test_context_respects_token_budget(indexed):
    c, _ = indexed
    c.settings.retrieval.max_context_tokens = 200
    llm = use_llm(c, FakeLLM(cite_first_source))
    c.pipeline.answer("rainfall variability maize yields", top_k=10)
    assert sum(len(s.text.split()) * 1.3 for s in llm.requests[0].sources[1:]) <= 200


def test_invalid_citation_from_llm_is_removed(indexed):
    c, _ = indexed
    use_llm(c, FakeLLM(lambda r: cite_first_source(r) + " Additionally the paper proves everything [S42]."))
    result = c.pipeline.answer("What methodology did the maize study use?")
    assert "[S42]" not in result.answer
    assert result.citation_report.invalid_citations == ["S42"]
    assert all(s["source_id"] != "S42" for s in result.sources)


def test_answer_without_valid_citations_is_marked_ungrounded(indexed):
    c, _ = indexed
    use_llm(c, FakeLLM("Maize yields are mostly determined by soil pH and farmer education levels in every region."))
    result = c.pipeline.answer("What determines maize yields according to the rainfall study?")
    assert result.sources == []
    assert result.citation_report.grounding_status == "ungrounded"
    assert "not grounded" in result.answer


def test_insufficient_evidence_skips_generation(indexed):
    c, _ = indexed
    llm = use_llm(c, FakeLLM("should not be called"))
    result = c.pipeline.answer("What is the boiling point of liquid nitrogen in quantum chromodynamics?")
    assert result.status == "insufficient_evidence"
    assert INSUFFICIENT_EVIDENCE in result.answer
    assert llm.requests == []
    assert result.sources == []


def test_empty_index_reports_insufficient_evidence(container):
    result = container.pipeline.answer("What are the main findings?")
    assert result.status == "insufficient_evidence"


def test_llm_abstention_is_reported(indexed):
    c, _ = indexed
    use_llm(c, FakeLLM(INSUFFICIENT_EVIDENCE + " The passages do not report costs."))
    result = c.pipeline.answer("What were the irrigation costs in the maize rainfall study?")
    assert result.status == "insufficient_evidence"


def test_multi_document_comparison_retrieves_each_document(indexed):
    c, ids = indexed
    llm = use_llm(c, FakeLLM(cite_first_source))
    docs = [ids["rainfall_maize_study.pdf"], ids["ml_crop_prediction.pdf"]]
    result = c.pipeline.answer("Compare the limitations reported by these papers.", document_ids=docs)
    assert result.comparison
    assert {s.document_id for s in llm.requests[0].sources} == set(docs)
    assert llm.requests[0].comparison


def test_comparison_without_selection_uses_all_documents(indexed):
    c, ids = indexed
    use_llm(c, FakeLLM(cite_first_source))
    result = c.pipeline.answer("Do the documents disagree about whether rainfall variability affects maize yields?")
    assert set(result.scope_document_ids) == set(ids.values())
    assert {e["document_id"] for e in result.evidence} >= {ids["rainfall_maize_study.pdf"], ids["ml_crop_prediction.pdf"]}


def test_document_mentioned_by_filename_scopes_retrieval(indexed):
    c, ids = indexed
    use_llm(c, FakeLLM(cite_first_source))
    result = c.pipeline.answer("Using only ml_crop_prediction, what are the limitations?")
    assert result.scope_document_ids == [ids["ml_crop_prediction.pdf"]]
    assert all(e["document_id"] == ids["ml_crop_prediction.pdf"] for e in result.evidence)


def test_follow_up_question_reuses_context_and_retrieves_again(indexed):
    c, ids = indexed
    llm = use_llm(c, FakeLLM(cite_first_source))
    doc = ids["rainfall_maize_study.pdf"]
    first = c.pipeline.answer("Summarize the methodology of the maize rainfall study.")
    history = [
        {"role": "user", "content": "Summarize the methodology of the maize rainfall study."},
        {"role": "assistant", "content": first.answer, "document_ids": [s["document_id"] for s in first.sources]},
    ]
    assert history[1]["document_ids"] == [doc]
    second = c.pipeline.answer("What were its main limitations?", history=history)
    assert second.scope_document_ids == [doc]
    assert "methodology" in second.retrieval_query.lower()
    assert any("limitation" in e["text"].lower() for e in second.evidence)
    # previous answers are context, not sources
    assert llm.requests[1].history[0]["role"] == "user"
    assert all("[S" not in t["content"] for t in llm.requests[1].history if t["role"] == "assistant")


def test_non_follow_up_question_is_not_rewritten(indexed):
    c, _ = indexed
    use_llm(c, FakeLLM(cite_first_source))
    history = [{"role": "user", "content": "Summarize the methodology of the maize study."},
               {"role": "assistant", "content": "x", "document_ids": []}]
    result = c.pipeline.answer("Which machine learning models were compared for crop yield prediction?", history=history)
    assert result.retrieval_query.startswith("Which machine learning")


def test_generation_failure_is_reported(indexed):
    c, _ = indexed

    class Broken(FakeLLM):
        def generate(self, request):
            raise LLMError("The local language model (Ollama) did not return an answer.")

    use_llm(c, Broken())
    result = c.pipeline.answer("What methodology did the maize study use?")
    assert result.status == "generation_failed"


def test_extractive_provider_quotes_and_cites_sources(indexed):
    c, _ = indexed
    c.pipeline.provider = ExtractiveProvider(c.embedder, min_score=c.settings.llm.extractive_min_score)
    result = c.pipeline.answer("What limitations did the maize rainfall study identify?")
    assert result.status == "answered"
    assert result.sources
    for claim in result.citation_report.claims:
        quoted = claim.sentence.strip('"')
        assert any(quoted in s["text"] for s in result.sources)


def test_extractive_comparison_groups_by_document(indexed):
    c, ids = indexed
    provider = ExtractiveProvider(c.embedder, min_score=0.0)
    sources = [
        SourceForPrompt("S1", "The study has several limitations including sparse weather stations in the north.", "a", "a.pdf", None, "p. 3", 1),
        SourceForPrompt("S2", "The models were trained on a single country and may not transfer to other regions.", "b", "b.pdf", None, "p. 2", 1),
    ]
    text = provider.generate(GenerationRequest("Compare the limitations", "", sources, comparison=True))
    assert "**a.pdf**" in text and "**b.pdf**" in text and "[S1]" in text and "[S2]" in text


def test_ollama_provider_with_mocked_http():
    def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        body = request.read().decode()
        assert '"stream": false' in body.replace(" ", "").replace(":false", ": false") or "stream" in body
        return httpx.Response(200, json={"message": {"content": "Rainfall matters [S1]."}})

    provider = OllamaProvider(LLMSettings(provider="ollama"), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.status()["available"]
    req = GenerationRequest("q?", "system", [SourceForPrompt("S1", "text", "d", "f.pdf", None, "p. 1", 1)])
    assert provider.generate(req) == "Rainfall matters [S1]."


def test_ollama_unreachable_reports_clearly():
    def handler(request):
        raise httpx.ConnectError("refused")

    provider = OllamaProvider(LLMSettings(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert not provider.status()["available"]
    with pytest.raises(LLMError):
        provider.generate(GenerationRequest("q", "s", []))


def test_openai_provider_reads_key_and_sends_bearer(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer [S1]."}}]})

    monkeypatch.setenv("REMOIR_LLM_API_KEY", "sk-test-123456789")
    provider = OpenAICompatibleProvider(LLMSettings(provider="openai"), client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.generate(GenerationRequest("q", "s", [])) == "Answer [S1]."
    assert seen["auth"] == "Bearer sk-test-123456789"
    assert "sk-test" not in str(provider.status())


def test_openai_provider_without_key(monkeypatch):
    monkeypatch.delenv("REMOIR_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(LLMSettings(provider="openai"))
    assert not provider.status()["available"]
    with pytest.raises(LLMError):
        provider.generate(GenerationRequest("q", "s", []))


def test_abstention_detection():
    assert is_abstention(INSUFFICIENT_EVIDENCE)
    assert not is_abstention("Rainfall reduced yields [S1].")


def _ollama_ready(model):
    try:
        names = [m["name"] for m in httpx.get("http://localhost:11434/api/tags", timeout=2).json().get("models", [])]
        return model in names
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_ready("llama3.2:3b"), reason="Ollama with llama3.2:3b is not running locally")
def test_ollama_integration_answers_with_valid_citations(indexed):
    c, _ = indexed
    c.pipeline.provider = OllamaProvider(LLMSettings(provider="ollama", model="llama3.2:3b"))
    result = c.pipeline.answer("What limitations did the maize rainfall study identify?")
    assert result.status in {"answered", "insufficient_evidence"}
    if result.citation_report:
        assert all(s["source_id"].startswith("S") for s in result.sources)
