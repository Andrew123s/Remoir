import numpy as np
import pytest

from app.dependencies import Container
from app.services.embedding_service import HashingEmbedder
from app.services.retrieval_service import BM25Index, InvalidQuestionError, normalize_question
from app.services.vector_store import FaissVectorStore, IncompatibleIndexError


def test_vector_store_add_search_delete_persist(tmp_path):
    store = FaissVectorStore(tmp_path, "hashing:384", None)
    e = HashingEmbedder()
    texts = ["maize yields and rainfall", "pollinators in hedgerows", "gradient boosting yield prediction"]
    store.add([10, 20, 30], e.embed_documents(texts))
    hits = store.search(e.embed_query("rainfall maize"), 2)
    assert hits[0][0] == 10
    # filtering
    filtered = store.search(e.embed_query("rainfall maize"), 3, allowed_ids=[20, 30])
    assert {i for i, _ in filtered} <= {20, 30}
    assert store.search(e.embed_query("x"), 3, allowed_ids=[]) == []
    # duplicate add does not duplicate vectors
    store.add([10], e.embed_documents([texts[0]]))
    assert store.size == 3
    # delete
    assert store.remove([20]) == 1
    assert 20 not in store.ids()
    # reload from disk
    reloaded = FaissVectorStore(tmp_path, "hashing:384", None)
    assert reloaded.size == 2 and reloaded.ids() == {10, 30}


def test_incompatible_model_is_detected(tmp_path):
    store = FaissVectorStore(tmp_path, "model-a", None)
    store.add([1], HashingEmbedder().embed_documents(["text"]))
    other = FaissVectorStore(tmp_path, "model-b", None)
    assert not other.compatible
    with pytest.raises(IncompatibleIndexError):
        other.search(np.zeros(384, dtype="float32"), 1)
    with pytest.raises(IncompatibleIndexError):
        other.add([2], HashingEmbedder().embed_documents(["more"]))


def test_question_validation():
    assert normalize_question("  What   is  this? ") == "What is this?"
    for bad in ["", "  ", "?!", "x" * 2001]:
        with pytest.raises(InvalidQuestionError):
            normalize_question(bad)


def test_bm25_ranks_lexical_matches():
    index = BM25Index()
    index.build([
        {"id": 1, "document_id": "a", "text": "Dry spells during flowering reduce maize yields."},
        {"id": 2, "document_id": "b", "text": "Hedgerows support pollinator diversity."},
    ])
    assert index.search("pollinator hedgerows", 2)[0][0] == 2
    assert index.search("pollinator", 2, allowed={1}) == []


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_semantic_search_finds_methodology(indexed, strategy):
    c, ids = indexed
    result = c.retriever.retrieve("Which regression with district fixed effects and clustered standard errors was estimated?", strategy=strategy, top_k=3)
    assert result.passages
    top = result.passages[0]
    assert top.document_id == ids["rainfall_maize_study.pdf"]
    assert "fixed-effects" in top.text
    assert top.page_start <= 2 <= top.page_end


def test_document_filter_restricts_results(indexed):
    c, ids = indexed
    only = ids["ml_crop_prediction.pdf"]
    result = c.retriever.retrieve("limitations of the study", document_ids=[only], top_k=5)
    assert result.passages and all(p.document_id == only for p in result.passages)


def test_balanced_retrieval_covers_each_document(indexed):
    c, ids = indexed
    docs = [ids["rainfall_maize_study.pdf"], ids["ml_crop_prediction.pdf"]]
    result = c.retriever.retrieve("limitations", document_ids=docs, top_k=4, balance_documents=True)
    assert {p.document_id for p in result.passages} == set(docs)


def test_passages_carry_metadata(indexed):
    c, ids = indexed
    p = c.retriever.retrieve("pollinator diversity hedgerows", top_k=1).passages[0]
    assert p.filename == "ecosystem_services_review.pdf"
    assert p.chunk_id.startswith(ids["ecosystem_services_review.pdf"])
    assert p.page_start == 1 and p.section


def test_duplicate_upload_does_not_add_chunks(indexed, fixtures_dir):
    c, ids = indexed
    before = c.vectors.size
    outcome = c.ingestion.ingest_bytes("copy.pdf", (fixtures_dir / "rainfall_maize_study.pdf").read_bytes())
    assert outcome.status == "duplicate" and outcome.duplicate_of == ids["rainfall_maize_study.pdf"]
    assert c.vectors.size == before


def test_delete_removes_vectors_and_chunks(indexed):
    c, ids = indexed
    doc_id = ids["ecosystem_services_review.pdf"]
    before = c.vectors.size
    n_chunks = len(c.store.document_chunks(doc_id))
    assert c.ingestion.delete(doc_id)
    c.retriever.invalidate()
    assert c.vectors.size == before - n_chunks
    assert c.store.get_document(doc_id) is None
    result = c.retriever.retrieve("pollinator diversity hedgerows", top_k=5)
    assert all(p.document_id != doc_id for p in result.passages)
    assert c.index_consistent()


def test_index_persists_across_restart(indexed, settings):
    c, ids = indexed
    size = c.vectors.size
    c.store.close()
    restarted = Container(settings, embedder=HashingEmbedder())
    assert restarted.vectors.size == size
    assert restarted.index_consistent()
    hits = restarted.retriever.retrieve("dry spell index flowering", top_k=1).passages
    assert hits[0].document_id == ids["rainfall_maize_study.pdf"]
    restarted.store.close()


def test_rebuild_index_reembeds_everything(indexed):
    c, _ = indexed
    size = c.vectors.size
    result = c.ingestion.rebuild_index()
    assert result["vectors"] == size and c.index_consistent()
    reparsed = c.ingestion.rebuild_index(reparse=True)
    assert reparsed["documents_reindexed"] == 3 and c.index_consistent()


def test_near_duplicate_passages_suppressed(container):
    from app.services.retrieval_service import Passage

    def p(i, text):
        return Passage(i, f"d:{i}", "d", "f.pdf", None, None, None, 1, 1, None, text, [], "native")

    text = "Rainfall variability reduced maize yields by eleven percent in rain-fed districts of the region."
    kept = container.retriever._deduplicate([p(1, text), p(2, text + " Extra."), p(3, "Pollinators need hedgerows.")])
    assert [k.vector_id for k in kept] == [1, 3]
