import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM


@pytest.fixture
def client(container):
    with TestClient(create_app()) as c:
        yield c


def upload(client, fixtures_dir, name, as_name=None):
    with open(fixtures_dir / name, "rb") as fh:
        return client.post("/documents/upload", files={"file": (as_name or name, fh, "application/pdf")})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm"]["provider"] == "extractive"
    assert "data" not in str(body).lower().replace("index_consistent", "")  # no filesystem paths


def test_upload_list_get_delete(client, fixtures_dir, container):
    r = upload(client, fixtures_dir, "rainfall_maize_study.pdf")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "indexed" and body["document"]["chunk_count"] > 0
    doc_id = body["document_id"]
    assert "stored_name" not in body["document"]  # no internal paths

    listing = client.get("/documents").json()
    assert [d["id"] for d in listing] == [doc_id]

    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["page_count"] == 3 and len(detail["pages"]) == 3
    assert detail["title"].startswith("Rainfall Variability")

    assert client.get(f"/documents/{doc_id}/pages/2/image").headers["content-type"] == "image/png"
    assert client.get(f"/documents/{doc_id}/pages/99/image").status_code == 404
    assert client.get(f"/documents/{doc_id}/chunks").json()[0]["chunk_id"].startswith(doc_id)

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json() == []
    assert container.vectors.size == 0


def test_duplicate_upload_detected(client, fixtures_dir):
    first = upload(client, fixtures_dir, "ml_crop_prediction.pdf").json()
    second = upload(client, fixtures_dir, "ml_crop_prediction.pdf", "renamed.pdf").json()
    assert second["status"] == "duplicate" and second["duplicate_of"] == first["document_id"]
    assert len(client.get("/documents").json()) == 1


def test_failed_documents_are_recorded_not_crashing(client, fixtures_dir):
    corrupted = upload(client, fixtures_dir, "corrupted.pdf").json()
    encrypted = upload(client, fixtures_dir, "encrypted.pdf").json()
    scanned = upload(client, fixtures_dir, "scanned_page.pdf").json()
    assert corrupted["status"] == "failed"
    assert encrypted["status"] == "failed" and "password" in encrypted["message"].lower()
    assert scanned["status"] == "ocr_required"
    statuses = {d["filename"]: d["status"] for d in client.get("/documents").json()}
    assert statuses == {"corrupted.pdf": "failed", "encrypted.pdf": "failed", "scanned_page.pdf": "ocr_required"}
    assert client.get("/health").json()["status"] == "ok"


def test_search_endpoint(client, fixtures_dir):
    doc_id = upload(client, fixtures_dir, "rainfall_maize_study.pdf").json()["document_id"]
    upload(client, fixtures_dir, "ecosystem_services_review.pdf")
    r = client.post("/search", json={"query": "dry spell index during flowering", "retrieval": {"top_k": 3}})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["document_id"] == doc_id and results[0]["pages"].startswith("p")
    filtered = client.post("/search", json={"query": "pollinators", "document_ids": [doc_id]}).json()["results"]
    assert all(x["document_id"] == doc_id for x in filtered)


def test_chat_endpoint_and_follow_up(client, fixtures_dir):
    upload(client, fixtures_dir, "rainfall_maize_study.pdf")
    upload(client, fixtures_dir, "ml_crop_prediction.pdf")
    r = client.post("/chat/ask", json={"question": "What limitations did the maize rainfall study identify?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    assert body["citations"] and body["citations"][0]["source_id"] == "S1" or body["citations"]
    assert body["grounding"]["invalid_citations"] == []
    conv = body["conversation_id"]

    follow = client.post("/chat/ask", json={"question": "What methodology did it use?", "conversation_id": conv}).json()
    assert follow["conversation_id"] == conv
    assert "limitations" in follow["retrieval_query"].lower()
    turns = client.get(f"/chat/conversations/{conv}").json()["turns"]
    assert len(turns) == 4

    assert client.delete(f"/chat/conversations/{conv}").status_code == 200
    assert client.get(f"/chat/conversations/{conv}").status_code == 404


def test_chat_insufficient_evidence(client, fixtures_dir):
    upload(client, fixtures_dir, "ecosystem_services_review.pdf")
    body = client.post("/chat/ask", json={"question": "What is the GDP of Iceland in 1850?"}).json()
    assert body["status"] == "insufficient_evidence" and body["citations"] == []


def test_chat_with_mocked_llm_validates_citations(client, fixtures_dir, container):
    upload(client, fixtures_dir, "rainfall_maize_study.pdf")
    container.pipeline.provider = FakeLLM("Rainfall variability reduced maize yields [S1]. Invented claim [S99].")
    body = client.post("/chat/ask", json={"question": "How does rainfall variability affect maize yields?"}).json()
    assert "[S99]" not in body["answer"]
    assert body["grounding"]["invalid_citations"] == ["S99"]
    assert [c["source_id"] for c in body["citations"]] == ["S1"]


def test_collections(client, fixtures_dir):
    a = upload(client, fixtures_dir, "rainfall_maize_study.pdf").json()["document_id"]
    b = upload(client, fixtures_dir, "ecosystem_services_review.pdf").json()["document_id"]
    coll = client.post("/collections", json={"name": "Agri", "document_ids": [a]}).json()
    assert coll["document_ids"] == [a]
    assert client.post("/collections", json={"name": "Agri"}).status_code == 409
    res = client.post("/search", json={"query": "pollinators hedgerows", "collection_id": coll["id"]}).json()
    assert all(r["document_id"] == a for r in res["results"])
    updated = client.put(f"/collections/{coll['id']}", json={"document_ids": [a, b]}).json()
    assert set(updated["document_ids"]) == {a, b}
    assert client.delete(f"/collections/{coll['id']}").status_code == 200
    assert client.get("/collections").json() == []


def test_invalid_requests(client):
    assert client.post("/chat/ask", json={}).status_code == 422
    assert client.post("/chat/ask", json={"question": ""}).status_code == 422
    assert client.post("/chat/ask", json={"question": "?!?"}).status_code == 422
    assert client.post("/chat/ask", json={"question": "hi there", "unexpected": 1}).status_code == 422
    assert client.post("/chat/ask", json={"question": "valid question", "conversation_id": "bad"}).status_code == 422
    assert client.post("/chat/ask", json={"question": "valid question", "conversation_id": "a" * 32}).status_code == 404
    assert client.post("/chat/ask", json={"question": "valid question", "document_ids": ["0" * 16]}).status_code == 404
    assert client.post("/search", json={"query": "x", "retrieval": {"top_k": 999}}).status_code == 422
    assert client.get("/documents/not-a-valid-id").status_code == 422
    assert client.delete("/documents/" + "0" * 16).status_code == 404
    assert client.post("/documents/upload").status_code == 422


def test_settings_endpoint(client):
    current = client.get("/settings").json()["settings"]
    current["top_k"] = 4
    current["strategy"] = "dense"
    r = client.put("/settings", json=current)
    assert r.status_code == 200 and r.json()["settings"]["top_k"] == 4
    bad = dict(current, chunk_size=100, chunk_overlap=120)
    assert client.put("/settings", json=bad).status_code == 422
    assert client.put("/settings", json=dict(current, top_k=500)).status_code == 422
    assert client.put("/settings", json=dict(current, llm_provider="shell")).status_code == 422


def test_rebuild_endpoint(client, fixtures_dir):
    upload(client, fixtures_dir, "rainfall_maize_study.pdf")
    r = client.post("/index/rebuild", json={"reparse": False})
    assert r.status_code == 200 and r.json()["documents_reindexed"] == 1
    assert client.get("/health").json()["index_consistent"]


def test_internal_errors_are_not_exposed(client, container, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal path C:/secret/file")

    monkeypatch.setattr(container.retriever, "retrieve", boom)
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        r = c.post("/search", json={"query": "rainfall"})
    assert r.status_code == 500 and "secret" not in r.text
