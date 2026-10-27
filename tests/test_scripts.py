from scripts.manage_index import backup, restore


def test_backup_and_restore_round_trip(indexed, tmp_path):
    c, ids = indexed
    data = c.data_path
    snapshot = backup(data, tmp_path / "backups")
    n_vectors = c.vectors.size
    c.ingestion.delete(ids["rainfall_maize_study.pdf"])
    assert c.vectors.size < n_vectors
    c.store.close()
    restore(data, snapshot)

    from app.dependencies import Container
    from app.services.embedding_service import HashingEmbedder

    restored = Container(c.settings, embedder=HashingEmbedder())
    assert restored.vectors.size == n_vectors
    assert restored.index_consistent()
    assert restored.store.get_document(ids["rainfall_maize_study.pdf"])["status"] == "indexed"
    assert restored.ingestion.stored_pdf_path(ids["rainfall_maize_study.pdf"]) is not None
    restored.store.close()
