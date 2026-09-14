from __future__ import annotations

from src.database import (
    get_medicine_readiness,
    get_ready_medicines,
    insert_medicine,
    replace_document_chunks,
    upsert_document,
)


def test_readiness_requires_processed_document_current_embedding_and_fts(tmp_path):
    database_path = tmp_path / "ready.db"
    medicine_id = insert_medicine("READY 10 MG TABLET", database_path=database_path)
    document_id, _ = upsert_document(
        medicine_id,
        "KT",
        "https://www.titck.gov.tr/ready.pdf",
        local_path="D:/ready.pdf",
        content_hash="hash",
        raw_text="1. READY nedir ve ne için kullanılır? Metin",
        database_path=database_path,
    )
    replace_document_chunks(
        document_id,
        medicine_id,
        [
            {
                "chunk_text": "İlaç: READY\nBölüm: Ne için kullanılır\nMetin",
                "chunk_type": "indications",
                "embedding": [1.0, 0.0],
                "embedding_model": "qwen3-embedding-0.6b",
            }
        ],
        database_path=database_path,
    )

    readiness = get_medicine_readiness(medicine_id, database_path=database_path)
    assert readiness["is_ready"] is True
    assert readiness["semantic_families"] == ["indications"]
    assert [item["medicine_id"] for item in get_ready_medicines(database_path=database_path)] == [medicine_id]


def test_catalog_only_medicine_is_not_ready(tmp_path):
    database_path = tmp_path / "not-ready.db"
    medicine_id = insert_medicine("CATALOG ONLY", database_path=database_path)
    readiness = get_medicine_readiness(medicine_id, database_path=database_path)
    assert readiness["is_ready"] is False
    assert "source_document_missing" in readiness["reasons"]
