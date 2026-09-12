"""Executable JSON-to-Streamlit-boundary pipeline test.

Foundry Local is replaced only at its two inference boundaries so the test is
deterministic. JSON loading, chunking, SQLite serialization, query embedding,
cosine ranking, RAG context construction and UI response parsing are real.
"""

from __future__ import annotations

import json

from app import split_rag_response
from src.database import get_chunks, get_database_stats
from src.ingestion import ingest_directory
from src.rag import DISCLAIMER, answer_query
from src.retrieval import get_top_chunks


def _deterministic_embedding(text: str) -> list[float]:
    lowered = text.casefold()
    if "yan etki" in lowered or "baş ağrısı" in lowered:
        return [1.0, 0.0, 0.0]
    if "uyarı" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def test_json_to_streamlit_response_pipeline(tmp_path):
    documents_dir = tmp_path / "medicines"
    documents_dir.mkdir()
    database_path = tmp_path / "pipeline.db"
    payload = {
        "medicine_name": "DENEME 10 mg tablet",
        "active_ingredient": "Deneme maddesi",
        "common_side_effects": "Baş ağrısı görülebilir.",
        "warnings": "Kayıtlı uyarı metni.",
        "source": {"name": "TİTCK DENEME KT", "reference": "KT-001"},
    }
    (documents_dir / "deneme.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    stats = ingest_directory(
        documents_dir,
        database_path=database_path,
        embedding_function=_deterministic_embedding,
    )
    stored_chunks = get_chunks(database_path=database_path)
    assert stats == {
        "files": 1,
        "medicines": 1,
        "inserted": 1,
        "updated": 0,
        "duplicates": 0,
        "errors": 0,
        "chunks": 3,
        "skipped_examples": 0,
    }
    assert get_database_stats(database_path) == {
        "medicine_count": 1,
        "chunk_count": 3,
    }
    assert all(chunk["embedding"] for chunk in stored_chunks)

    trace: dict[str, object] = {}

    def retrieve(query: str, top_k: int = 5, debug_trace=None):
        return get_top_chunks(
            query,
            top_k=top_k,
            database_path=database_path,
            embedding_function=_deterministic_embedding,
            debug_trace=debug_trace,
        )

    def grounded_chat(messages):
        context = messages[1]["content"]
        assert "[KAYNAK 1]" in context
        assert "Kategori: side_effects" in context
        assert "Baş ağrısı görülebilir." in context
        return "Kayıtlı kaynakta baş ağrısı yan etki olarak belirtilmiştir."

    raw_response = answer_query(
        "DENEME ilacının yan etkileri nelerdir?",
        retrieval_function=retrieve,
        chat_function=grounded_chat,
        debug_trace=trace,
    )
    answer, sources, disclaimer = split_rag_response(raw_response)

    assert "baş ağrısı" in answer.casefold()
    assert sources == "TİTCK DENEME KT"
    assert disclaimer == DISCLAIMER
    assert trace["query_embedding_created"] is True
    assert trace["model_called"] is True
    assert "[KAYNAK 1]" in str(trace["retrieved_context"])
