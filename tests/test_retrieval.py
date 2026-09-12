from __future__ import annotations

import pytest

from src import retrieval


def test_cosine_similarity():
    assert retrieval.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert retrieval.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_get_top_chunks_orders_and_returns_required_fields(monkeypatch):
    chunks = [
        {
            "medicine_name": "ALFA 10 mg tablet",
            "chunk_type": "warnings",
            "chunk_text": "Alfa uyarısı",
            "embedding": [0.8, 0.6],
            "source_name": "Kaynak A",
        },
        {
            "medicine_name": "BETA 20 mg tablet",
            "chunk_type": "warnings",
            "chunk_text": "Beta uyarısı",
            "embedding": [0.85, 0.5268],
            "source_name": "Kaynak B",
        },
    ]
    monkeypatch.setattr(retrieval, "get_chunks", lambda **_: chunks)
    monkeypatch.setattr(
        retrieval,
        "find_candidate_medicines",
        lambda *_, **__: [
            {
                "medicine_id": 1,
                "medicine_name": "ALFA 10 mg tablet",
                "alias": "ALFA",
                "name_score": 1.0,
            }
        ],
    )

    results = retrieval.get_top_chunks(
        "Alfa uyarıları nelerdir?",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.0,
    )

    # Beta has a slightly higher raw score, but the explicit Alfa name receives
    # a limited boost. Semantic results remain in the returned list.
    assert [result["medicine_name"] for result in results] == [
        "ALFA 10 mg tablet",
        "BETA 20 mg tablet",
    ]
    assert {
        "medicine_name",
        "chunk_type",
        "chunk_text",
        "similarity_score",
        "source_name",
    }.issubset(results[0])


def test_get_top_chunks_returns_empty_below_threshold(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda **_: [
            {
                "medicine_name": "ALFA",
                "chunk_type": "usage",
                "chunk_text": "Metin",
                "embedding": [0.0, 1.0],
                "source_name": "Kaynak",
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "find_candidate_medicines",
        lambda *_, **__: [
            {
                "medicine_id": 1,
                "medicine_name": "ALFA",
                "alias": "ALFA",
                "name_score": 1.0,
            }
        ],
    )

    assert (
        retrieval.get_top_chunks(
            "soru",
            embedding_function=lambda _: [1.0, 0.0],
            minimum_score=0.35,
        )
        == []
    )


def test_empty_database_does_not_generate_query_embedding(monkeypatch):
    monkeypatch.setattr(retrieval, "get_chunks", lambda **_: [])

    def fail_embedding(_text):
        raise AssertionError("embedding model should not load for an empty database")

    trace = {}
    assert retrieval.get_top_chunks(
        "herhangi bir soru",
        embedding_function=fail_embedding,
        debug_trace=trace,
    ) == []
    assert trace["query_embedding_created"] is False
    assert trace["database_chunk_count"] == 0


def test_debug_trace_contains_embedding_and_ranked_chunks(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda **_: [
            {
                "medicine_name": "ALFA",
                "chunk_type": "warnings",
                "chunk_text": "Uyarı metni",
                "embedding": [1.0, 0.0],
                "source_name": "Kaynak",
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "find_candidate_medicines",
        lambda *_, **__: [
            {
                "medicine_id": 1,
                "medicine_name": "ALFA",
                "alias": "ALFA",
                "name_score": 1.0,
            }
        ],
    )
    trace = {}

    results = retrieval.get_top_chunks(
        "Alfa uyarısı",
        embedding_function=lambda _: [1.0, 0.0],
        debug_trace=trace,
    )

    assert results
    assert trace["query_embedding_created"] is True
    assert trace["query_embedding_dimension"] == 2
    assert trace["query_embedding_preview"] == [1.0, 0.0]
    assert trace["top_chunks"][0]["medicine_name"] == "ALFA"
    assert trace["top_chunks"][0]["similarity_score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "question",
    [
        "Muscoflex yan etkileri neler?",
        "Muscoflex yan etkisi nedir?",
        "Muscoflex yan etkilerineler?",
        "Muscoflex istenmeyen etkiler",
        "Muscoflex yan etkilri neler?",
    ],
)
def test_side_effect_intent_tolerates_turkish_variants_and_typos(question):
    assert retrieval.detect_intent(question) == "SIDE_EFFECTS"


def test_retrieval_does_not_search_global_chunks_without_medicine(monkeypatch):
    monkeypatch.setattr(retrieval, "find_candidate_medicines", lambda *_, **__: [])

    def fail_get_chunks(**_):
        raise AssertionError("Global chunk scan must not run")

    monkeypatch.setattr(retrieval, "get_chunks", fail_get_chunks)
    trace = {}

    assert retrieval.get_top_chunks("Bilinmeyen ilaç yan etkileri", debug_trace=trace) == []
    assert trace["retrieval_state"] == "medicine_not_found"


def test_irrelevant_sections_are_not_sent_as_semantic_fallback(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "find_candidate_medicines",
        lambda *_, **__: [
            {
                "medicine_id": 7,
                "medicine_name": "MUSCOFLEX 8 MG",
                "alias": "MUSCOFLEX",
                "matched_alias": "MUSCOFLEX",
                "match_type": "exact_alias",
                "name_score": 1.0,
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "get_medicine_data_status",
        lambda *_, **__: {
            "medicine_exists": True,
            "documents_count": 1,
            "kt_count": 1,
            "kub_count": 0,
            "chunks_count": 1,
            "side_effect_chunks": 0,
            "dosage_chunks": 0,
            "embeddings_count": 1,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda **_: [
            {
                "chunk_id": 1,
                "medicine_name": "MUSCOFLEX 8 MG",
                "chunk_type": "indications",
                "chunk_text": "Yalnız endikasyon",
                "embedding": [1.0, 0.0],
            }
        ],
    )
    trace = {}

    assert retrieval.get_top_chunks("muscoflex yan etkileri", debug_trace=trace) == []
    assert trace["retrieval_state"] == "section_missing"
