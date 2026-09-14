from __future__ import annotations

import pytest

from src import retrieval


@pytest.fixture(autouse=True)
def isolate_database_helpers(monkeypatch):
    """Keep pure ranking tests independent from the developer's D: database."""

    monkeypatch.setattr(retrieval, "search_medicine_chunks_fts", lambda *_, **__: [])
    monkeypatch.setattr(
        retrieval,
        "get_medicine_data_status",
        lambda *_, **__: {
            "medicine_exists": True,
            "documents_count": 1,
            "kt_count": 1,
            "kub_count": 0,
            "chunks_count": 1,
            "side_effect_chunks": 1,
            "dosage_chunks": 1,
            "embeddings_count": 1,
        },
    )


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


@pytest.mark.parametrize(
    "question",
    [
        "A-FERİN nasıl kullanılır?",
        "A-FERİN NASIL KULLANILIR?",
        "Arveles aç karnına mı içilir?",
        "İlaç yemekten sonra nasıl alınır?",
    ],
)
def test_usage_intent_covers_common_turkish_questions(question):
    assert retrieval.detect_intent(question) == "USAGE"


def test_muscle_relaxant_question_is_an_indication_intent():
    assert retrieval.detect_intent("Muscoflex kas gevşetir mi?") == "INDICATION"


def test_ambiguous_brand_returns_only_processed_product_questions(monkeypatch):
    candidates = [
        {"medicine_id": 1, "medicine_name": "A-FERİN ŞURUP", "alias": "A-FERİN"},
        {"medicine_id": 2, "medicine_name": "A-FERİN KAPSÜL", "alias": "A-FERİN"},
        {"medicine_id": 3, "medicine_name": "A-FERİN FORTE", "alias": "A-FERİN"},
    ]
    monkeypatch.setattr(retrieval, "find_candidate_medicines", lambda *_, **__: candidates)
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda **_: [
            {
                "medicine_id": 1,
                "medicine_name": "A-FERİN ŞURUP",
                "chunk_type": "usage",
                "chunk_text": "Şurup kullanım bilgisi",
                "embedding": [1.0, 0.0],
            },
            {
                "medicine_id": 2,
                "medicine_name": "A-FERİN KAPSÜL",
                "chunk_type": "usage",
                "chunk_text": "Kapsül kullanım bilgisi",
                "embedding": [1.0, 0.0],
            },
        ],
    )
    trace = {}

    assert retrieval.get_top_chunks("AFERİN nasıl kullanılır?", debug_trace=trace) == []
    assert trace["retrieval_state"] == "medicine_ambiguous"
    assert trace["similar_medicines"] == [
        "A-FERİN ŞURUP nasıl kullanılır?",
        "A-FERİN KAPSÜL nasıl kullanılır?",
    ]


def test_only_processed_variant_is_selected_from_brand_family(monkeypatch):
    candidates = [
        {"medicine_id": 1, "medicine_name": "PAROL ŞURUP", "alias": "PAROL"},
        {"medicine_id": 2, "medicine_name": "PAROL 500 MG TABLET", "alias": "PAROL"},
    ]
    chunks = [
        {
            "medicine_id": 2,
            "medicine_name": "PAROL 500 MG TABLET",
            "chunk_type": "indications",
            "chunk_text": "Kayıtlı kullanım alanı",
            "embedding": [1.0, 0.0],
            "source_name": "TİTCK KT",
        }
    ]
    monkeypatch.setattr(retrieval, "find_candidate_medicines", lambda *_, **__: candidates)
    monkeypatch.setattr(retrieval, "get_chunks", lambda **_: chunks)
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
    trace = {}

    results = retrieval.get_top_chunks(
        "Parol ne için kullanılır?",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.0,
        debug_trace=trace,
    )

    assert [item["medicine_name"] for item in results] == ["PAROL 500 MG TABLET"]
    assert trace["selected_medicine_id"] == 2


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


def _mock_single_medicine(monkeypatch, chunk):
    monkeypatch.setattr(
        retrieval,
        "find_candidate_medicines",
        lambda *_, **__: [
            {
                "medicine_id": 7,
                "medicine_name": "ALFA 10 MG TABLET",
                "alias": "ALFA",
                "matched_alias": "ALFA",
                "match_type": "exact_alias",
                "name_score": 1.0,
            }
        ],
    )
    monkeypatch.setattr(retrieval, "get_medicines_by_ids", lambda *_, **__: [])
    monkeypatch.setattr(retrieval, "get_chunks", lambda **_: [chunk])


def test_exact_pregnancy_section_survives_low_cosine_threshold(monkeypatch):
    chunk = {
        "chunk_id": 70,
        "medicine_id": 7,
        "medicine_name": "ALFA 10 MG TABLET",
        "chunk_type": "pregnancy",
        "chunk_text": "Gebelik döneminde kullanımı için doktorunuza danışınız.",
        "embedding": [0.0, 1.0],
        "source_name": "TİTCK KÜB",
    }
    _mock_single_medicine(monkeypatch, chunk)

    results = retrieval.get_top_chunks(
        "ALFA hamilelikte kullanılır mı?",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.35,
    )

    assert results
    assert results[0]["medicine_id"] == 7
    assert results[0]["chunk_type"] == "pregnancy"


def test_fts_lexical_result_is_used_when_vectors_do_not_qualify(monkeypatch):
    irrelevant_vector_chunk = {
        "chunk_id": 71,
        "medicine_id": 7,
        "medicine_name": "ALFA 10 MG TABLET",
        "chunk_type": "indications",
        "chunk_text": "Kullanım alanı metni.",
        "embedding": [0.0, 1.0],
    }
    lexical_chunk = {
        "chunk_id": 72,
        "medicine_id": 7,
        "medicine_name": "ALFA 10 MG TABLET",
        "chunk_type": "interactions",
        "chunk_text": "Rifampisin ile birlikte kullanımda etkileşim görülebilir.",
        "source_name": "TİTCK KÜB",
    }
    _mock_single_medicine(monkeypatch, irrelevant_vector_chunk)
    monkeypatch.setattr(
        retrieval, "search_medicine_chunks_fts", lambda *_, **__: [lexical_chunk]
    )

    results = retrieval.get_top_chunks(
        "ALFA hangi ilaçlarla etkileşir?",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.35,
    )

    assert results[0]["chunk_id"] == 72
    assert results[0]["retrieval_method"] == "fts5"


def test_embedding_retrieval_works_without_fts_match(monkeypatch):
    chunk = {
        "chunk_id": 73,
        "medicine_id": 7,
        "medicine_name": "ALFA 10 MG TABLET",
        "chunk_type": "general",
        "chunk_text": "Yerel prospektüs kayıt özeti.",
        "embedding": [1.0, 0.0],
        "source_name": "TİTCK KT",
    }
    _mock_single_medicine(monkeypatch, chunk)

    results = retrieval.get_top_chunks(
        "ALFA kayıt özeti",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.35,
    )

    assert results[0]["chunk_id"] == 73
    assert results[0]["retrieval_method"] == "embedding"


def test_production_path_skips_embedding_when_exact_section_exists(monkeypatch):
    chunk = {
        "chunk_id": 74,
        "medicine_id": 7,
        "medicine_name": "ALFA 10 MG TABLET",
        "chunk_type": "storage",
        "chunk_text": "25°C'nin altında saklayınız.",
        "source_name": "TİTCK KT",
    }
    _mock_single_medicine(monkeypatch, chunk)
    monkeypatch.setattr(
        retrieval,
        "generate_embedding",
        lambda _text: (_ for _ in ()).throw(
            AssertionError("embedding model should stay asleep")
        ),
    )
    trace = {}

    results = retrieval.get_top_chunks("ALFA nasıl saklanır?", debug_trace=trace)

    assert results[0]["chunk_id"] == 74
    assert trace["query_embedding_created"] is False
    assert trace["embedding_skipped_reason"] == "deterministic_evidence_found"


def test_explicit_strength_narrows_same_brand_candidates(monkeypatch):
    candidates = [
        {"medicine_id": 2, "medicine_name": "ACEPER 2 MG TABLET", "alias": "ACEPER"},
        {"medicine_id": 4, "medicine_name": "ACEPER 4 MG TABLET", "alias": "ACEPER"},
    ]
    chunks = [
        {
            "chunk_id": medicine_id,
            "medicine_id": medicine_id,
            "medicine_name": name,
            "chunk_type": "indications",
            "chunk_text": "Kayıtlı kullanım alanı.",
            "embedding": [1.0, 0.0],
            "source_name": "TİTCK KT",
        }
        for medicine_id, name in ((2, "ACEPER 2 MG TABLET"), (4, "ACEPER 4 MG TABLET"))
    ]
    monkeypatch.setattr(retrieval, "find_candidate_medicines", lambda *_, **__: candidates)
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda medicine_ids, **_: [
            chunk for chunk in chunks if chunk["medicine_id"] in medicine_ids
        ],
    )

    results = retrieval.get_top_chunks(
        "ACEPER 4 MG TABLET ne için kullanılır?",
        embedding_function=lambda _: [1.0, 0.0],
    )

    assert {result["medicine_id"] for result in results} == {4}
