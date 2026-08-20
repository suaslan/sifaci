from __future__ import annotations

from src.rag import DISCLAIMER, answer_query


SOURCES = [
    {
        "medicine_name": "ALFA 10 mg",
        "chunk_type": "dosage",
        "chunk_text": "İlaç: ALFA 10 mg\nKategori: dosage\nGenel doz metni kaynakta bulunur.",
        "similarity_score": 0.9,
        "source_name": "TİTCK KÜB",
    }
]


def test_answer_query_builds_separated_context_and_sources():
    captured = {}

    def fake_chat(messages):
        captured["messages"] = messages
        return "Kaynağa dayalı cevap."

    answer = answer_query(
        "Alfa genel doz bilgisi nedir?",
        retrieval_function=lambda *_, **__: SOURCES,
        chat_function=fake_chat,
    )

    prompt = captured["messages"][1]["content"]
    assert "[KAYNAK 1]" in prompt
    assert "İlaç: ALFA 10 mg" in prompt
    assert "Kategori: dosage" in prompt
    assert "Kaynak: TİTCK KÜB" in prompt
    assert "Kaynaklar: TİTCK KÜB" in answer
    assert answer.endswith(DISCLAIMER)


def test_missing_retrieval_does_not_call_chat():
    def fail_chat(_):
        raise AssertionError("chat model must not be called")

    answer = answer_query(
        "Bilinmeyen bilgi",
        retrieval_function=lambda *_, **__: [],
        chat_function=fail_chat,
    )

    assert "Bu bilgi mevcut ilaç veri tabanında bulunmuyor." in answer
    assert "Kaynaklar: Bulunamadı" in answer
    assert answer.endswith(DISCLAIMER)


def test_personalized_dose_is_blocked_before_chat():
    def fail_chat(_):
        raise AssertionError("personal dose request must not reach chat model")

    answer = answer_query(
        "Ben 70 kiloyum, kaç tane almalıyım?",
        retrieval_function=lambda *_, **__: SOURCES,
        chat_function=fail_chat,
    )

    assert "Kişiye özel doz" in answer
    assert "doktorunuza veya eczacınıza danışın" in answer
    assert "Genel doz metni" in answer
    assert answer.endswith(DISCLAIMER)


def test_unsupported_generated_quantity_is_rejected():
    answer = answer_query(
        "Alfa nasıl kullanılır?",
        retrieval_function=lambda *_, **__: SOURCES,
        chat_function=lambda _: "Günde 50 mg kullanılmalıdır.",
    )

    assert answer.startswith("Bu bilgi mevcut ilaç veri tabanında bulunmuyor.")


def test_serious_side_effect_gets_visible_warning():
    chunks = [
        {
            **SOURCES[0],
            "chunk_type": "serious_side_effects",
            "chunk_text": "Solunum güçlüğü oluşabilir.",
        }
    ]
    answer = answer_query(
        "Yan etkiler nelerdir?",
        retrieval_function=lambda *_, **__: chunks,
        chat_function=lambda _: "Solunum güçlüğü kaynakta listelenmiştir.",
    )

    assert answer.startswith("Önemli güvenlik uyarısı:")


def test_debug_trace_contains_retrieved_context_and_model_status():
    trace = {}

    def debug_retrieval(query, top_k, debug_trace):
        debug_trace["query_embedding_created"] = True
        debug_trace["query_embedding_dimension"] = 2
        debug_trace["top_chunks"] = SOURCES
        return SOURCES

    answer_query(
        "Alfa genel doz bilgisi nedir?",
        retrieval_function=debug_retrieval,
        chat_function=lambda _: "Kaynağa dayalı cevap.",
        debug_trace=trace,
    )

    assert trace["query_embedding_created"] is True
    assert "[KAYNAK 1]" in trace["retrieved_context"]
    assert "Kategori: dosage" in trace["retrieved_context"]
    assert trace["model_called"] is True
