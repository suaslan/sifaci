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

    assert "İlgili bilgi belgelerde bulunamadı." in answer
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

    assert answer.startswith("İlgili bilgi belgelerde bulunamadı.")


def test_false_missing_model_answer_falls_back_to_retrieved_side_effects():
    chunks = [
        {
            **SOURCES[0],
            "chunk_type": "side_effects",
            "chunk_text": "İlaç: ALFA 10 mg\nKategori: side_effects\nBulantı\nBaş dönmesi",
        },
        {
            **SOURCES[0],
            "chunk_type": "serious_side_effects",
            "chunk_text": "İlaç: ALFA 10 mg\nKategori: serious_side_effects\nSolunum güçlüğü",
        },
    ]

    answer = answer_query(
        "Alfa ilacının yan etkileri nelerdir?",
        retrieval_function=lambda *_, **__: chunks,
        chat_function=lambda _: "İlgili bilgi belgelerde bulunamadı.",
    )

    assert "Kaynakta bildirilen yan etkiler:" in answer
    assert "- Bulantı" in answer
    assert "Ciddi yan etkiler:" in answer
    assert "- Solunum güçlüğü" in answer
    assert "- İlaç:" not in answer
    assert "- Kategori:" not in answer
    assert "Kaynaklar: TİTCK KÜB" in answer


def test_unknown_medicine_does_not_call_chat_or_cite_wrong_source():
    def fail_chat(_):
        raise AssertionError("Unknown medicine must not reach the chat model")

    answer = answer_query(
        "Aspirin ilacının yan etkileri nelerdir?",
        retrieval_function=lambda *_, **__: [
            {
                **SOURCES[0],
                "medicine_name": "PAROL 500 MG TABLET",
                "chunk_type": "side_effects",
                "chunk_text": "İlaç: PAROL 500 MG TABLET\nKategori: side_effects\nBulantı",
            }
        ],
        chat_function=fail_chat,
    )

    assert answer.startswith("İlgili bilgi belgelerde bulunamadı.")
    assert "Kaynaklar: Bulunamadı" in answer
    assert "TİTCK KÜB" not in answer


def test_fuzzy_registered_name_still_reaches_chat():
    answer = answer_query(
        "Alfaa ilacının doz bilgisi nedir?",
        retrieval_function=lambda *_, **__: SOURCES,
        chat_function=lambda _: "Kaynakta genel doz bilgisi vardır.",
    )

    assert "Kaynakta genel doz bilgisi vardır." in answer


def test_chat_failure_uses_exact_retrieved_field_fallback():
    chunks = [
        {
            **SOURCES[0],
            "chunk_type": "side_effects",
            "chunk_text": "İlaç: ALFA 10 mg\nKategori: side_effects\nBulantı",
        }
    ]

    def fail_chat(_):
        raise RuntimeError("local model allocation failed")

    answer = answer_query(
        "Alfa ilacının yan etkileri nelerdir?",
        retrieval_function=lambda *_, **__: chunks,
        chat_function=fail_chat,
    )

    assert "Kaynakta bildirilen yan etkiler:" in answer
    assert "- Bulantı" in answer


def test_unhelpful_model_sentence_is_replaced_with_grounded_answer():
    chunks = [
        {
            **SOURCES[0],
            "chunk_type": "interactions",
            "chunk_text": (
                "İlaç: ALFA 10 mg\nKategori: interactions\nBölüm: Etkileşimler\n"
                "• Rifampisin ile birlikte kullanıldığında doktorunuza bildiriniz.\n"
                "Belge Doğrulama Kodu: ABC"
            ),
        }
    ]

    answer = answer_query(
        "Alfa hangi ilaçlarla birlikte kullanılamaz?",
        retrieval_function=lambda *_, **__: chunks,
        chat_function=lambda _: (
            "Bu bilgilere göre kaynakta veri tabanında bulunabilmeniz için "
            "birlikte kullanılamadı."
        ),
    )

    assert "Etkileşimler:" in answer
    assert "Rifampisin" in answer
    assert "bulunabilmeniz" not in answer
    assert "Belge Doğrulama Kodu" not in answer


def test_excessive_source_copy_is_replaced_with_bounded_answer():
    chunks = [
        {
            **SOURCES[0],
            "chunk_type": "side_effects",
            "chunk_text": (
                "İlaç: ALFA 10 mg\nKategori: side_effects\nBölüm: Yan etkiler\n"
                "• Bulantı\n• Baş dönmesi\nBelge Takip Adresi: https://example.test"
            ),
        }
    ]

    answer = answer_query(
        "Alfa yan etkileri nelerdir?",
        retrieval_function=lambda *_, **__: chunks,
        chat_function=lambda _: "Bulantı " * 400,
    )

    assert "- Bulantı" in answer
    assert "- Baş dönmesi" in answer
    assert len(answer) < 1_500


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
    assert "KULLANICI SORUSU:" in trace["llm_user_prompt"]
    assert "RETRIEVED CONTEXT:" in trace["llm_user_prompt"]
    assert trace["llm_system_prompt"]
    assert trace["model_called"] is True


def test_catalog_not_found_uses_distinct_fallback_and_skips_llm():
    trace = {}

    def retrieval(_query, top_k, debug_trace):
        debug_trace.update(
            {
                "retrieval_state": "medicine_not_found",
                "fallback_reason": "medicine_not_found",
                "detected_medicine": None,
            }
        )
        return []

    answer = answer_query(
        "Olmayanilaç yan etkileri",
        retrieval_function=retrieval,
        chat_function=lambda _: (_ for _ in ()).throw(AssertionError("LLM called")),
        debug_trace=trace,
    )

    assert answer.startswith("Bu ilaç ürün kataloğunda bulunamadı.")
    assert trace["model_called"] is False


def test_catalog_entry_without_rag_data_uses_processing_fallback():
    trace = {}

    def retrieval(_query, top_k, debug_trace):
        debug_trace.update(
            {
                "retrieval_state": "rag_not_processed",
                "fallback_reason": "rag_not_processed",
                "detected_medicine": "MUSCOFLEX",
            }
        )
        return []

    answer = answer_query(
        "muscoflex yan etkileri",
        retrieval_function=retrieval,
        chat_function=lambda _: (_ for _ in ()).throw(AssertionError("LLM called")),
        debug_trace=trace,
    )

    assert answer.startswith(
        "MUSCOFLEX kayıtlı ancak KÜB/Kullanma Talimatı henüz yerel bilgi tabanına işlenmemiş."
    )


def test_existing_source_without_requested_section_uses_section_fallback():
    trace = {}

    def retrieval(_query, top_k, debug_trace):
        debug_trace.update(
            {
                "retrieval_state": "section_missing",
                "fallback_reason": "section_missing",
                "detected_medicine": "MUSCOFLEX",
            }
        )
        return []

    answer = answer_query(
        "muscoflex yan etkileri",
        retrieval_function=retrieval,
        chat_function=lambda _: (_ for _ in ()).throw(AssertionError("LLM called")),
        debug_trace=trace,
    )

    assert answer.startswith(
        "MUSCOFLEX için kaynak mevcut ancak sorulan bilgi ilgili belgelerde bulunamadı."
    )
