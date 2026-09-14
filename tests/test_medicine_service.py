from __future__ import annotations

from src.database import insert_chunk, insert_medicine, search_medicine_chunks_fts
from src.medicine_service import find_medicine, get_relevant_chunks
from src.rag import answer_query


def _medicine_database(tmp_path):
    path = tmp_path / "medicines.db"
    medicine_id = insert_medicine(
        {
            "medicine_name": "PAROL 500 MG TABLET",
            "active_ingredient": "Parasetamol",
            "source_name": "TİTCK KT",
            "source_reference": "https://example.test/parol-kt",
        },
        database_path=path,
    )
    insert_chunk(
        medicine_id,
        "Olası yan etkiler arasında bulantı yer alır.",
        "side_effects",
        section="4. Olası yan etkiler",
        source_url="https://example.test/parol-kt",
        source_type="TITCK",
        database_path=path,
    )
    insert_chunk(
        medicine_id,
        "Tablet ağızdan bir bardak su ile alınır.",
        "usage",
        section="3. Nasıl kullanılır?",
        source_url="https://example.test/parol-kt",
        source_type="TITCK",
        database_path=path,
    )
    return path, medicine_id


def test_find_medicine_by_name_and_turkish_normalization(tmp_path):
    path, medicine_id = _medicine_database(tmp_path)
    assert find_medicine("  parol   nedir? ", database_path=path)["medicine_id"] == medicine_id


def test_find_medicine_by_active_ingredient(tmp_path):
    path, medicine_id = _medicine_database(tmp_path)
    match = find_medicine("parasetamol yan etkileri?", database_path=path)
    assert match["medicine_id"] == medicine_id
    assert match["match_type"] == "active_ingredient"


def test_fts5_search_and_no_embedding_retrieval(tmp_path):
    path, medicine_id = _medicine_database(tmp_path)
    fts = search_medicine_chunks_fts(
        "yan etkiler bulantı",
        medicine_ids=[medicine_id],
        chunk_types=["side_effects"],
        database_path=path,
    )
    assert fts and "bulantı" in fts[0]["chunk_text"]
    chunks = get_relevant_chunks(
        "Parolun yan etkileri nelerdir?", database_path=path
    )
    assert chunks and chunks[0]["chunk_type"] == "side_effects"


def test_usage_question_selects_usage_section(tmp_path):
    path, _ = _medicine_database(tmp_path)
    chunks = get_relevant_chunks("Parol nasıl kullanılır?", database_path=path)
    assert chunks and {chunk["chunk_type"] for chunk in chunks} == {"usage"}


def test_unknown_medicine_and_missing_section_do_not_call_model(tmp_path):
    path, _ = _medicine_database(tmp_path)

    def must_not_call(_):
        raise AssertionError("Context yokken model çağrılmamalı")

    unknown = answer_query(
        "BİLİNMEYENİLAÇ yan etkileri?",
        retrieval_function=lambda question, top_k=5: get_relevant_chunks(
            question, database_path=path, top_k=top_k
        ),
        chat_function=must_not_call,
    )
    missing = answer_query(
        "Parol kontrendikasyonları nelerdir?",
        retrieval_function=lambda question, top_k=5: get_relevant_chunks(
            question, database_path=path, top_k=top_k
        ),
        chat_function=must_not_call,
    )
    assert "yeterli kaynak bulunamadı" in unknown
    assert "yeterli kaynak bulunamadı" in missing
