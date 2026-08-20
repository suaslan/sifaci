from __future__ import annotations

import sqlite3

from src.database import (
    get_database_stats,
    get_chunks,
    get_medicine_by_name,
    initialize_database,
    insert_chunk,
    insert_medicine,
)
from src.ingestion import create_chunks, ingest_medicine_record, split_text_safely


def test_database_round_trip_and_parameterized_values(tmp_path):
    database_path = tmp_path / "nested" / "medicines.db"
    initialize_database(database_path)

    medicine_id = insert_medicine(
        {
            "medicine_name": "Test'); DROP TABLE medicines; --",
            "active_ingredient": "Etkin madde",
            "source_name": "Test kaynağı",
        },
        database_path=database_path,
    )
    insert_chunk(
        medicine_id,
        "Test parçası",
        "warnings",
        [0.1, -0.2, 0.3],
        database_path=database_path,
    )

    medicine = get_medicine_by_name(
        "test'); drop table medicines; --", database_path=database_path
    )
    chunks = get_chunks(medicine_id, database_path=database_path)

    assert medicine is not None
    assert medicine["medicine_id"] == medicine_id
    assert chunks[0]["embedding"] == [0.1, -0.2, 0.3]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM medicines").fetchone()[0] == 1


def test_ingestion_is_idempotent_and_creates_typed_chunks(tmp_path):
    database_path = tmp_path / "medicines.db"
    record = {
        "medicine_name": "Deneme 10 mg",
        "indications": "Birinci endikasyon. İkinci endikasyon.",
        "dosage": "Doz yalnızca resmi dokümandan alınır.",
        "side_effects": ["Baş ağrısı", "Bulantı"],
        "source": {"name": "TİTCK KT", "reference": "KT-123"},
    }

    first_id, first_count = ingest_medicine_record(
        record, database_path=database_path, embedding_function=lambda _: [1.0, 2.0]
    )
    second_id, second_count = ingest_medicine_record(
        record, database_path=database_path, embedding_function=lambda _: [1.0, 2.0]
    )
    chunks = get_chunks(first_id, database_path=database_path)

    assert first_id == second_id
    assert first_count == second_count == 3
    assert len(chunks) == 3
    assert {chunk["chunk_type"] for chunk in chunks} == {
        "indications",
        "dosage",
        "side_effects",
    }


def test_reimport_without_source_reuses_the_record(tmp_path):
    database_path = tmp_path / "medicines.db"

    first_id = insert_medicine("Kaynaksız kayıt", database_path=database_path)
    second_id = insert_medicine("kaynaksız kayıt", database_path=database_path)

    assert first_id == second_id


def test_long_sentences_are_never_cut_mid_sentence():
    sentence_one = "A" * 110 + "."
    sentence_two = "B" * 110 + "."

    chunks = split_text_safely(f"{sentence_one} {sentence_two}", max_chars=100)

    assert chunks == [sentence_one, sentence_two]


def test_create_chunks_uses_alias_without_duplicates():
    chunks = create_chunks(
        {
            "medicine_name": "Deneme",
            "dosage_information": "Bir doz metni.",
            "dosage": "Yinelenmemesi gereken metin.",
        }
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "dosage"


def test_database_stats(tmp_path):
    database_path = tmp_path / "medicines.db"
    medicine_id = insert_medicine("İstatistik ilacı", database_path=database_path)
    insert_chunk(
        medicine_id,
        "İstatistik parçası",
        "usage",
        [1.0, 0.0],
        database_path=database_path,
    )

    assert get_database_stats(database_path) == {
        "medicine_count": 1,
        "chunk_count": 1,
    }
