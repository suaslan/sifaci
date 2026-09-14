from __future__ import annotations

from src.database import (
    find_candidate_medicines,
    get_database_stats,
    insert_medicine,
)
from src.ingestion import ingest_directory, load_medicine_file
from src.leaflet_parser import create_document_chunks, parse_leaflet_sections


def test_csv_bulk_ingestion_reports_insert_duplicate_and_update(tmp_path):
    documents = tmp_path / "medicines"
    documents.mkdir()
    csv_path = documents / "bulk.csv"
    database_path = tmp_path / "bulk.db"
    csv_path.write_text(
        "medicine_name;active_ingredient;warnings;source_name\n"
        "CSV İLACI 10 MG;Etken;İlk uyarı;TİTCK KT\n",
        encoding="utf-8",
    )

    assert load_medicine_file(csv_path)[0]["medicine_name"] == "CSV İLACI 10 MG"
    first = ingest_directory(
        documents,
        database_path=database_path,
        embedding_function=lambda _: [1.0, 0.0],
    )
    second = ingest_directory(
        documents,
        database_path=database_path,
        embedding_function=lambda _: [1.0, 0.0],
    )

    csv_path.write_text(
        "medicine_name;active_ingredient;warnings;source_name\n"
        "CSV İLACI 10 MG;Etken;Güncellenen uyarı;TİTCK KT\n",
        encoding="utf-8",
    )
    third = ingest_directory(
        documents,
        database_path=database_path,
        embedding_function=lambda _: [1.0, 0.0],
    )

    assert first["inserted"] == 1
    assert first["chunks"] == 2
    assert second["duplicates"] == 1
    assert second["chunks"] == 0
    assert third["updated"] == 1
    assert third["chunks"] == 2
    assert get_database_stats(database_path) == {
        "medicine_count": 1,
        "chunk_count": 2,
    }


def test_brand_alias_matches_all_strength_variants(tmp_path):
    database_path = tmp_path / "aliases.db"
    insert_medicine("SELECTRA 25 MG FİLM TABLET", database_path=database_path)
    insert_medicine("SELECTRA 50 MG FİLM TABLET", database_path=database_path)
    insert_medicine("SELECTRA 100 MG FİLM TABLET", database_path=database_path)

    matches = find_candidate_medicines(
        "selectra yan etkileri",
        database_path=database_path,
    )

    assert {match["medicine_name"] for match in matches} == {
        "SELECTRA 25 MG FİLM TABLET",
        "SELECTRA 50 MG FİLM TABLET",
        "SELECTRA 100 MG FİLM TABLET",
    }


def test_hyphenated_brand_matches_query_without_hyphen(tmp_path):
    database_path = tmp_path / "hyphenated-aliases.db"
    insert_medicine("A-FERİN 300 MG KAPSÜL", database_path=database_path)
    insert_medicine("A-FERİN FORTE 650 MG TABLET", database_path=database_path)

    matches = find_candidate_medicines(
        "AFERİN nasıl kullanılır?",
        database_path=database_path,
    )

    assert {match["medicine_name"] for match in matches} == {
        "A-FERİN 300 MG KAPSÜL",
        "A-FERİN FORTE 650 MG TABLET",
    }


def test_leaflet_parser_keeps_sections_separate():
    raw_text = """
1. TEST nedir ve ne için kullanılır?
Kayıtlı endikasyon metni.
2. TEST'i kullanmadan önce dikkat edilmesi gerekenler
Kayıtlı uyarı metni.
3. TEST nasıl kullanılır?
Kayıtlı kullanım metni.
4. Olası yan etkiler nelerdir?
Aşağıdakilerden herhangi birini fark ederseniz hemen doktora başvurunuz.
Ciddi reaksiyon.
Çok yaygın:
Baş ağrısı.
5. TEST'in saklanması
25 derecenin altında saklanır.
"""

    sections = parse_leaflet_sections(raw_text, "KT")
    chunks = create_document_chunks("TEST", sections, min_chars=200, max_chars=800)
    chunk_types = {chunk["chunk_type"] for chunk in chunks}

    assert "indications" in chunk_types
    assert "warnings" in chunk_types
    assert "usage" in chunk_types
    assert "serious_side_effects" in chunk_types
    assert "common_side_effects" in chunk_types
    assert "storage" in chunk_types


def test_leaflet_parser_handles_spaced_numbering_and_nested_kt_topics():
    raw_text = """
4. 1   Terapötik endikasyonlar
Kayıtlı endikasyon metni.
4．2. Pozoloji ve uygulama şekli
Günde iki kez uygulanır.

Hamilelikte kullanım
Tedavi sırasında doktorunuza danışınız.

Emzirme döneminde kullanım
Anne sütü hakkında kaynak bilgisi.

Araç ve makine kullanımı
Araç kullanırken dikkat edilmelidir.

Kullanmanız gerekenden daha fazlasını kullandıysanız
Doktorunuza başvurunuz.

Kullanmayı unutursanız
Çift doz almayınız.
"""

    kub_sections = parse_leaflet_sections(raw_text, "KUB")
    kt_sections = parse_leaflet_sections(raw_text, "KT")

    kub_types = {section["chunk_type"] for section in kub_sections}
    kt_types = {section["chunk_type"] for section in kt_sections}

    assert "indications" in kub_types
    assert "dosage" in kub_types
    assert "pregnancy" in kt_types
    assert "breastfeeding" in kt_types
    assert "driving" in kt_types
    assert "overdose" in kt_types
    assert "missed_dose" in kt_types
