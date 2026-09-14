from __future__ import annotations

from src.database import (
    get_chunks,
    get_documents,
    get_medicine_by_name,
    insert_medicine,
    merge_medicine_into_canonical,
    replace_document_chunks,
    upsert_document,
)
from src.medicine_resolution import resolve_canonical_medicine


def test_document_spelling_variant_resolves_to_canonical_product():
    canonical = [
        {
            "medicine_id": 7,
            "medicine_name": "MUSCOFLEX DUO 100 MG/8 MG DEĞİŞTİRİLMİŞ SALIMLI TABLET",
            "active_ingredient": "Diklofenak sodyum, tiyokolşikosid",
            "company": "Bilim İlaç",
            "pharmaceutical_form": "Değiştirilmiş salımlı tablet",
        }
    ]
    resolution = resolve_canonical_medicine(
        {
            "product_name": "Muscoflex Duo 100mg / 8mg değiştirlmiş salımlı tablet",
            "active_ingredient": "diklofenak sodyum + tiyokolşikosid",
            "company": "Bilim İlaç Sanayi",
        },
        canonical,
    )
    assert resolution["status"] == "matched"
    assert resolution["medicine_id"] == 7


def test_ambiguous_strengthless_document_is_not_silently_merged():
    canonical = [
        {"medicine_id": 1, "medicine_name": "ALFA 10 MG TABLET"},
        {"medicine_id": 2, "medicine_name": "ALFA 20 MG TABLET"},
    ]
    resolution = resolve_canonical_medicine({"product_name": "ALFA TABLET"}, canonical)
    assert resolution["status"] != "matched"


def test_transactional_merge_preserves_document_chunk_embedding_and_alias(tmp_path):
    database_path = tmp_path / "merge.db"
    canonical_id = insert_medicine(
        {
            "medicine_name": "MUSCOFLEX DUO 100 MG/8 MG TABLET",
            "source_name": "TİTCK Ruhsatlı Beşeri Tıbbi Ürünler Listesi",
        },
        database_path=database_path,
    )
    duplicate_id = insert_medicine(
        {
            "medicine_name": "Muscoflex Duo 100mg / 8mg tablet",
            "source_name": "TİTCK KÜB/KT Listesi",
        },
        database_path=database_path,
    )
    document_id, _ = upsert_document(
        duplicate_id,
        "KT",
        "https://www.titck.gov.tr/test.pdf",
        local_path="D:/test.pdf",
        content_hash="hash",
        raw_text="3. Nasıl kullanılır? Metin",
        database_path=database_path,
    )
    replace_document_chunks(
        document_id,
        duplicate_id,
        [
            {
                "chunk_text": "İlaç: Muscoflex\nBölüm: Nasıl kullanılır\nKaynak metni",
                "chunk_type": "usage",
                "embedding": [1.0, 0.0],
                "embedding_model": "qwen3-embedding-0.6b",
                "source_type": "TITCK",
                "source_url": "https://www.titck.gov.tr/test.pdf",
            }
        ],
        database_path=database_path,
    )

    merge_medicine_into_canonical(
        duplicate_id, canonical_id, database_path=database_path
    )

    assert get_medicine_by_name(
        "Muscoflex Duo 100mg / 8mg tablet", database_path=database_path
    ) is None
    assert {row["medicine_id"] for row in get_documents(database_path=database_path)} == {
        canonical_id
    }
    chunks = get_chunks(canonical_id, database_path=database_path)
    assert len(chunks) == 1
    assert chunks[0]["embedding"] == [1.0, 0.0]
