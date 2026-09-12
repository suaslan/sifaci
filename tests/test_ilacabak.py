from __future__ import annotations

from pathlib import Path

import pytest

from src import retrieval
from src.database import (
    get_chunks,
    get_dosage_rules,
    get_ilacabak_document,
    initialize_database,
    insert_medicine,
    iter_ilacabak_candidates,
    save_ilacabak_document,
)
from src.dosage_parser import extract_query_age, parse_dosage_rules
from src.ilacabak_client import CachedPage, IlacabakClient, IlacabakError, RobotsPolicy
from src.prospektus_parser import parse_prospektus_html


def test_robots_policy_allows_html_but_denies_pdf():
    policy = RobotsPolicy("User-agent: *\nDisallow: /pdf/\n")

    policy.assert_allowed("https://www.ilacabak.com/selectra-50-123/prospektus")
    with pytest.raises(IlacabakError):
        policy.assert_allowed("https://www.ilacabak.com/pdf/selectra.pdf")


def test_search_and_product_metadata_are_extracted(monkeypatch, tmp_path):
    policy = RobotsPolicy("User-agent: *\nDisallow: /pdf/\n")
    client = IlacabakClient(policy)
    search_html = """
        <a href="/selectra-25-mg-film-tablet-111">SELECTRA 25 MG FILM TABLET</a>
        <a href="/selectra-50-mg-film-tablet-222">SELECTRA 50 MG FILM TABLET</a>
    """
    product_html = """
        <a href="/pdf/hkt.pdf" title="Kullanma Talimatı">HKT</a>
        <a href="https://titck.gov.tr/kub.pdf" title="Kısa Ürün Bilgisi">KÜB</a>
        <div>Son Güncelleme: 4 Eylül 2026 Paylaş</div>
    """

    def fake_get_html(url: str, *, refresh: bool = False):
        html = product_html if url.endswith("-222") else search_html
        return CachedPage(url, html, tmp_path / "page.html", "hash", 0, False)

    monkeypatch.setattr(client, "get_html", fake_get_html)
    try:
        match = client.search_medicine("SELECTRA 50 MG FILM TABLET")
        assert match is not None
        assert match.product_url.endswith("-222")
        metadata = client.get_product_metadata(match.product_url)
    finally:
        client.close()

    assert metadata.prospectus_url.endswith("-222/prospektus")
    assert metadata.hkt_source_type == "ILACABAK_MIRROR"
    assert metadata.kub_source_type == "TITCK"
    assert metadata.source_date == "4 Eylül 2026"


def test_section_aware_prospectus_parser_keeps_categories_separate():
    html = """
    <div id="iceriksol"><div class="kutucuksol"><p>
      ENDİKASYONLARI<br/>Depresyon tedavisinde kullanılır.<br/>
      KULLANIM ŞEKLİ VE DOZU<br/>7-12 yaş çocuklarda günde 2 kez ağızdan alınır.<br/>
      KONTRENDİKASYONLARI<br/>Etken maddeye duyarlılıkta kullanılmaz.<br/>
      UYARILAR / ÖNLEMLER<br/>Doktor kontrolü gerektirir.<br/>
      YAN ETKİLER<br/>Bulantı görülebilir.<br/>
      İLAÇ ETKİLEŞİMLERİ<br/>Birlikte kullanılan ilaçları bildiriniz.
    </p></div></div>
    """

    sections = parse_prospektus_html(html)

    assert [item["chunk_type"] for item in sections] == [
        "indications",
        "dosage",
        "contraindications",
        "warnings",
        "common_side_effects",
        "interactions",
    ]
    assert sections[1]["text"] == "7-12 yaş çocuklarda günde 2 kez ağızdan alınır."


@pytest.mark.parametrize(
    ("population", "expected_min", "expected_max"),
    [
        ("7-12 yaş", 7, 12),
        ("12 yaş ve üzeri", 12, None),
        ("12 yaşından büyük", 12, None),
        ("6 yaş altı", None, 6),
    ],
)
def test_dosage_parser_extracts_only_explicit_age_bounds(
    population, expected_min, expected_max
):
    text = f"{population} çocuklarda günde 3-4 defa ağızdan 1 tablet alınır."

    [rule] = parse_dosage_rules(
        text,
        source_type="ILACABAK",
        source_url="https://www.ilacabak.com/test/prospektus",
    )

    assert rule["min_age"] == expected_min
    assert rule["max_age"] == expected_max
    assert rule["age_unit"] == "year"
    assert rule["frequency_text"] == "günde 3-4 defa"
    assert "ağızdan" in rule["route_text"]


def test_dosage_parser_does_not_invent_age_and_preserves_frequency():
    [rule] = parse_dosage_rules(
        "Yetişkinler için 8 saatte bir uygulanır; 24 saatte en fazla 4 doz.",
        source_type="ILACABAK",
        source_url="https://www.ilacabak.com/test/prospektus",
    )

    assert rule["min_age"] is None
    assert rule["max_age"] is None
    assert rule["frequency_text"] == "8 saatte bir; 24 saatte en fazla 4 doz"
    assert extract_query_age("Bu ilaç 8 yaşında nasıl kullanılır?") == (8.0, "year")


def test_database_migration_save_and_resume(tmp_path):
    database = tmp_path / "medicines.db"
    initialize_database(database)
    medicine_id = insert_medicine(
        {"medicine_name": "SELECTRA 50 MG FILM TABLET", "source_name": "TİTCK"},
        database_path=database,
    )
    metadata = {
        "product_url": "https://www.ilacabak.com/selectra-50-222",
        "prospectus_url": "https://www.ilacabak.com/selectra-50-222/prospektus",
        "hkt_url": "/pdf/hkt.pdf",
        "kub_url": "/pdf/kub.pdf",
        "hkt_source_type": "ILACABAK_MIRROR",
        "kub_source_type": "ILACABAK_MIRROR",
        "source_date": "4 Eylül 2026",
        "cache_path": str(tmp_path / "cached.html"),
        "content_hash": "content-hash",
        "retry_count": 0,
    }
    chunk = {
        "chunk_text": "İlaç: SELECTRA\nBölüm: Kullanım Şekli ve Dozu\n7-12 yaş günde 2 kez.",
        "chunk_type": "dosage",
        "section": "Kullanım Şekli ve Dozu",
        "chunk_hash": "chunk-hash",
        "embedding": [1.0, 0.0],
        "embedding_model": "test-model",
    }
    rules = parse_dosage_rules(
        "7-12 yaş çocuklarda günde 2 kez ağızdan 1 tablet alınır.",
        source_type="ILACABAK",
        source_url=metadata["prospectus_url"],
        source_date=metadata["source_date"],
    )

    save_ilacabak_document(
        medicine_id,
        metadata,
        [chunk],
        rules,
        database_path=database,
    )

    stored = get_ilacabak_document(medicine_id, database_path=database)
    assert stored is not None
    assert stored["embedding_status"] == "DONE"
    assert get_chunks(medicine_id, database_path=database)[0]["source_type"] == "ILACABAK"
    age_rules = get_dosage_rules(
        [medicine_id], age=8, database_path=database
    )
    assert len(age_rules) == 1
    assert list(iter_ilacabak_candidates(database_path=database)) == []
    assert len(list(iter_ilacabak_candidates(update=True, database_path=database))) == 1
    assert len(
        list(
            iter_ilacabak_candidates(
                update=True,
                medicine_query="SELECTRA 50",
                database_path=database,
            )
        )
    ) == 1
    assert list(
        iter_ilacabak_candidates(
            update=True,
            medicine_query="SELECTRA%' OR 1=1 --",
            database_path=database,
        )
    ) == []


def test_retrieval_places_age_rule_before_semantic_results(monkeypatch):
    candidates = [
        {
            "medicine_id": 5,
            "medicine_name": "TESTRA 50 MG TABLET",
            "alias": "TESTRA",
            "name_score": 1.0,
        }
    ]
    monkeypatch.setattr(retrieval, "find_candidate_medicines", lambda *_, **__: candidates)
    monkeypatch.setattr(
        retrieval,
        "get_dosage_rules",
        lambda *_, **__: [
            {
                "medicine_id": 5,
                "frequency_text": "günde 2 kez",
                "raw_source_text": "7-12 yaş çocuklarda günde 2 kez.",
                "source_type": "ILACABAK",
            }
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "get_chunks",
        lambda **_: [
            {
                "medicine_name": "TESTRA 50 MG TABLET",
                "chunk_type": "frequency",
                "chunk_text": "Semantik genel sonuç",
                "embedding": [1.0, 0.0],
                "source_name": "Başka kaynak",
                "source_type": "USER",
            }
        ],
    )

    results = retrieval.get_top_chunks(
        "TESTRA 8 yaşında günde kaç kez kullanılır?",
        embedding_function=lambda _: [1.0, 0.0],
        minimum_score=0.0,
    )

    assert results[0]["chunk_text"] == "günde 2 kez"
    assert results[0]["source_name"] == "İlacabak prospektüsü"
    assert results[1]["chunk_text"] == "Semantik genel sonuç"
