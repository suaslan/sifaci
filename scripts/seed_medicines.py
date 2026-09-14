"""Create a small, idempotent RAG development database.

The bundled text is deliberately marked as sample data. Production imports
must replace it with current TİTCK-approved KÜB/KT documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.database import insert_chunk, upsert_medicine


SOURCE_URL = "https://www.titck.gov.tr/kubkt"
SOURCE_NAME = "ÖRNEK VERİ — üretimde güncel TİTCK KÜB/KT ile değiştirin"

SAMPLE_MEDICINES = (
    {
        "medicine_name": "PAROL ÖRNEK TABLET",
        "active_ingredient": "Parasetamol",
        "manufacturer": "Örnek üretici",
        "chunks": (
            ("active_ingredient", "Etkin madde: parasetamol.", "Etkin madde"),
            ("indications", "Ağrı ve ateş belirtilerinin giderilmesi amacıyla kullanılır.", "Ne için kullanılır"),
            ("usage", "Kullanım için ürünün güncel Kullanma Talimatını izleyin; bu örnek kayıtta doz ve sıklık yer almaz.", "Nasıl kullanılır"),
            ("side_effects", "Bu örnek kayıtta yan etki ayrıntısı bulunmamaktadır.", "Olası yan etkiler"),
        ),
    },
    {
        "medicine_name": "MUSCOFLEX ÖRNEK KAPSÜL",
        "active_ingredient": "Tiyokolşikosid",
        "manufacturer": "Örnek üretici",
        "chunks": (
            ("active_ingredient", "Etkin madde: tiyokolşikosid.", "Etkin madde"),
            ("indications", "Kas gevşetici etki amacıyla kullanılan bir üründür.", "Ne için kullanılır"),
        ),
    },
    {
        "medicine_name": "ARVELES ÖRNEK TABLET",
        "active_ingredient": "Deksketoprofen trometamol",
        "manufacturer": "Örnek üretici",
        "chunks": (
            ("active_ingredient", "Etkin madde: deksketoprofen trometamol.", "Etkin madde"),
            ("usage", "Aç veya tok kullanım bilgisi bu örnek kaynağa eklenmemiştir.", "Nasıl kullanılır"),
        ),
    },
)


def seed(database_path: Path) -> int:
    inserted_chunks = 0
    for item in SAMPLE_MEDICINES:
        medicine_id, status = upsert_medicine(
            {
                "medicine_name": item["medicine_name"],
                "active_ingredient": item["active_ingredient"],
                "source_name": SOURCE_NAME,
                "source_reference": SOURCE_URL,
            },
            database_path=database_path,
        )
        if status == "inserted":
            for chunk_type, content, section in item["chunks"]:
                insert_chunk(
                    medicine_id,
                    content,
                    chunk_type,
                    section=section,
                    source_url=SOURCE_URL,
                    source_type="SAMPLE",
                    source_priority=1,
                    database_path=database_path,
                )
                inserted_chunks += 1
    return inserted_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data") / "seed_medicines.db",
        help="Hedef SQLite dosyası (varsayılan: data/seed_medicines.db)",
    )
    args = parser.parse_args()
    count = seed(args.database.resolve())
    print(f"Seed tamamlandı: {args.database.resolve()} ({count} yeni chunk)")


if __name__ == "__main__":
    main()
