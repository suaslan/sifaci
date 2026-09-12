"""Verify public İlacabak matching and accessible HTML parsing for SELECTRA."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import find_medicines_by_name, get_ilacabak_document
from src.ilacabak_client import IlacabakClient, load_robots_policy
from src.prospektus_parser import parse_prospektus_html


def main() -> None:
    medicines = find_medicines_by_name("SELECTRA 50", limit=10)
    if not medicines:
        raise SystemExit("SELECTRA 50 veritabanında bulunamadı.")
    provider_records = {
        int(item["medicine_id"]): get_ilacabak_document(int(item["medicine_id"]))
        for item in medicines
    }
    medicine = next(
        (
            item
            for item in medicines
            if provider_records[int(item["medicine_id"])] is not None
        ),
        medicines[0],
    )
    client = IlacabakClient(load_robots_policy())
    try:
        candidates = client.search_candidates(str(medicine["medicine_name"]))
        if not candidates:
            raise SystemExit("İlacabak SELECTRA arama sonucu bulunamadı.")
        match = candidates[0]
        if match.score < 0.78:
            details = ", ".join(
                f"{item.product_name} ({item.score:.3f})" for item in candidates[:5]
            )
            raise SystemExit(f"İlacabak SELECTRA güvenilir eşleşmesi bulunamadı: {details}")
        metadata = client.get_product_metadata(match.product_url)
        prospectus = client.get_html(metadata.prospectus_url)
        sections = parse_prospektus_html(prospectus.html)
    finally:
        client.close()

    print(f"Medicine: {medicine['medicine_name']}")
    print(f"Match: {match.product_name}")
    print(f"Score: {match.score:.3f}")
    print(f"Product URL: {match.product_url}")
    print(f"HKT URL: {metadata.hkt_url or 'NONE'}")
    print(f"HKT source type: {metadata.hkt_source_type or 'NONE'}")
    print(f"KUB URL: {metadata.kub_url or 'NONE'}")
    print(f"KUB source type: {metadata.kub_source_type or 'NONE'}")
    print(f"Source date: {metadata.source_date or 'UNKNOWN'}")
    print(f"Accessible HTML sections: {len(sections)}")
    print(
        "Section types: "
        + (", ".join(item["chunk_type"] for item in sections) or "NONE")
    )
    print(f"Cached at: {prospectus.cache_path}")
    stored = provider_records.get(int(medicine["medicine_id"]))
    print(f"Stored provider record: {'YES' if stored else 'NO (verification is read-only)'}")
    if not sections:
        print(
            "Not: SELECTRA'nın güncel HKT/KÜB içeriği robots.txt ile yasaklanan "
            "/pdf/ yolunda; bağlantılar kaydedilebilir ancak crawler PDF'yi indirmez."
        )


if __name__ == "__main__":
    main()
