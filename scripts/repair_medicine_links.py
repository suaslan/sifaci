"""Safely relink legacy KÜB/KT-only medicine rows to master catalog products."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config import DATABASE_PATH
from src.database import get_all_medicines, get_documents, merge_medicine_into_canonical
from src.medicine_names import normalize_medicine_name
from src.medicine_resolution import medicine_signature, resolve_canonical_medicine


def _is_master(row: dict[str, object]) -> bool:
    source = normalize_medicine_name(str(row.get("source_name") or ""))
    return "ruhsatli beseri tibbi urunler listesi" in source


def _is_legacy_document_identity(row: dict[str, object]) -> bool:
    source = normalize_medicine_name(str(row.get("source_name") or ""))
    return "kub kt listesi" in source


def _backup_database(database_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = database_path.with_name(f"{database_path.stem}.pre-link-repair-{stamp}.db")
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--apply", action="store_true", help="Dry-run yerine güvenli merge uygular.")
    parser.add_argument("--limit", type=int, help="İncelenecek legacy kimlik sayısı.")
    args = parser.parse_args()

    database_path = args.database.resolve()
    medicines = get_all_medicines(database_path=database_path)
    documented_ids = {int(row["medicine_id"]) for row in get_documents(database_path=database_path)}
    masters = [row for row in medicines if _is_master(row)]
    legacy = [
        row for row in medicines
        if _is_legacy_document_identity(row) and int(row["medicine_id"]) in documented_ids
    ]
    if args.limit is not None:
        legacy = legacy[: max(0, args.limit)]

    by_brand: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_first_token: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in masters:
        signature = medicine_signature(
            str(row["medicine_name"]), str(row.get("pharmaceutical_form") or "")
        )
        brand = str(signature["brand"])
        by_brand[brand].append(row)
        if brand:
            by_first_token[brand.split()[0]].append(row)

    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    ambiguous: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for row in legacy:
        signature = medicine_signature(
            str(row["medicine_name"]), str(row.get("pharmaceutical_form") or "")
        )
        brand = str(signature["brand"])
        pool = by_brand.get(brand) or by_first_token.get(brand.split()[0] if brand else "", [])
        resolution = resolve_canonical_medicine(row, pool)
        if resolution["status"] == "matched":
            matched.append((row, resolution))
        elif resolution["status"] == "ambiguous":
            ambiguous.append({"medicine_id": row["medicine_id"], "medicine_name": row["medicine_name"], **resolution})
        else:
            unresolved.append({"medicine_id": row["medicine_id"], "medicine_name": row["medicine_name"], **resolution})

    print(json.dumps({
        "database": str(database_path),
        "mode": "apply" if args.apply else "dry-run",
        "master_catalog_medicines": len(masters),
        "legacy_document_identities_checked": len(legacy),
        "high_confidence_matches": len(matched),
        "ambiguous_not_merged": len(ambiguous),
        "unresolved_not_merged": len(unresolved),
        "matched_examples": [
            {
                "duplicate_id": row["medicine_id"],
                "duplicate_name": row["medicine_name"],
                "canonical_id": resolution["medicine_id"],
                "canonical_name": resolution["medicine_name"],
                "score": resolution["score"],
            }
            for row, resolution in matched[:20]
        ],
        "ambiguous_examples": ambiguous[:10],
        "unresolved_examples": unresolved[:10],
    }, ensure_ascii=False, indent=2))

    if not args.apply:
        print("Dry-run tamamlandı. Değişiklik için --apply kullanın.")
        return
    backup_path = _backup_database(database_path)
    totals = {"moved_documents": 0, "merged_documents": 0, "moved_unlinked_chunks": 0}
    for index, (row, resolution) in enumerate(matched, start=1):
        result = merge_medicine_into_canonical(
            int(row["medicine_id"]),
            int(resolution["medicine_id"]),
            database_path=database_path,
        )
        for key in totals:
            totals[key] += result[key]
        if index % 250 == 0:
            print(f"Relink ilerleme: {index}/{len(matched)}")
    print(json.dumps({"backup": str(backup_path), "merged_medicines": len(matched), **totals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
