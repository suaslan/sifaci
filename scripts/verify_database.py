"""Report TİTCK synchronization coverage and verify one medicine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import VERIFICATION_MEDICINE_NAME
from src.database import (
    find_candidate_medicines,
    find_duplicate_database_paths,
    get_documents,
    get_extended_database_stats,
    get_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Şifacı veritabanını doğrular.")
    parser.add_argument("--medicine", default=VERIFICATION_MEDICINE_NAME)
    args = parser.parse_args()

    stats = get_extended_database_stats()
    print(f"Total medicines: {stats.get('medicines', 0)}")
    print(f"Total aliases: {stats.get('aliases', 0)}")
    print(f"Total KÜB documents: {stats.get('kub_documents', 0)}")
    print(f"Total KT documents: {stats.get('kt_documents', 0)}")
    print(f"Total chunks: {stats.get('chunks', 0)}")
    print(f"Chunks with embeddings: {stats.get('embedded_chunks', 0)}")
    print(f"Last sync: {stats.get('last_sync') or '-'}")
    duplicate_paths = find_duplicate_database_paths()
    print(
        "Other medicines.db files: "
        + (", ".join(str(path) for path in duplicate_paths) or "NONE")
    )

    matches = find_candidate_medicines(args.medicine)
    print(f"\nMatched products: {len(matches)}")
    for match in matches:
        medicine_id = int(match["medicine_id"])
        documents = get_documents(medicine_id)
        chunks = get_chunks(medicine_id)
        sections = sorted(
            {
                str(chunk.get("section") or chunk.get("chunk_type") or "general")
                for chunk in chunks
            }
        )
        embedded = sum(bool(chunk.get("embedding")) for chunk in chunks)
        print(f"- {match['medicine_name']}")
        print(
            "  Documents: "
            + (", ".join(str(item["document_type"]) for item in documents) or "NONE")
        )
        print(f"  Sections: {', '.join(sections) or 'NONE'}")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Embedding status: {embedded}/{len(chunks)}")

    if not matches:
        raise SystemExit(f"Medicine not found: {args.medicine}")


if __name__ == "__main__":
    main()
