"""Diagnose the canonical JSON -> SQLite -> retrieval path."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATABASE_PATH, EMBEDDING_MODEL_NAME, MEDICINES_DIR
from src.database import (
    find_duplicate_database_paths,
    get_chunks,
    get_database_stats,
    get_readiness_stats,
)
from src.ingestion import discover_medicine_files
from src.retrieval import get_top_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Şifacı veri zincirini kontrol eder.")
    parser.add_argument(
        "--query",
        default="Parol'un yan etkileri nelerdir?",
        help="Retrieval için kullanılacak test sorusu.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Foundry embedding modelini çağırmadan yalnız dosya/DB kontrolü yapar.",
    )
    args = parser.parse_args()

    database_path = Path(DATABASE_PATH).resolve()
    medicine_directory = Path(MEDICINES_DIR).resolve()
    duplicate_candidates = find_duplicate_database_paths()
    stats = get_database_stats()
    readiness = get_readiness_stats()
    chunks = get_chunks()
    embedded_chunks = sum(bool(chunk.get("embedding")) for chunk in chunks)

    print(f"Database: {database_path}")
    print(f"Medicines folder: {medicine_directory}")
    print(f"JSON/CSV files: {len(discover_medicine_files())}")
    print(f"Medicines: {stats['medicine_count']}")
    print(f"READY medicines: {readiness['ready_medicine_count']}")
    print(f"Chunks: {stats['chunk_count']}")
    print(f"Embedded chunks: {embedded_chunks}")
    print(f"FTS rows: {readiness['fts_chunk_count']}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(
        "Duplicate databases: "
        + (", ".join(str(path) for path in duplicate_candidates) or "NONE")
    )

    if stats["medicine_count"] <= 0 or stats["chunk_count"] <= 0:
        raise SystemExit(
            "ERROR: Database is empty. Run `python -m src.ingestion` first."
        )
    if embedded_chunks != stats["chunk_count"]:
        raise SystemExit(
            "ERROR: Some chunks have no embedding. Run ingestion again."
        )
    if duplicate_candidates:
        raise SystemExit("ERROR: More than one medicines.db file was found.")

    if args.skip_retrieval:
        print("Retrieval: SKIPPED")
        return

    results = get_top_chunks(args.query, top_k=5)
    print(f"Retrieval results: {len(results)}")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['medicine_name']} / {result['chunk_type']} / "
            f"score={result['similarity_score']:.4f}"
        )
    if not results:
        raise SystemExit("ERROR: Retrieval returned no chunks.")


if __name__ == "__main__":
    main()
