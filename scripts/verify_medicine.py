"""Verify one medicine from SQLite through embedding retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATABASE_PATH,
    VERIFICATION_MEDICINE_NAME,
    VERIFICATION_QUERY,
)
from src.database import (
    find_duplicate_database_paths,
    find_medicines_by_name,
    get_chunks,
    get_database_stats,
)
from src.retrieval import get_top_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir ilacın kayıt, chunk, embedding ve retrieval zincirini doğrular."
    )
    parser.add_argument("medicine", nargs="?", default=VERIFICATION_MEDICINE_NAME)
    parser.add_argument("--query", default=VERIFICATION_QUERY)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Foundry embedding modelini çağırmadan yalnızca SQLite kaydını doğrular.",
    )
    args = parser.parse_args()

    stats = get_database_stats(args.database)
    print(f"Database: {Path(args.database).resolve()}")
    print(f"SELECT COUNT(*) FROM medicines: {stats['medicine_count']}")
    print(f"SELECT COUNT(*) FROM document_chunks: {stats['chunk_count']}")

    if Path(args.database).resolve() == Path(DATABASE_PATH).resolve():
        duplicates = find_duplicate_database_paths()
        print(
            "Başka medicines.db: "
            + (", ".join(str(path) for path in duplicates) or "YOK")
        )

    medicines = find_medicines_by_name(
        args.medicine,
        database_path=args.database,
    )
    if not medicines:
        raise SystemExit(f"Kayıt bulunamadı: {args.medicine}")

    print(f"Eşleşen ilaç sayısı: {len(medicines)}")
    for medicine in medicines:
        print(f"\nİlaç: {medicine['medicine_name']}")
        print(f"Etken madde: {medicine.get('active_ingredient') or '-'}")
        print(f"Kaynak: {medicine.get('source_name') or '-'}")
        print(f"Kaynak referansı: {medicine.get('source_reference') or '-'}")
        chunks = get_chunks(
            int(medicine["medicine_id"]),
            database_path=args.database,
        )
        print(f"Chunk sayısı: {len(chunks)}")
        for chunk in chunks:
            embedding = chunk.get("embedding")
            dimension = len(embedding) if isinstance(embedding, list) else 0
            print(
                f"  - #{chunk['chunk_id']} {chunk['chunk_type']} | "
                f"embedding={'VAR' if dimension else 'YOK'} | boyut={dimension}"
            )

    if args.skip_retrieval:
        print("\nRetrieval: ATLANDI")
        return

    print(f"\nRetrieval sorgusu: {args.query}")
    results = get_top_chunks(
        args.query,
        top_k=5,
        database_path=args.database,
    )
    if not results:
        raise SystemExit("Retrieval güvenilir bir chunk döndürmedi.")
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['medicine_name']} | {result['chunk_type']} | "
            f"similarity={result['similarity_score']:.4f}"
        )
        print(f"   {result['chunk_text']}")


if __name__ == "__main__":
    main()
