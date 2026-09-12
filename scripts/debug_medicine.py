"""Display medicine, document, chunk and embedding diagnostics as tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._console_table import print_table
from src.database import (
    find_candidate_medicines,
    get_document_diagnostics,
    get_medicine_data_status,
    get_medicine_source_urls,
    get_medicines_by_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir ilacın katalog, KT/KÜB, chunk ve embedding durumunu gösterir."
    )
    parser.add_argument("medicine", help='Aranacak ilaç adı; örn. "MUSCOFLEX"')
    parser.add_argument("--limit", type=int, default=20, help="En fazla eşleşme sayısı.")
    parser.add_argument("--database", type=Path, help="Varsayılan yerine kullanılacak SQLite yolu.")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit pozitif olmalıdır")

    database_path = args.database
    matches = find_candidate_medicines(
        args.medicine,
        limit=args.limit,
        database_path=database_path,
    )
    medicine_ids = list(dict.fromkeys(int(item["medicine_id"]) for item in matches))

    print(f"Aranan ilaç: {args.medicine}")
    print_table(
        "1. İlaç eşleşmeleri",
        ("ID", "Kayıtlı ad", "Eşleşen alias", "Eşleşme", "Skor"),
        [
            (
                item["medicine_id"],
                item["medicine_name"],
                item.get("matched_alias") or item.get("alias"),
                item.get("match_type"),
                f"{float(item.get('name_score') or 0):.4f}",
            )
            for item in matches
        ],
    )
    if not medicine_ids:
        print("\nBu adla eşleşen medicines kaydı bulunamadı.")
        return

    medicines = get_medicines_by_ids(medicine_ids, database_path=database_path)
    for medicine in medicines:
        print_table(
            f"2. medicines kaydı — ID {medicine['medicine_id']}",
            ("Alan", "Değer"),
            [(field, value) for field, value in medicine.items()],
            max_cell_width=120,
        )

    status = get_medicine_data_status(medicine_ids, database_path=database_path)
    print_table(
        "3. KT/KÜB ve RAG özeti",
        ("Doküman", "KT", "KÜB", "Chunk", "Embedding", "Yan etki", "Doz"),
        [
            (
                status["documents_count"],
                status["kt_count"],
                status["kub_count"],
                status["chunks_count"],
                status["embeddings_count"],
                status["side_effect_chunks"],
                status["dosage_chunks"],
            )
        ],
    )

    documents = get_document_diagnostics(medicine_ids, database_path=database_path)
    print_table(
        "4. documents pipeline ve embedding durumu",
        (
            "Doc ID", "İlaç", "Tür", "Genel", "Download", "Parse",
            "Embedding", "Chunk", "Embedded", "Bekleyen", "Model", "Retry", "Hata",
        ),
        [
            (
                item["document_id"], item["medicine_name"], item["document_type"],
                item["status"], item["download_status"], item["parse_status"],
                item["embedding_status"], item["chunk_count"],
                item["embedded_chunk_count"], item["pending_embedding_count"],
                item.get("embedding_models"), item["retry_count"], item.get("last_error"),
            )
            for item in documents
        ],
        max_cell_width=48,
    )

    print_table(
        "5. Kaynak URL'leri",
        ("URL",),
        [(url,) for url in get_medicine_source_urls(medicine_ids, database_path=database_path)],
        max_cell_width=140,
    )


if __name__ == "__main__":
    main()
