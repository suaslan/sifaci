"""Report medicine/document/chunk/embedding coverage with percentages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._console_table import print_table
from src.database import get_data_coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="İlaç kataloğunun doküman, KT/KÜB, chunk ve embedding kapsamasını raporlar."
    )
    parser.add_argument("--missing-limit", type=int, default=20)
    parser.add_argument("--database", type=Path, help="Varsayılan yerine kullanılacak SQLite yolu.")
    args = parser.parse_args()
    if args.missing_limit < 0:
        parser.error("--missing-limit negatif olamaz")

    coverage = get_data_coverage(missing_limit=args.missing_limit, database_path=args.database)
    total = int(coverage["total_medicines"])
    print_table(
        "İlaç veri kapsama raporu",
        ("Metrik", "İlaç sayısı", "Toplama oranı"),
        [
            ("Toplam ilaç", total, "100.00%" if total else "0.00%"),
            ("Dokümanı olan ilaç", coverage["medicines_with_documents"], f"{coverage['documents_percentage']:.2f}%"),
            ("KT olan ilaç", coverage["medicines_with_kt"], f"{coverage['kt_percentage']:.2f}%"),
            ("KÜB olan ilaç", coverage["medicines_with_kub"], f"{coverage['kub_percentage']:.2f}%"),
            ("Chunk üretilmiş ilaç", coverage["medicines_with_chunks"], f"{coverage['chunks_percentage']:.2f}%"),
            ("Embedding üretilmiş ilaç", coverage["medicines_with_embeddings"], f"{coverage['embeddings_percentage']:.2f}%"),
        ],
    )
    print_table(
        "Ham veri adetleri",
        ("Doküman", "Chunk", "Embedding'li chunk"),
        [(coverage["total_documents"], coverage["total_chunks"], coverage["total_embedded_chunks"])],
    )
    print_table(
        f"İlk {args.missing_limit} embedding'i eksik ilaç",
        ("ID", "İlaç adı"),
        [(item["medicine_id"], item["medicine_name"]) for item in coverage["missing_examples"]],
        max_cell_width=100,
    )


if __name__ == "__main__":
    main()
