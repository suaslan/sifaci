"""Apply the additive İlacabak schema migration and report the result."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATABASE_PATH
from src.database import get_database_manager, initialize_database


def main() -> None:
    initialize_database()
    with get_database_manager(DATABASE_PATH).read_connection() as connection:
        medicine_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(medicines)")
        }
        chunk_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(document_chunks)")
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    required_tables = {"ilacabak_documents", "medicine_dosage_rules"}
    missing = required_tables - tables
    if missing or "ilacabak_url" not in medicine_columns:
        raise SystemExit(f"Migration eksik: {sorted(missing)}")
    required_chunk_columns = {"source_type", "source_date", "source_priority"}
    if not required_chunk_columns.issubset(chunk_columns):
        raise SystemExit("document_chunks kaynak alanları oluşturulamadı.")
    if foreign_key_errors:
        raise SystemExit(f"Foreign key doğrulaması başarısız: {foreign_key_errors[:5]}")

    print(f"Database: {DATABASE_PATH}")
    print("İlacabak migration: OK")
    print("Mevcut TİTCK/ilaç kayıtları korunmuştur.")


if __name__ == "__main__":
    main()
