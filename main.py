"""Terminal status entry point for the Şifacı application."""

from config import (
    APP_TITLE,
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    FOUNDRY_MODEL_ALIAS,
    MEDICINE_DOCUMENTS_DIR,
)
from src.database import get_database_stats, initialize_database


def main() -> None:
    """Initialize SQLite and print the current local configuration."""

    initialize_database()
    stats = get_database_stats()
    print(APP_TITLE)
    print(f"JSON klasörü: {MEDICINE_DOCUMENTS_DIR}")
    print(f"Veritabanı: {DATABASE_PATH}")
    print(f"İlaç / chunk: {stats['medicine_count']} / {stats['chunk_count']}")
    print(f"Foundry Local chat modeli: {FOUNDRY_MODEL_ALIAS}")
    print(f"Foundry Local embedding modeli: {EMBEDDING_MODEL_NAME}")
    print("JSON aktarımı: python -m src.ingestion")
    print("Arayüz: streamlit run app.py")


if __name__ == "__main__":
    main()
