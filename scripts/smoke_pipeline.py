"""Run the complete local JSON-to-RAG pipeline with temporary SQLite data."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from config import MEDICINE_DOCUMENTS_DIR
from src.database import get_database_stats
from src.ingestion import ingest_medicine_record, load_medicine_file
from src.rag import answer_query
from src.retrieval import get_top_chunks


def main() -> None:
    example_path = Path(MEDICINE_DOCUMENTS_DIR) / "example_medicine.json"
    record = dict(load_medicine_file(example_path)[0])
    record.pop("is_example", None)

    with TemporaryDirectory(prefix="sifaci-smoke-") as temporary_directory:
        database_path = Path(temporary_directory) / "smoke.db"
        medicine_id, chunk_count = ingest_medicine_record(
            record,
            database_path=database_path,
        )

        debug_trace: dict[str, object] = {}

        def retrieve(query: str, top_k: int = 5, debug_trace=None):
            return get_top_chunks(
                query,
                top_k=top_k,
                database_path=database_path,
                debug_trace=debug_trace,
            )

        answer = answer_query(
            "ÖRNEK İLAÇ ürününün yaygın yan etkileri nelerdir?",
            retrieval_function=retrieve,
            debug_trace=debug_trace,
        )
        stats = get_database_stats(database_path)

        print(f"İlaç ID: {medicine_id}")
        print(f"Oluşturulan chunk: {chunk_count}")
        print(f"SQLite ilaç/chunk: {stats['medicine_count']}/{stats['chunk_count']}")
        print(
            "Query embedding boyutu: "
            f"{debug_trace.get('query_embedding_dimension')}"
        )
        print(f"Top chunk sayısı: {len(debug_trace.get('top_chunks', []))}")
        print(f"Model çağrıldı: {debug_trace.get('model_called')}")
        print("\nRAG cevabı:\n")
        print(answer)


if __name__ == "__main__":
    main()
