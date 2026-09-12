from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from config import SQLITE_BUSY_TIMEOUT_MS, SQLITE_SYNCHRONOUS
from src.database import (
    get_database_manager,
    get_database_stats,
    get_chunks,
    initialize_database,
    insert_medicine,
    replace_chunks,
    save_chunk_embedding_batch,
    upsert_products,
)


def test_connection_manager_applies_performance_pragmas(tmp_path):
    database_path = tmp_path / "pragmas.db"
    initialize_database(database_path)
    manager = get_database_manager(database_path)

    with manager.read_connection() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        query_only = connection.execute("PRAGMA query_only").fetchone()[0]

    assert str(journal_mode).casefold() == "wal"
    assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS == 30_000
    assert synchronous == 1
    assert SQLITE_SYNCHRONOUS == "NORMAL"
    assert query_only == 1


def test_manager_is_singleton_and_serializes_threaded_writes(tmp_path):
    database_path = tmp_path / "threaded.db"
    initialize_database(database_path)

    assert get_database_manager(database_path) is get_database_manager(database_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        medicine_ids = list(
            executor.map(
                lambda index: insert_medicine(
                    f"Thread ilaci {index}", database_path=database_path
                ),
                range(100),
            )
        )

    assert len(set(medicine_ids)) == 100
    assert get_database_stats(database_path)["medicine_count"] == 100


def test_executemany_and_product_upsert_use_bounded_batches(tmp_path):
    database_path = tmp_path / "batches.db"
    initialize_database(database_path)
    manager = get_database_manager(database_path)

    with manager.write_connection() as connection:
        connection.execute(
            "CREATE TABLE batch_probe (probe_id INTEGER PRIMARY KEY, value TEXT)"
        )
    written = manager.executemany_batched(
        "INSERT INTO batch_probe (probe_id, value) VALUES (?, ?)",
        ((index, f"value-{index}") for index in range(250)),
        batch_size=100,
    )
    with manager.read_connection() as connection:
        count = connection.execute("SELECT COUNT(*) FROM batch_probe").fetchone()[0]

    products = [
        {"product_name": f"Batch urunu {index}", "source": "TITCK"}
        for index in range(120)
    ]
    results = upsert_products(products, batch_size=100, database_path=database_path)

    assert written == count == 250
    assert len(results) == 120
    assert {status for _, status in results} == {"inserted"}
    assert get_database_stats(database_path)["medicine_count"] == 120


def test_generated_embedding_batch_is_persisted_in_one_bulk_update(tmp_path):
    database_path = tmp_path / "embedding-batch.db"
    medicine_id = insert_medicine("Embedding batch ilaci", database_path=database_path)
    chunk_ids = replace_chunks(
        medicine_id,
        [
            {"chunk_text": f"chunk-{index}", "chunk_type": "test"}
            for index in range(32)
        ],
        database_path=database_path,
    )

    written = save_chunk_embedding_batch(
        [(chunk_id, [float(index), 1.0]) for index, chunk_id in enumerate(chunk_ids)],
        embedding_model="batch-model",
        database_path=database_path,
    )
    chunks = get_chunks(medicine_id, database_path=database_path)

    assert written == 32
    assert all(chunk["embedding_model"] == "batch-model" for chunk in chunks)
    assert all(chunk["embedding"] for chunk in chunks)
