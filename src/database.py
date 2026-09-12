"""SQLite schema and data-access helpers for medicine documents.

Embeddings are stored as JSON arrays.  The public read functions deserialize
them back to Python lists so callers do not need to know about the storage
format.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import islice
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from config import (
    BASE_DIR,
    DATABASE_PATH,
    DB_WRITE_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_SYNCHRONOUS,
    SQLITE_WAL,
)
from src.medicine_names import generate_medicine_aliases, normalize_medicine_name


MEDICINE_FIELDS = (
    "medicine_name",
    "active_ingredient",
    "indications",
    "usage_information",
    "dosage_information",
    "frequency_information",
    "route_of_administration",
    "common_side_effects",
    "serious_side_effects",
    "warnings",
    "contraindications",
    "interactions",
    "source_name",
    "source_reference",
)

_INITIALIZED_DATABASES: set[Path] = set()
_INITIALIZE_LOCK = threading.Lock()
_MANAGERS: dict[Path, "SQLiteConnectionManager"] = {}
_MANAGERS_LOCK = threading.Lock()


def _database_path(database_path: str | Path | None = None) -> Path:
    """Resolve the configured database path and ensure its directory exists."""

    path = Path(database_path) if database_path is not None else Path(DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SQLiteConnectionManager:
    """Own SQLite connections and serialize every write transaction per database.

    Connections are intentionally short-lived and remain owned by the calling
    thread. WAL readers therefore continue in parallel, while the re-entrant
    writer lock guarantees one in-process writer at a time.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = _database_path(database_path).resolve()
        self._writer_lock = threading.RLock()

    def _open(self, *, read_only: bool) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(SQLITE_BUSY_TIMEOUT_MS)}")
        connection.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS}")
        connection.execute("PRAGMA foreign_keys = ON")
        if SQLITE_WAL:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            if journal_mode != "wal":
                with self._writer_lock:
                    connection.execute("PRAGMA journal_mode = WAL")
        if read_only:
            connection.execute("PRAGMA query_only = ON")
        return connection

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield an independent query-only connection for concurrent readers."""

        connection = self._open(read_only=True)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the process-wide single-writer transaction for this database."""

        with self._writer_lock:
            connection = self._open(read_only=False)
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def executemany_batched(
        self,
        statement: str,
        parameter_rows: Iterable[Sequence[Any]],
        *,
        batch_size: int = DB_WRITE_BATCH_SIZE,
    ) -> int:
        """Execute rows in bounded, independently committed transactions."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        written = 0
        iterator = iter(parameter_rows)
        while batch := list(islice(iterator, batch_size)):
            with self.write_connection() as connection:
                connection.executemany(statement, batch)
            written += len(batch)
        return written


def get_database_manager(
    database_path: str | Path | None = None,
) -> SQLiteConnectionManager:
    """Return the one thread-safe connection manager for a resolved DB path."""

    resolved_path = _database_path(database_path).resolve()
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(resolved_path)
        if manager is None:
            manager = SQLiteConnectionManager(resolved_path)
            _MANAGERS[resolved_path] = manager
        return manager


@contextmanager
def _connect(
    database_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Backward-compatible alias for the centralized write transaction."""

    with get_database_manager(database_path).write_connection() as connection:
        yield connection


@contextmanager
def _read_connect(
    database_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Yield a concurrent, query-only WAL connection."""

    with get_database_manager(database_path).read_connection() as connection:
        yield connection


def _executemany_in_chunks(
    connection: sqlite3.Connection,
    statement: str,
    parameter_rows: Iterable[Sequence[Any]],
    *,
    batch_size: int = DB_WRITE_BATCH_SIZE,
) -> int:
    """Use executemany in bounded groups inside the caller's transaction."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    written = 0
    iterator = iter(parameter_rows)
    while batch := list(islice(iterator, batch_size)):
        connection.executemany(statement, batch)
        written += len(batch)
    return written


@contextmanager
def _write_scope(
    database_path: str | Path | None,
    connection: sqlite3.Connection | None = None,
) -> Iterator[sqlite3.Connection]:
    """Reuse a caller's central transaction or open a new one."""

    if connection is not None:
        yield connection
        return
    with _connect(database_path) as managed_connection:
        yield managed_connection


def initialize_database(database_path: str | Path | None = None) -> None:
    """Create the medicine and document chunk tables when they do not exist."""

    resolved_path = _database_path(database_path).resolve()
    if resolved_path in _INITIALIZED_DATABASES and resolved_path.exists():
        return

    with _INITIALIZE_LOCK:
        if resolved_path in _INITIALIZED_DATABASES and resolved_path.exists():
            return
        _initialize_database_schema(resolved_path)
        _INITIALIZED_DATABASES.add(resolved_path)


def _initialize_database_schema(database_path: Path) -> None:
    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS medicines (
                medicine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_name TEXT NOT NULL COLLATE NOCASE,
                active_ingredient TEXT,
                indications TEXT,
                usage_information TEXT,
                dosage_information TEXT,
                frequency_information TEXT,
                route_of_administration TEXT,
                common_side_effects TEXT,
                serious_side_effects TEXT,
                warnings TEXT,
                contraindications TEXT,
                interactions TEXT,
                source_name TEXT,
                source_reference TEXT,
                normalized_name TEXT,
                company TEXT,
                license_number TEXT,
                license_date TEXT,
                barcode TEXT,
                pharmaceutical_form TEXT,
                strength TEXT,
                status TEXT,
                source TEXT,
                ilacabak_url TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (medicine_name, source_name, source_reference)
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                embedding TEXT,
                embedding_model TEXT,
                document_id INTEGER,
                section TEXT,
                chunk_hash TEXT,
                source_url TEXT,
                approval_date TEXT,
                source_type TEXT,
                source_date TEXT,
                source_priority INTEGER,
                embedding_status TEXT NOT NULL DEFAULT 'PENDING',
                FOREIGN KEY (medicine_id)
                    REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (document_id)
                    REFERENCES documents (document_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS medicine_aliases (
                alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                FOREIGN KEY (medicine_id)
                    REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE,
                UNIQUE (medicine_id, normalized_alias)
            );

            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                document_type TEXT NOT NULL CHECK (document_type IN ('KUB', 'KT')),
                document_url TEXT NOT NULL,
                local_path TEXT,
                approval_date TEXT,
                content_hash TEXT,
                parser_version TEXT,
                raw_text TEXT,
                downloaded_at TEXT,
                source TEXT NOT NULL DEFAULT 'TİTCK',
                download_status TEXT NOT NULL DEFAULT 'PENDING',
                parse_status TEXT NOT NULL DEFAULT 'PENDING',
                embedding_status TEXT NOT NULL DEFAULT 'PENDING',
                status TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (medicine_id)
                    REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE,
                UNIQUE (medicine_id, document_type, document_url)
            );

            CREATE TABLE IF NOT EXISTS sync_runs (
                sync_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                products_seen INTEGER NOT NULL DEFAULT 0,
                products_added INTEGER NOT NULL DEFAULT 0,
                products_updated INTEGER NOT NULL DEFAULT 0,
                documents_downloaded INTEGER NOT NULL DEFAULT 0,
                documents_updated INTEGER NOT NULL DEFAULT 0,
                chunks_created INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                last_checkpoint_at TEXT,
                downloaded INTEGER NOT NULL DEFAULT 0,
                parsed INTEGER NOT NULL DEFAULT 0,
                embedded INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running',
                error_log TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sync_run_id INTEGER,
                document_id INTEGER,
                stage TEXT NOT NULL,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                retryable INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,
                FOREIGN KEY (sync_run_id) REFERENCES sync_runs (sync_run_id),
                FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ilacabak_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL UNIQUE,
                product_url TEXT,
                prospectus_url TEXT,
                hkt_url TEXT,
                kub_url TEXT,
                hkt_source_type TEXT,
                kub_source_type TEXT,
                source_date TEXT,
                cache_path TEXT,
                content_hash TEXT,
                match_status TEXT NOT NULL DEFAULT 'PENDING',
                download_status TEXT NOT NULL DEFAULT 'PENDING',
                parse_status TEXT NOT NULL DEFAULT 'PENDING',
                embedding_status TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (medicine_id) REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS medicine_dosage_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                population TEXT,
                min_age REAL,
                max_age REAL,
                age_unit TEXT,
                dose_text TEXT,
                frequency_text TEXT,
                route_text TEXT,
                duration_text TEXT,
                instruction_text TEXT,
                source_type TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_date TEXT,
                raw_source_text TEXT NOT NULL,
                rule_hash TEXT NOT NULL UNIQUE,
                FOREIGN KEY (medicine_id) REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_medicines_name
                ON medicines (medicine_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_medicine_id
                ON document_chunks (medicine_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_type
                ON document_chunks (chunk_type);
            CREATE INDEX IF NOT EXISTS idx_aliases_normalized
                ON medicine_aliases (normalized_alias);
            CREATE INDEX IF NOT EXISTS idx_documents_medicine
                ON documents (medicine_id, document_type);
            """
        )
        chunk_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(document_chunks)")
        }
        for column, declaration in {
            "embedding_model": "TEXT",
            "document_id": "INTEGER",
            "section": "TEXT",
            "chunk_hash": "TEXT",
            "source_url": "TEXT",
            "approval_date": "TEXT",
            "embedding_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "source_type": "TEXT",
            "source_date": "TEXT",
            "source_priority": "INTEGER",
        }.items():
            if column not in chunk_columns:
                connection.execute(
                    f"ALTER TABLE document_chunks ADD COLUMN {column} {declaration}"
                )

        document_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(documents)")
        }
        for column, declaration in {
            "download_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "parse_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "embedding_status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "status": "TEXT NOT NULL DEFAULT 'PENDING'",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "last_error": "TEXT",
            "updated_at": "TEXT",
        }.items():
            if column not in document_columns:
                connection.execute(
                    f"ALTER TABLE documents ADD COLUMN {column} {declaration}"
                )

        sync_run_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sync_runs)")
        }
        for column, declaration in {
            "last_checkpoint_at": "TEXT",
            "downloaded": "INTEGER NOT NULL DEFAULT 0",
            "parsed": "INTEGER NOT NULL DEFAULT 0",
            "embedded": "INTEGER NOT NULL DEFAULT 0",
            "failed": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if column not in sync_run_columns:
                connection.execute(
                    f"ALTER TABLE sync_runs ADD COLUMN {column} {declaration}"
                )

        medicine_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(medicines)")
        }
        for column, declaration in {
            "normalized_name": "TEXT",
            "company": "TEXT",
            "license_number": "TEXT",
            "license_date": "TEXT",
            "barcode": "TEXT",
            "pharmaceutical_form": "TEXT",
            "strength": "TEXT",
            "status": "TEXT",
            "source": "TEXT",
            "ilacabak_url": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }.items():
            if column not in medicine_columns:
                connection.execute(
                    f"ALTER TABLE medicines ADD COLUMN {column} {declaration}"
                )
        ilacabak_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(ilacabak_documents)")
        }
        if "parser_version" not in ilacabak_columns:
            connection.execute(
                "ALTER TABLE ilacabak_documents ADD COLUMN parser_version TEXT"
            )
        connection.execute(
            "UPDATE medicines SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP), "
            "updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)"
        )
        for row in connection.execute(
            """
            SELECT m.medicine_id, m.medicine_name
            FROM medicines AS m
            WHERE m.normalized_name IS NULL
               OR NOT EXISTS (
                   SELECT 1 FROM medicine_aliases AS a
                   WHERE a.medicine_id = m.medicine_id
               )
            """
        ).fetchall():
            normalized_name = normalize_medicine_name(str(row["medicine_name"]))
            connection.execute(
                "UPDATE medicines SET normalized_name = COALESCE(normalized_name, ?) "
                "WHERE medicine_id = ?",
                (normalized_name, row["medicine_id"]),
            )
            _replace_aliases_in_connection(
                connection,
                int(row["medicine_id"]),
                str(row["medicine_name"]),
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document "
            "ON document_chunks (document_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_hash "
            "ON document_chunks (chunk_hash)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_status "
            "ON document_chunks (embedding_status, chunk_hash)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_pipeline_status "
            "ON documents (download_status, parse_status, embedding_status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status "
            "ON documents (status, document_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_errors_document "
            "ON sync_errors (document_id, stage, resolved_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ilacabak_pipeline "
            "ON ilacabak_documents (match_status, download_status, parse_status, embedding_status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_dosage_medicine_age "
            "ON medicine_dosage_rules (medicine_id, age_unit, min_age, max_age)"
        )
        connection.execute(
            "UPDATE document_chunks SET embedding_status = CASE "
            "WHEN embedding IS NOT NULL THEN 'DONE' ELSE 'PENDING' END "
            "WHERE embedding_status IS NULL OR embedding_status = 'PENDING'"
        )
        connection.execute(
            """
            UPDATE document_chunks
            SET source_type = COALESCE(
                    source_type,
                    CASE WHEN document_id IS NULL THEN 'USER' ELSE 'TITCK' END
                ),
                source_priority = COALESCE(
                    source_priority,
                    CASE
                        WHEN document_id IN (
                            SELECT document_id FROM documents WHERE document_type = 'KUB'
                        ) THEN 500
                        WHEN document_id IN (
                            SELECT document_id FROM documents WHERE document_type = 'KT'
                        ) THEN 450
                        ELSE 300
                    END
                ),
                source_date = COALESCE(source_date, approval_date)
            WHERE source_type IS NULL OR source_priority IS NULL
            """
        )
        connection.execute(
            """
            UPDATE documents
            SET download_status = CASE
                    WHEN content_hash IS NOT NULL AND local_path IS NOT NULL THEN 'DONE'
                    ELSE COALESCE(download_status, 'PENDING')
                END,
                parse_status = CASE
                    WHEN raw_text IS NOT NULL AND LENGTH(TRIM(raw_text)) > 0 THEN 'DONE'
                    ELSE COALESCE(parse_status, 'PENDING')
                END,
                embedding_status = CASE
                    WHEN EXISTS (
                        SELECT 1 FROM document_chunks AS dc
                        WHERE dc.document_id = documents.document_id
                    ) AND NOT EXISTS (
                        SELECT 1 FROM document_chunks AS dc
                        WHERE dc.document_id = documents.document_id
                          AND (dc.embedding IS NULL OR dc.embedding_status != 'DONE')
                    ) THEN 'DONE'
                    ELSE COALESCE(embedding_status, 'PENDING')
                END,
                retry_count = COALESCE(retry_count, 0),
                updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)
            """
        )
        connection.execute(
            """
            UPDATE documents
            SET status = CASE
                WHEN download_status = 'FAILED_PERMANENT'
                  OR parse_status = 'FAILED_PERMANENT'
                  OR embedding_status = 'FAILED_PERMANENT'
                    THEN 'FAILED_PERMANENT'
                WHEN download_status = 'DONE'
                  AND parse_status = 'DONE'
                  AND embedding_status = 'DONE'
                    THEN 'DONE'
                WHEN download_status = 'FAILED_RETRYABLE'
                  OR parse_status = 'FAILED_RETRYABLE'
                  OR embedding_status = 'FAILED_RETRYABLE'
                    THEN 'FAILED_RETRYABLE'
                ELSE 'PENDING'
            END
            """
        )


def insert_medicine(
    medicine: Mapping[str, Any] | str | None = None,
    *,
    database_path: str | Path | None = None,
    **medicine_fields: Any,
) -> int:
    """Insert a medicine and return its integer id.

    Data may be supplied as a mapping, keyword arguments, or a combination of
    both.  Re-importing the same name/source record updates it and returns the
    existing id.
    """

    medicine_id, _ = upsert_medicine(
        medicine,
        database_path=database_path,
        **medicine_fields,
    )
    return medicine_id


def upsert_medicine(
    medicine: Mapping[str, Any] | str | None = None,
    *,
    database_path: str | Path | None = None,
    **medicine_fields: Any,
) -> tuple[int, Literal["inserted", "updated", "duplicate"]]:
    """Insert or update one logical medicine record.

    Medicine names are treated as the stable logical key. The returned status
    lets bulk ingestion report whether a record was inserted, updated, or was
    already byte-for-byte current. All SQL values remain parameterized.
    """

    if isinstance(medicine, str):
        data: dict[str, Any] = {"medicine_name": medicine}
    else:
        data = dict(medicine or {})
    data.update(medicine_fields)
    unknown_fields = set(data) - set(MEDICINE_FIELDS) - {"medicine_id"}
    if unknown_fields:
        raise ValueError(f"Unsupported medicine fields: {sorted(unknown_fields)}")

    medicine_name = str(data.get("medicine_name") or "").strip()
    if not medicine_name:
        raise ValueError("medicine_name is required")

    values = {
        field: _to_optional_text(data.get(field)) for field in MEDICINE_FIELDS
    }
    values["medicine_name"] = medicine_name

    initialize_database(database_path)
    with _connect(database_path) as connection:
        existing = connection.execute(
            f"""
            SELECT medicine_id, {", ".join(MEDICINE_FIELDS)}
            FROM medicines
            WHERE medicine_name = ? COLLATE NOCASE
            ORDER BY medicine_id
            LIMIT 1
            """,
            (values["medicine_name"],),
        ).fetchone()

        if existing is not None:
            unchanged = all(existing[field] == values[field] for field in MEDICINE_FIELDS)
            if unchanged:
                _replace_aliases_in_connection(
                    connection,
                    int(existing["medicine_id"]),
                    values["medicine_name"],
                )
                return int(existing["medicine_id"]), "duplicate"

            assignments = ", ".join(f"{field} = ?" for field in MEDICINE_FIELDS)
            connection.execute(
                f"UPDATE medicines SET {assignments} WHERE medicine_id = ?",
                tuple(values[field] for field in MEDICINE_FIELDS)
                + (existing["medicine_id"],),
            )
            connection.execute(
                "UPDATE medicines SET normalized_name = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE medicine_id = ?",
                (
                    normalize_medicine_name(values["medicine_name"]),
                    existing["medicine_id"],
                ),
            )
            _replace_aliases_in_connection(
                connection,
                int(existing["medicine_id"]),
                values["medicine_name"],
            )
            return int(existing["medicine_id"]), "updated"

        placeholders = ", ".join("?" for _ in MEDICINE_FIELDS)
        columns = ", ".join(MEDICINE_FIELDS)
        cursor = connection.execute(
            f"INSERT INTO medicines ({columns}) VALUES ({placeholders})",
            tuple(values[field] for field in MEDICINE_FIELDS),
        )
        medicine_id = int(cursor.lastrowid)
        connection.execute(
            "UPDATE medicines SET normalized_name = ?, source = COALESCE(source, ?) "
            "WHERE medicine_id = ?",
            (normalize_medicine_name(values["medicine_name"]), values["source_name"], medicine_id),
        )
        _replace_aliases_in_connection(
            connection,
            medicine_id,
            values["medicine_name"],
        )
        return medicine_id, "inserted"


def insert_chunk(
    medicine_id: int,
    chunk_text: str,
    chunk_type: str,
    embedding: Sequence[float] | str | None = None,
    embedding_model: str | None = None,
    *,
    database_path: str | Path | None = None,
) -> int:
    """Insert a document chunk and return its integer id."""

    clean_text = str(chunk_text).strip()
    clean_type = str(chunk_type).strip()
    if not clean_text:
        raise ValueError("chunk_text is required")
    if not clean_type:
        raise ValueError("chunk_type is required")

    serialized_embedding = _serialize_embedding(embedding)
    serialized_model = (
        _to_optional_text(embedding_model) or EMBEDDING_MODEL_NAME
        if serialized_embedding is not None
        else None
    )
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO document_chunks
                (medicine_id, chunk_text, chunk_type, embedding, embedding_model,
                 embedding_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                medicine_id,
                clean_text,
                clean_type,
                serialized_embedding,
                serialized_model,
                "DONE" if serialized_embedding is not None else "PENDING",
            ),
        )
        return int(cursor.lastrowid)


def replace_chunks(
    medicine_id: int,
    chunks: Sequence[Mapping[str, Any]],
    *,
    database_path: str | Path | None = None,
) -> list[int]:
    """Atomically replace all chunks belonging to one medicine."""

    prepared: list[tuple[int, str, str, str | None, str | None, str]] = []
    for chunk in chunks:
        chunk_text = str(chunk.get("chunk_text") or "").strip()
        chunk_type = str(chunk.get("chunk_type") or "").strip()
        if not chunk_text or not chunk_type:
            raise ValueError("Each chunk requires chunk_text and chunk_type")
        serialized_embedding = _serialize_embedding(chunk.get("embedding"))
        embedding_model = (
            _to_optional_text(chunk.get("embedding_model"))
            or EMBEDDING_MODEL_NAME
            if serialized_embedding is not None
            else None
        )
        prepared.append(
            (
                medicine_id,
                chunk_text,
                chunk_type,
                serialized_embedding,
                embedding_model,
                "DONE" if serialized_embedding is not None else "PENDING",
            )
        )

    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE medicine_id = ?", (medicine_id,)
        )
        _executemany_in_chunks(
            connection,
            """
            INSERT INTO document_chunks
                (medicine_id, chunk_text, chunk_type, embedding, embedding_model,
                 embedding_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            prepared,
        )
        rows = connection.execute(
            "SELECT chunk_id FROM document_chunks WHERE medicine_id = ? "
            "ORDER BY chunk_id",
            (medicine_id,),
        ).fetchall()
        chunk_ids = [int(row["chunk_id"]) for row in rows]
    return chunk_ids


def save_chunk_embedding_batch(
    chunk_embeddings: Sequence[tuple[int, Sequence[float]]],
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    database_path: str | Path | None = None,
) -> int:
    """Persist one generated embedding batch in a single transaction."""

    if not chunk_embeddings:
        return 0
    prepared = [
        (
            _serialize_embedding(vector),
            embedding_model,
            int(chunk_id),
        )
        for chunk_id, vector in chunk_embeddings
    ]
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return _executemany_in_chunks(
            connection,
            """
            UPDATE document_chunks
            SET embedding = ?, embedding_model = ?, embedding_status = 'DONE'
            WHERE chunk_id = ?
            """,
            prepared,
            batch_size=len(prepared),
        )


def get_chunks(
    medicine_id: int | None = None,
    *,
    medicine_ids: Sequence[int] | None = None,
    chunk_type: str | None = None,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return chunks, optionally filtered by medicine and/or chunk category."""

    initialize_database(database_path)
    conditions: list[str] = []
    parameters: list[Any] = []

    if medicine_id is not None:
        conditions.append("dc.medicine_id = ?")
        parameters.append(medicine_id)
    if medicine_ids is not None:
        clean_ids = [int(value) for value in medicine_ids]
        if not clean_ids:
            return []
        conditions.append(
            f"dc.medicine_id IN ({', '.join('?' for _ in clean_ids)})"
        )
        parameters.extend(clean_ids)
    if chunk_type is not None:
        conditions.append("dc.chunk_type = ?")
        parameters.append(chunk_type)

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        "SELECT dc.chunk_id, dc.medicine_id, dc.document_id, dc.chunk_text, "
        "dc.chunk_type, dc.section, dc.chunk_hash, dc.source_url, "
        "dc.approval_date, dc.source_date, dc.embedding, dc.embedding_model, "
        "m.medicine_name, d.document_type, "
        "COALESCE(dc.source_type, CASE WHEN d.document_id IS NOT NULL "
        "THEN 'TITCK' ELSE 'USER' END) AS source_type, "
        "COALESCE(dc.source_priority, CASE d.document_type "
        "WHEN 'KUB' THEN 500 WHEN 'KT' THEN 450 ELSE 300 END) AS source_priority, "
        "CASE WHEN dc.source_type = 'ILACABAK' THEN 'İlacabak prospektüsü' "
        "WHEN dc.source_type = 'MANUFACTURER' THEN 'Üretici resmi belgesi' "
        "ELSE COALESCE(d.source || ' ' || d.document_type, m.source_name) END AS source_name, "
        "COALESCE(dc.source_url, m.source_reference) AS source_reference "
        "FROM document_chunks AS dc "
        "JOIN medicines AS m ON m.medicine_id = dc.medicine_id"
        " LEFT JOIN documents AS d ON d.document_id = dc.document_id"
        f"{where_clause} ORDER BY dc.chunk_id"
    )

    with _read_connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    chunks = [dict(row) for row in rows]
    for chunk in chunks:
        chunk["embedding"] = _deserialize_embedding(chunk["embedding"])
    return chunks


def get_document_chunks(
    document_id: int,
    *,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT dc.*, m.medicine_name,
                   d.document_type,
                   COALESCE(dc.source_type, CASE WHEN d.document_id IS NOT NULL
                       THEN 'TITCK' ELSE 'USER' END) AS resolved_source_type,
                   COALESCE(dc.source_priority, CASE d.document_type
                       WHEN 'KUB' THEN 500 WHEN 'KT' THEN 450 ELSE 300 END)
                       AS resolved_source_priority,
                   CASE WHEN dc.source_type = 'ILACABAK' THEN 'İlacabak prospektüsü'
                       WHEN dc.source_type = 'MANUFACTURER' THEN 'Üretici resmi belgesi'
                       ELSE COALESCE(d.source || ' ' || d.document_type, m.source_name)
                   END AS source_name,
                   COALESCE(dc.source_url, m.source_reference) AS source_reference
            FROM document_chunks AS dc
            JOIN medicines AS m ON m.medicine_id = dc.medicine_id
            LEFT JOIN documents AS d ON d.document_id = dc.document_id
            WHERE dc.document_id = ?
            ORDER BY dc.chunk_id
            """,
            (document_id,),
        ).fetchall()
    chunks = [dict(row) for row in rows]
    for chunk in chunks:
        chunk["embedding"] = _deserialize_embedding(chunk["embedding"])
    return chunks


def _replace_aliases_in_connection(
    connection: sqlite3.Connection,
    medicine_id: int,
    medicine_name: str,
) -> None:
    aliases = generate_medicine_aliases(medicine_name)
    _executemany_in_chunks(
        connection,
        """
        INSERT INTO medicine_aliases (medicine_id, alias, normalized_alias)
        VALUES (?, ?, ?)
        ON CONFLICT (medicine_id, normalized_alias)
        DO UPDATE SET alias = excluded.alias
        """,
        (
            (medicine_id, alias, normalize_medicine_name(alias))
            for alias in aliases
        ),
    )


def get_medicine_by_name(
    medicine_name: str,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find a medicine by exact name, case-insensitively."""

    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT medicine_id, medicine_name, active_ingredient, indications,
                   usage_information, dosage_information,
                   frequency_information, route_of_administration,
                   common_side_effects, serious_side_effects, warnings,
                   contraindications, interactions, source_name,
                   source_reference
            FROM medicines
            WHERE medicine_name = ? COLLATE NOCASE
            ORDER BY medicine_id
            LIMIT 1
            """,
            (medicine_name.strip(),),
        ).fetchone()
    return dict(row) if row is not None else None


def find_medicines_by_name(
    medicine_name: str,
    *,
    limit: int = 20,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Find medicine names containing a term, safely and case-insensitively."""

    term = medicine_name.strip()
    if not term:
        return []
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    # Escape SQL LIKE metacharacters so user input is treated as literal text.
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT medicine_id, {", ".join(MEDICINE_FIELDS)}
            FROM medicines
            WHERE medicine_name LIKE ? ESCAPE '\\' COLLATE NOCASE
            ORDER BY medicine_name, medicine_id
            LIMIT ?
            """,
            (f"%{escaped}%", limit),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_product(
    product: Mapping[str, Any],
    *,
    database_path: str | Path | None = None,
    _connection: sqlite3.Connection | None = None,
) -> tuple[int, Literal["inserted", "updated", "duplicate"]]:
    """Upsert one product from the TİTCK master list without inventing fields."""

    product_name = _to_optional_text(
        product.get("product_name") or product.get("medicine_name")
    )
    if not product_name:
        raise ValueError("product_name is required")
    normalized_name = normalize_medicine_name(product_name)
    product_columns = {
        "medicine_name": product_name,
        "normalized_name": normalized_name,
        "active_ingredient": _to_optional_text(product.get("active_ingredient")),
        "company": _to_optional_text(product.get("company")),
        "license_number": _to_optional_text(product.get("license_number")),
        "license_date": _to_optional_text(product.get("license_date")),
        "barcode": _to_optional_text(product.get("barcode")),
        "pharmaceutical_form": _to_optional_text(product.get("pharmaceutical_form")),
        "strength": _to_optional_text(product.get("strength")),
        "status": _to_optional_text(product.get("status")),
        "source": _to_optional_text(product.get("source")) or "TİTCK",
        "source_name": _to_optional_text(product.get("source_name"))
        or "TİTCK Ruhsatlı Beşeri Tıbbi Ürünler Listesi",
        "source_reference": _to_optional_text(product.get("source_reference")),
    }

    initialize_database(database_path)
    with _write_scope(database_path, _connection) as connection:
        existing = connection.execute(
            "SELECT * FROM medicines WHERE normalized_name = ? ORDER BY medicine_id LIMIT 1",
            (normalized_name,),
        ).fetchone()
        if existing is None:
            columns = tuple(product_columns)
            cursor = connection.execute(
                f"INSERT INTO medicines ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(product_columns[column] for column in columns),
            )
            medicine_id = int(cursor.lastrowid)
            status: Literal["inserted", "updated", "duplicate"] = "inserted"
        else:
            medicine_id = int(existing["medicine_id"])
            changed = any(
                existing[column] != value for column, value in product_columns.items()
            )
            if changed:
                connection.execute(
                    f"UPDATE medicines SET "
                    f"{', '.join(f'{column} = ?' for column in product_columns)}, "
                    "updated_at = CURRENT_TIMESTAMP WHERE medicine_id = ?",
                    tuple(product_columns.values()) + (medicine_id,),
                )
                status = "updated"
            else:
                status = "duplicate"
        _replace_aliases_in_connection(connection, medicine_id, product_name)
    return medicine_id, status


def upsert_products(
    products: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = DB_WRITE_BATCH_SIZE,
    database_path: str | Path | None = None,
) -> list[tuple[int, Literal["inserted", "updated", "duplicate"]]]:
    """Upsert products in bounded single-writer transactions."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    initialize_database(database_path)
    results: list[tuple[int, Literal["inserted", "updated", "duplicate"]]] = []
    iterator = iter(products)
    while batch := list(islice(iterator, batch_size)):
        with _connect(database_path) as connection:
            results.extend(
                upsert_product(
                    product,
                    database_path=database_path,
                    _connection=connection,
                )
                for product in batch
            )
    return results


def find_candidate_medicines(
    query: str,
    *,
    limit: int = 20,
    minimum_fuzzy_score: float = 0.72,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve a medicine before retrieval using deterministic staged matching.

    Match order is exact normalized alias, prefix, contains, then RapidFuzz.
    Returned rows include ``match_type`` and ``matched_alias`` for diagnostics.
    """

    from src.medicine_names import fuzzy_name_score

    normalized_query = normalize_medicine_name(query)
    if not normalized_query:
        return []
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        exact_rows = connection.execute(
            """
            SELECT DISTINCT m.medicine_id, m.medicine_name, a.alias,
                   a.normalized_alias
            FROM medicine_aliases AS a
            JOIN medicines AS m ON m.medicine_id = a.medicine_id
            WHERE (' ' || ? || ' ') LIKE ('% ' || a.normalized_alias || ' %')
            ORDER BY LENGTH(a.normalized_alias) DESC, m.medicine_name
            LIMIT ?
            """,
            (normalized_query, limit),
        ).fetchall()
        if exact_rows:
            longest = max(len(str(row["normalized_alias"])) for row in exact_rows)
            return [
                dict(row)
                | {
                    "name_score": 1.0,
                    "match_type": "exact_alias",
                    "matched_alias": str(row["alias"]),
                }
                for row in exact_rows
                if len(str(row["normalized_alias"])) == longest
            ][:limit]

        query_terms = _medicine_query_terms(normalized_query)
        if not query_terms:
            return []

        for match_type, condition, score in (
            ("prefix", "a.normalized_alias LIKE ? ESCAPE '\\'", 0.92),
            ("contains", "a.normalized_alias LIKE ? ESCAPE '\\'", 0.84),
        ):
            matched_rows: list[sqlite3.Row] = []
            for term in query_terms:
                escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                pattern = f"{escaped}%" if match_type == "prefix" else f"%{escaped}%"
                matched_rows.extend(
                    connection.execute(
                        f"""
                        SELECT m.medicine_id, m.medicine_name, a.alias,
                               a.normalized_alias
                        FROM medicine_aliases AS a
                        JOIN medicines AS m ON m.medicine_id = a.medicine_id
                        WHERE {condition}
                        ORDER BY LENGTH(a.normalized_alias), m.medicine_name
                        LIMIT ?
                        """,
                        (pattern, limit),
                    ).fetchall()
                )
            staged = _unique_medicine_matches(
                matched_rows,
                limit=limit,
                match_type=match_type,
                name_score=score,
            )
            if staged:
                return staged

        alias_rows = connection.execute(
            """
            SELECT m.medicine_id, m.medicine_name, a.alias, a.normalized_alias
            FROM medicine_aliases AS a
            JOIN medicines AS m ON m.medicine_id = a.medicine_id
            """
        ).fetchall()

    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in alias_rows:
        candidate = str(row["normalized_alias"])
        score = max(
            [fuzzy_name_score(normalized_query, candidate)]
            + [fuzzy_name_score(token, candidate) for token in query_terms]
        )
        if score >= minimum_fuzzy_score:
            ranked.append(
                (
                    score,
                    dict(row)
                    | {
                        "name_score": score,
                        "match_type": "rapidfuzz",
                        "matched_alias": str(row["alias"]),
                    },
                )
            )
    ranked.sort(key=lambda item: (item[0], len(item[1]["normalized_alias"])), reverse=True)
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for _, row in ranked:
        medicine_id = int(row["medicine_id"])
        if medicine_id not in seen:
            seen.add(medicine_id)
            unique.append(row)
        if len(unique) >= limit:
            break
    return unique


_MEDICINE_QUERY_STOP_WORDS = {
    "ac", "acik", "aktif", "agir", "alabilir", "alabilirim", "almalıyım",
    "ben", "bilgi", "bu", "ciddi", "cocuk", "defa", "doz", "etken",
    "etki", "etkiler", "etkileri", "etkisi", "gunde", "hangi", "hakkinda",
    "ilac", "ilaci", "ilacin", "ilacinin", "istenmeyen", "kac", "kez",
    "kimler", "kullanilir", "kullanim", "kullanma", "maddesi", "nasil",
    "ne", "nedir", "neler", "nelerdir", "siklik", "uyari", "uyarilar",
    "ve", "yan",
}


def _medicine_query_terms(normalized_query: str) -> list[str]:
    tokens = [
        token
        for token in normalized_query.split()
        if len(token) >= 3
        and not token.isdigit()
        and token not in _MEDICINE_QUERY_STOP_WORDS
    ]
    terms: list[str] = []
    for size in range(min(3, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - size + 1):
            term = " ".join(tokens[start : start + size])
            if term not in terms:
                terms.append(term)
    return terms


def _unique_medicine_matches(
    rows: Sequence[sqlite3.Row],
    *,
    limit: int,
    match_type: str,
    name_score: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        medicine_id = int(row["medicine_id"])
        if medicine_id in seen:
            continue
        seen.add(medicine_id)
        matches.append(
            dict(row)
            | {
                "name_score": name_score,
                "match_type": match_type,
                "matched_alias": str(row["alias"]),
            }
        )
        if len(matches) >= limit:
            break
    return matches


def upsert_document(
    medicine_id: int,
    document_type: str,
    document_url: str,
    *,
    local_path: str | None = None,
    approval_date: str | None = None,
    content_hash: str | None = None,
    raw_text: str | None = None,
    downloaded_at: str | None = None,
    source: str = "TİTCK",
    database_path: str | Path | None = None,
) -> tuple[int, Literal["inserted", "updated", "duplicate"]]:
    """Upsert one KÜB/KT document for a product."""

    normalized_type = document_type.strip().upper()
    if normalized_type not in {"KUB", "KT"}:
        raise ValueError("document_type must be KUB or KT")
    clean_url = document_url.strip()
    if not clean_url:
        raise ValueError("document_url is required")
    values = {
        "document_url": clean_url,
        "local_path": _to_optional_text(local_path),
        "approval_date": _to_optional_text(approval_date),
        "content_hash": _to_optional_text(content_hash),
        "raw_text": _to_optional_text(raw_text),
        "downloaded_at": _to_optional_text(downloaded_at),
        "source": _to_optional_text(source) or "TİTCK",
    }
    initialize_database(database_path)
    with _connect(database_path) as connection:
        existing = connection.execute(
            """
            SELECT * FROM documents
            WHERE medicine_id = ? AND document_type = ?
            ORDER BY document_id LIMIT 1
            """,
            (medicine_id, normalized_type),
        ).fetchone()
        if existing is None:
            columns = ("medicine_id", "document_type", *values.keys())
            cursor = connection.execute(
                f"INSERT INTO documents ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                (medicine_id, normalized_type, *values.values()),
            )
            return int(cursor.lastrowid), "inserted"
        document_id = int(existing["document_id"])
        if all(existing[column] == value for column, value in values.items()):
            return document_id, "duplicate"
        connection.execute(
            f"UPDATE documents SET {', '.join(f'{column} = ?' for column in values)} "
            "WHERE document_id = ?",
            tuple(values.values()) + (document_id,),
        )
        return document_id, "updated"


def get_document(
    medicine_id: int,
    document_type: str,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE medicine_id = ? AND document_type = ? "
            "ORDER BY document_id LIMIT 1",
            (medicine_id, document_type.strip().upper()),
        ).fetchone()
    return dict(row) if row is not None else None


def get_documents(
    medicine_id: int | None = None,
    *,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    query = "SELECT * FROM documents"
    parameters: tuple[Any, ...] = ()
    if medicine_id is not None:
        query += " WHERE medicine_id = ?"
        parameters = (medicine_id,)
    query += " ORDER BY document_id"
    with _read_connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def register_document_catalog(
    documents: Sequence[Mapping[str, Any]],
    *,
    database_path: str | Path | None = None,
) -> dict[str, int]:
    """Upsert discovered document URLs in one transaction without resetting DONE work."""

    initialize_database(database_path)
    inserted = 0
    updated = 0
    duplicates = 0
    with _connect(database_path) as connection:
        for item in documents:
            medicine_id = int(item["medicine_id"])
            document_type = str(item["document_type"]).strip().upper()
            document_url = str(item["document_url"]).strip()
            if document_type not in {"KUB", "KT"} or not document_url:
                raise ValueError("Each catalog document requires medicine_id, KUB/KT and URL")
            approval_date = _to_optional_text(item.get("approval_date"))
            source = _to_optional_text(item.get("source")) or "TİTCK"
            existing = connection.execute(
                "SELECT document_id, document_url, approval_date, source FROM documents "
                "WHERE medicine_id = ? AND document_type = ? ORDER BY document_id LIMIT 1",
                (medicine_id, document_type),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO documents (
                        medicine_id, document_type, document_url, approval_date, source,
                        download_status, parse_status, embedding_status, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'PENDING', 'PENDING', 'PENDING', 'PENDING', CURRENT_TIMESTAMP)
                    """,
                    (medicine_id, document_type, document_url, approval_date, source),
                )
                inserted += 1
                continue
            changed = (
                existing["document_url"] != document_url
                or existing["approval_date"] != approval_date
                or existing["source"] != source
            )
            if not changed:
                duplicates += 1
                continue
            url_changed = existing["document_url"] != document_url
            connection.execute(
                """
                UPDATE documents
                SET document_url = ?, approval_date = ?, source = ?,
                    download_status = CASE WHEN ? THEN 'PENDING' ELSE download_status END,
                    parse_status = CASE WHEN ? THEN 'PENDING' ELSE parse_status END,
                    embedding_status = CASE WHEN ? THEN 'PENDING' ELSE embedding_status END,
                    status = CASE WHEN ? THEN 'PENDING' ELSE status END,
                    last_error = CASE WHEN ? THEN NULL ELSE last_error END,
                    retry_count = CASE WHEN ? THEN 0 ELSE retry_count END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (
                    document_url,
                    approval_date,
                    source,
                    url_changed,
                    url_changed,
                    url_changed,
                    url_changed,
                    url_changed,
                    url_changed,
                    existing["document_id"],
                ),
            )
            updated += 1
    return {"inserted": inserted, "updated": updated, "duplicates": duplicates}


def get_sync_state(
    state_key: str,
    *,
    database_path: str | Path | None = None,
) -> str | None:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            "SELECT state_value FROM sync_state WHERE state_key = ?", (state_key,)
        ).fetchone()
    return str(row["state_value"]) if row is not None else None


def set_sync_state(
    state_key: str,
    state_value: str,
    *,
    database_path: str | Path | None = None,
) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sync_state (state_key, state_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(state_key) DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (state_key.strip(), str(state_value)),
        )


def iter_ilacabak_candidates(
    *,
    update: bool = False,
    limit: int | None = None,
    medicine_query: str | None = None,
    database_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream medicines whose İlacabak provider stages are incomplete."""

    initialize_database(database_path)
    query = """
        SELECT m.medicine_id, m.medicine_name, m.barcode, m.company,
               m.active_ingredient, m.ilacabak_url
        FROM medicines AS m
        LEFT JOIN ilacabak_documents AS i ON i.medicine_id = m.medicine_id
    """
    conditions: list[str] = []
    parameters: list[Any] = []
    if not update:
        conditions.append("""(
        i.id IS NULL OR (
            i.match_status != 'FAILED_PERMANENT'
            AND i.download_status != 'FAILED_PERMANENT'
            AND i.parse_status != 'FAILED_PERMANENT'
            AND i.embedding_status != 'FAILED_PERMANENT'
            AND (
                i.match_status != 'DONE'
                OR i.download_status != 'DONE'
                OR i.parse_status != 'DONE'
                OR i.embedding_status != 'DONE'
            )
        )
        )""")
    if medicine_query and medicine_query.strip():
        escaped = (
            medicine_query.strip()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        conditions.append("m.medicine_name LIKE ? ESCAPE '\\' COLLATE NOCASE")
        parameters.append(f"%{escaped}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY m.medicine_id"
    if limit is not None:
        if limit <= 0:
            return
        query += " LIMIT ?"
        parameters.append(int(limit))
    with _read_connect(database_path) as connection:
        cursor = connection.execute(query, parameters)
        while rows := cursor.fetchmany(100):
            for row in rows:
                yield dict(row)


def mark_ilacabak_in_progress(
    medicine_id: int,
    *,
    database_path: str | Path | None = None,
) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ilacabak_documents (medicine_id, match_status, updated_at)
            VALUES (?, 'IN_PROGRESS', CURRENT_TIMESTAMP)
            ON CONFLICT(medicine_id) DO UPDATE SET
                match_status = 'IN_PROGRESS', last_error = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (medicine_id,),
        )


def save_ilacabak_failure(
    medicine_id: int,
    stage: str,
    message: str,
    *,
    retryable: bool,
    attempts: int = 0,
    database_path: str | Path | None = None,
) -> None:
    stage_columns = {
        "match": "match_status",
        "download": "download_status",
        "parse": "parse_status",
        "embedding": "embedding_status",
    }
    if stage not in stage_columns:
        raise ValueError("Unsupported İlacabak stage")
    status = "FAILED_RETRYABLE" if retryable else "FAILED_PERMANENT"
    column = stage_columns[stage]
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "INSERT INTO ilacabak_documents (medicine_id) VALUES (?) "
            "ON CONFLICT(medicine_id) DO NOTHING",
            (medicine_id,),
        )
        connection.execute(
            f"UPDATE ilacabak_documents SET {column} = ?, retry_count = retry_count + ?, "
            "last_error = ?, updated_at = CURRENT_TIMESTAMP WHERE medicine_id = ?",
            (status, attempts, message[:4000], medicine_id),
        )


def save_ilacabak_document(
    medicine_id: int,
    metadata: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    dosage_rules: Sequence[Mapping[str, Any]],
    *,
    database_path: str | Path | None = None,
    _connection: sqlite3.Connection | None = None,
) -> None:
    """Atomically replace only İlacabak-derived content for one medicine."""

    product_url = str(metadata["product_url"])
    prospectus_url = str(metadata["prospectus_url"])
    initialize_database(database_path)
    with _write_scope(database_path, _connection) as connection:
        connection.execute(
            "UPDATE medicines SET ilacabak_url = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE medicine_id = ?",
            (product_url, medicine_id),
        )
        connection.execute(
            """
            INSERT INTO ilacabak_documents (
                medicine_id, product_url, prospectus_url, hkt_url, kub_url,
                hkt_source_type, kub_source_type, source_date, cache_path,
                content_hash, parser_version, match_status, download_status, parse_status,
                embedding_status, retry_count, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DONE', 'DONE', 'DONE',
                      'DONE', ?, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(medicine_id) DO UPDATE SET
                product_url = excluded.product_url,
                prospectus_url = excluded.prospectus_url,
                hkt_url = excluded.hkt_url,
                kub_url = excluded.kub_url,
                hkt_source_type = excluded.hkt_source_type,
                kub_source_type = excluded.kub_source_type,
                source_date = excluded.source_date,
                cache_path = excluded.cache_path,
                content_hash = excluded.content_hash,
                parser_version = excluded.parser_version,
                match_status = 'DONE', download_status = 'DONE',
                parse_status = 'DONE', embedding_status = 'DONE',
                retry_count = excluded.retry_count, last_error = NULL,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                medicine_id,
                product_url,
                prospectus_url,
                _to_optional_text(metadata.get("hkt_url")),
                _to_optional_text(metadata.get("kub_url")),
                _to_optional_text(metadata.get("hkt_source_type")),
                _to_optional_text(metadata.get("kub_source_type")),
                _to_optional_text(metadata.get("source_date")),
                _to_optional_text(metadata.get("cache_path")),
                _to_optional_text(metadata.get("content_hash")),
                _to_optional_text(metadata.get("parser_version")),
                int(metadata.get("retry_count") or 0),
            ),
        )
        connection.execute(
            "DELETE FROM document_chunks WHERE medicine_id = ? AND source_type = 'ILACABAK'",
            (medicine_id,),
        )
        chunk_values = [
            (
                medicine_id,
                str(chunk["chunk_text"]).strip(),
                str(chunk["chunk_type"]),
                _to_optional_text(chunk.get("section")),
                _serialize_embedding(chunk.get("embedding")),
                _to_optional_text(chunk.get("embedding_model")),
                _to_optional_text(chunk.get("chunk_hash")),
                prospectus_url,
                _to_optional_text(metadata.get("source_date")),
                "ILACABAK",
                _to_optional_text(metadata.get("source_date")),
                200,
                "DONE" if chunk.get("embedding") is not None else "PENDING",
            )
            for chunk in chunks
        ]
        _executemany_in_chunks(
            connection,
            """
            INSERT INTO document_chunks (
                medicine_id, chunk_text, chunk_type, section, embedding,
                embedding_model, chunk_hash, source_url, approval_date,
                source_type, source_date, source_priority, embedding_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_values,
        )
        connection.execute(
            "DELETE FROM medicine_dosage_rules "
            "WHERE medicine_id = ? AND source_type = 'ILACABAK'",
            (medicine_id,),
        )
        rule_values = [
            (
                medicine_id,
                _to_optional_text(rule.get("population")),
                rule.get("min_age"),
                rule.get("max_age"),
                _to_optional_text(rule.get("age_unit")),
                _to_optional_text(rule.get("dose_text")),
                _to_optional_text(rule.get("frequency_text")),
                _to_optional_text(rule.get("route_text")),
                _to_optional_text(rule.get("duration_text")),
                _to_optional_text(rule.get("instruction_text")),
                "ILACABAK",
                prospectus_url,
                _to_optional_text(metadata.get("source_date")),
                str(rule["raw_source_text"]),
                str(rule["rule_hash"]),
            )
            for rule in dosage_rules
        ]
        _executemany_in_chunks(
            connection,
            """
            INSERT OR REPLACE INTO medicine_dosage_rules (
                medicine_id, population, min_age, max_age, age_unit,
                dose_text, frequency_text, route_text, duration_text,
                instruction_text, source_type, source_url, source_date,
                raw_source_text, rule_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rule_values,
        )


def save_ilacabak_documents_batch(
    documents: Sequence[Mapping[str, Any]],
    *,
    database_path: str | Path | None = None,
) -> int:
    """Persist one embedding batch of İlacabak documents atomically."""

    if not documents:
        return 0
    initialize_database(database_path)
    with _connect(database_path) as connection:
        for document in documents:
            save_ilacabak_document(
                int(document["medicine_id"]),
                document["metadata"],
                document["chunks"],
                document["dosage_rules"],
                database_path=database_path,
                _connection=connection,
            )
    return len(documents)


def refresh_ilacabak_metadata(
    medicine_id: int,
    metadata: Mapping[str, Any],
    *,
    database_path: str | Path | None = None,
) -> None:
    """Refresh link/date/cache metadata without rebuilding unchanged embeddings."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "UPDATE medicines SET ilacabak_url = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE medicine_id = ?",
            (str(metadata["product_url"]), medicine_id),
        )
        connection.execute(
            """
            UPDATE ilacabak_documents
            SET product_url = ?, prospectus_url = ?, hkt_url = ?, kub_url = ?,
                hkt_source_type = ?, kub_source_type = ?, source_date = ?,
                cache_path = ?, parser_version = ?, retry_count = ?, last_error = NULL,
                match_status = 'DONE', download_status = 'DONE',
                parse_status = 'DONE', embedding_status = 'DONE',
                updated_at = CURRENT_TIMESTAMP
            WHERE medicine_id = ?
            """,
            (
                str(metadata["product_url"]),
                str(metadata["prospectus_url"]),
                _to_optional_text(metadata.get("hkt_url")),
                _to_optional_text(metadata.get("kub_url")),
                _to_optional_text(metadata.get("hkt_source_type")),
                _to_optional_text(metadata.get("kub_source_type")),
                _to_optional_text(metadata.get("source_date")),
                _to_optional_text(metadata.get("cache_path")),
                _to_optional_text(metadata.get("parser_version")),
                int(metadata.get("retry_count") or 0),
                medicine_id,
            ),
        )


def save_ilacabak_no_content(
    medicine_id: int,
    metadata: Mapping[str, Any],
    message: str,
    *,
    database_path: str | Path | None = None,
) -> None:
    """Persist a valid match whose accessible HTML has no prospectus sections."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "UPDATE medicines SET ilacabak_url = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE medicine_id = ?",
            (str(metadata["product_url"]), medicine_id),
        )
        connection.execute(
            """
            INSERT INTO ilacabak_documents (
                medicine_id, product_url, prospectus_url, hkt_url, kub_url,
                hkt_source_type, kub_source_type, source_date, cache_path,
                content_hash, parser_version, match_status, download_status, parse_status,
                embedding_status, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DONE', 'DONE',
                      'FAILED_PERMANENT', 'FAILED_PERMANENT', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(medicine_id) DO UPDATE SET
                product_url = excluded.product_url,
                prospectus_url = excluded.prospectus_url,
                hkt_url = excluded.hkt_url, kub_url = excluded.kub_url,
                hkt_source_type = excluded.hkt_source_type,
                kub_source_type = excluded.kub_source_type,
                source_date = excluded.source_date, cache_path = excluded.cache_path,
                content_hash = excluded.content_hash, match_status = 'DONE',
                parser_version = excluded.parser_version,
                download_status = 'DONE', parse_status = 'FAILED_PERMANENT',
                embedding_status = 'FAILED_PERMANENT',
                last_error = excluded.last_error, updated_at = CURRENT_TIMESTAMP
            """,
            (
                medicine_id,
                str(metadata["product_url"]),
                str(metadata["prospectus_url"]),
                _to_optional_text(metadata.get("hkt_url")),
                _to_optional_text(metadata.get("kub_url")),
                _to_optional_text(metadata.get("hkt_source_type")),
                _to_optional_text(metadata.get("kub_source_type")),
                _to_optional_text(metadata.get("source_date")),
                _to_optional_text(metadata.get("cache_path")),
                _to_optional_text(metadata.get("content_hash")),
                _to_optional_text(metadata.get("parser_version")),
                message[:4000],
            ),
        )


def get_ilacabak_document(
    medicine_id: int,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM ilacabak_documents WHERE medicine_id = ?", (medicine_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def get_dosage_rules(
    medicine_ids: Sequence[int],
    *,
    age: float | None = None,
    age_unit: str = "year",
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    clean_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
    if not clean_ids:
        return []
    conditions = [f"medicine_id IN ({', '.join('?' for _ in clean_ids)})"]
    parameters: list[Any] = list(clean_ids)
    if age is not None:
        conditions.extend(
            [
                "age_unit = ?",
                "(min_age IS NULL OR min_age <= ?)",
                "(max_age IS NULL OR max_age >= ?)",
                "(min_age IS NOT NULL OR max_age IS NOT NULL)",
            ]
        )
        parameters.extend((age_unit, age, age))
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM medicine_dosage_rules WHERE "
            + " AND ".join(conditions)
            + " ORDER BY CASE source_type WHEN 'TITCK' THEN 3 "
              "WHEN 'MANUFACTURER' THEN 2 ELSE 1 END DESC, id",
            parameters,
        ).fetchall()
    return [dict(row) for row in rows]


def iter_pipeline_documents(
    *,
    limit: int | None = None,
    medicine_ids: Sequence[int] | None = None,
    resume: bool = False,
    database_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream actionable documents without loading their raw text into RAM.

    Resume mode intentionally admits only durable PENDING and FAILED_RETRYABLE
    records. FAILED_PERMANENT and DONE records therefore cannot loop forever.
    """

    initialize_database(database_path)
    query = """
        SELECT d.document_id, d.medicine_id, m.medicine_name, d.document_type,
               d.document_url, d.local_path, d.approval_date, d.content_hash,
               d.download_status, d.parse_status, d.embedding_status, d.status,
               d.retry_count, d.last_error
        FROM documents AS d
        JOIN medicines AS m ON m.medicine_id = d.medicine_id
        WHERE (d.download_status != 'DONE'
            OR d.parse_status != 'DONE'
            OR d.embedding_status != 'DONE')
          AND d.download_status != 'FAILED_PERMANENT'
          AND d.parse_status != 'FAILED_PERMANENT'
          AND d.embedding_status != 'FAILED_PERMANENT'
    """
    parameters: list[Any] = []
    if resume:
        query += " AND d.status IN ('PENDING', 'FAILED_RETRYABLE')"
    if medicine_ids is not None:
        selected_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
        if not selected_ids:
            return
        placeholders = ",".join("?" for _ in selected_ids)
        query += f" AND d.medicine_id IN ({placeholders})"
        parameters.extend(selected_ids)
    query += " ORDER BY CASE d.document_type WHEN 'KT' THEN 0 ELSE 1 END, d.document_id"
    if limit is not None:
        if limit <= 0:
            return
        query += " LIMIT ?"
        parameters.append(int(limit))
    with _read_connect(database_path) as connection:
        cursor = connection.execute(query, parameters)
        while rows := cursor.fetchmany(100):
            for row in rows:
                yield dict(row)


def recover_interrupted_documents(
    *,
    database_path: str | Path | None = None,
) -> int:
    """Return checkpointed IN_PROGRESS work to the retryable queue.

    A suspended process normally continues by itself. If the process or machine
    stopped completely, its last committed stage marker can remain IN_PROGRESS;
    the next --resume run converts only those stale markers atomically.
    """

    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET download_status = CASE
                    WHEN download_status = 'IN_PROGRESS'
                    THEN 'FAILED_RETRYABLE' ELSE download_status END,
                parse_status = CASE
                    WHEN parse_status = 'IN_PROGRESS'
                    THEN 'FAILED_RETRYABLE' ELSE parse_status END,
                embedding_status = CASE
                    WHEN embedding_status = 'IN_PROGRESS'
                    THEN 'FAILED_RETRYABLE' ELSE embedding_status END,
                status = 'FAILED_RETRYABLE',
                last_error = COALESCE(
                    last_error,
                    'Önceki süreç kesildi; checkpoint kaydı yeniden kuyruğa alındı.'
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE status != 'FAILED_PERMANENT'
              AND (download_status = 'IN_PROGRESS'
                OR parse_status = 'IN_PROGRESS'
                OR embedding_status = 'IN_PROGRESS')
            """
        )
        return int(cursor.rowcount)


def requeue_ocr_documents(
    *,
    database_path: str | Path | None = None,
) -> int:
    """Retry image-only PDFs that were classified before OCR was available."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
        predicate = (
            "parse_status = 'FAILED_PERMANENT' "
            "AND last_error LIKE '%OCR gerekebilir%'"
        )
        connection.execute(
            f"""
            UPDATE sync_errors
            SET resolved_at = CURRENT_TIMESTAMP
            WHERE resolved_at IS NULL
              AND stage = 'parse'
              AND document_id IN (
                  SELECT document_id FROM documents WHERE {predicate}
              )
            """
        )
        cursor = connection.execute(
            f"""
            UPDATE documents
            SET parse_status = 'FAILED_RETRYABLE',
                embedding_status = CASE
                    WHEN embedding_status = 'FAILED_PERMANENT' THEN 'PENDING'
                    ELSE embedding_status
                END,
                status = 'FAILED_RETRYABLE',
                last_error = 'OCR fallback etkinleştirildi; belge yeniden kuyruğa alındı.',
                updated_at = CURRENT_TIMESTAMP
            WHERE {predicate}
            """
        )
        return int(cursor.rowcount)


def get_document_raw_text(
    document_id: int,
    *,
    database_path: str | Path | None = None,
) -> str:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            "SELECT raw_text FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
    return str(row["raw_text"] or "") if row is not None else ""


def get_embeddings_by_hashes(
    chunk_hashes: Sequence[str],
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    database_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """Return reusable embeddings for identical chunks from the current model."""

    hashes = list(dict.fromkeys(value for value in chunk_hashes if value))
    if not hashes:
        return {}
    initialize_database(database_path)
    result: dict[str, list[float]] = {}
    with _read_connect(database_path) as connection:
        for offset in range(0, len(hashes), 500):
            batch = hashes[offset : offset + 500]
            rows = connection.execute(
                "SELECT chunk_hash, embedding FROM document_chunks "
                f"WHERE chunk_hash IN ({', '.join('?' for _ in batch)}) "
                "AND embedding_status = 'DONE' AND embedding_model = ? "
                "AND embedding IS NOT NULL",
                (*batch, embedding_model),
            ).fetchall()
            for row in rows:
                vector = _deserialize_embedding(row["embedding"])
                if vector is not None:
                    result[str(row["chunk_hash"])] = vector
    return result


def apply_pipeline_events(
    events: Sequence[Mapping[str, Any]],
    *,
    sync_run_id: int,
    database_path: str | Path | None = None,
) -> None:
    """Apply worker results through one batched SQLite transaction."""

    if not events:
        return
    stage_columns = {
        "download": "download_status",
        "parse": "parse_status",
        "embedding": "embedding_status",
    }
    initialize_database(database_path)
    with _connect(database_path) as connection:
        for event in events:
            kind = str(event["kind"])
            document_id = int(event.get("document_id") or 0)
            if kind == "stage_status":
                stage = str(event["stage"])
                column = stage_columns[stage]
                status = str(event["status"])
                connection.execute(
                    f"UPDATE documents SET {column} = ?, status = 'PENDING', "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE document_id = ?",
                    (status, document_id),
                )
            elif kind == "download_done":
                content_hash = str(event["content_hash"])
                connection.execute(
                    """
                    UPDATE documents
                    SET parse_status = CASE
                            WHEN content_hash IS NOT NULL AND content_hash != ?
                            THEN 'PENDING' ELSE parse_status END,
                        embedding_status = CASE
                            WHEN content_hash IS NOT NULL AND content_hash != ?
                            THEN 'PENDING' ELSE embedding_status END,
                        local_path = ?, content_hash = ?, downloaded_at = ?,
                        download_status = 'DONE', retry_count = retry_count + ?,
                        status = 'PENDING', last_error = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                    """,
                    (
                        content_hash,
                        content_hash,
                        str(event["local_path"]),
                        content_hash,
                        str(event["downloaded_at"]),
                        int(event.get("retry_count") or 0),
                        document_id,
                    ),
                )
                connection.execute(
                    "UPDATE sync_errors SET resolved_at = CURRENT_TIMESTAMP "
                    "WHERE document_id = ? AND stage = 'download' AND resolved_at IS NULL",
                    (document_id,),
                )
            elif kind == "parse_done":
                connection.execute(
                    "UPDATE documents SET raw_text = ?, parse_status = 'DONE', "
                    "status = 'PENDING', last_error = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE document_id = ?",
                    (str(event["raw_text"]), document_id),
                )
                connection.execute(
                    "UPDATE sync_errors SET resolved_at = CURRENT_TIMESTAMP "
                    "WHERE document_id = ? AND stage = 'parse' AND resolved_at IS NULL",
                    (document_id,),
                )
            elif kind == "chunks_done":
                chunks = list(event["chunks"])
                connection.execute(
                    "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
                )
                values = [
                    (
                        document_id,
                        int(event["medicine_id"]),
                        str(chunk["chunk_text"]).strip(),
                        str(chunk.get("chunk_type") or chunk.get("section") or "general"),
                        _to_optional_text(chunk.get("section")),
                        _serialize_embedding(chunk.get("embedding")),
                        _to_optional_text(chunk.get("embedding_model")),
                        _to_optional_text(chunk.get("chunk_hash")),
                        _to_optional_text(chunk.get("source_url")),
                        _to_optional_text(chunk.get("approval_date")),
                        _to_optional_text(chunk.get("source_type")),
                        _to_optional_text(chunk.get("source_date")),
                        int(chunk.get("source_priority") or 300),
                        "DONE" if chunk.get("embedding") is not None else "PENDING",
                    )
                    for chunk in chunks
                ]
                _executemany_in_chunks(
                    connection,
                    """
                    INSERT INTO document_chunks (
                        document_id, medicine_id, chunk_text, chunk_type, section,
                        embedding, embedding_model, chunk_hash, source_url,
                        approval_date, source_type, source_date, source_priority,
                        embedding_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                connection.execute(
                    "UPDATE documents SET embedding_status = 'DONE', status = 'DONE', "
                    "last_error = NULL, "
                    "updated_at = CURRENT_TIMESTAMP WHERE document_id = ?",
                    (document_id,),
                )
                connection.execute(
                    "UPDATE sync_errors SET resolved_at = CURRENT_TIMESTAMP "
                    "WHERE document_id = ? AND stage = 'embedding' AND resolved_at IS NULL",
                    (document_id,),
                )
            elif kind == "failure":
                stage = str(event["stage"])
                column = stage_columns[stage]
                status = (
                    "FAILED_RETRYABLE" if bool(event.get("retryable"))
                    else "FAILED_PERMANENT"
                )
                message = str(event["message"])
                attempts = int(event.get("attempts") or 0)
                connection.execute(
                    f"UPDATE documents SET {column} = ?, status = ?, "
                    "retry_count = retry_count + ?, "
                    "last_error = ?, updated_at = CURRENT_TIMESTAMP WHERE document_id = ?",
                    (status, status, attempts, message, document_id),
                )
                connection.execute(
                    """
                    INSERT INTO sync_errors (
                        sync_run_id, document_id, stage, error_type, message,
                        retryable, attempts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sync_run_id,
                        document_id,
                        stage,
                        str(event.get("error_type") or "Error"),
                        message,
                        int(bool(event.get("retryable"))),
                        attempts,
                    ),
                )
            elif kind == "checkpoint":
                connection.execute(
                    """
                    UPDATE sync_runs
                    SET last_checkpoint_at = CURRENT_TIMESTAMP,
                        downloaded = ?, parsed = ?, embedded = ?, failed = ?,
                        documents_downloaded = ?, chunks_created = ?, errors = ?
                    WHERE sync_run_id = ?
                    """,
                    (
                        int(event.get("downloaded") or 0),
                        int(event.get("parsed") or 0),
                        int(event.get("embedded") or 0),
                        int(event.get("failed") or 0),
                        int(event.get("downloaded") or 0),
                        int(event.get("chunks") or 0),
                        int(event.get("failed") or 0),
                        sync_run_id,
                    ),
                )
            else:
                raise ValueError(f"Unsupported pipeline event kind: {kind}")


def replace_document_chunks(
    document_id: int,
    medicine_id: int,
    chunks: Sequence[Mapping[str, Any]],
    *,
    database_path: str | Path | None = None,
) -> int:
    """Atomically replace chunks for one official document only."""

    initialize_database(database_path)
    prepared = [
        (
            document_id,
            medicine_id,
            str(chunk["chunk_text"]).strip(),
            str(chunk.get("chunk_type") or chunk.get("section") or "general"),
            _to_optional_text(chunk.get("section")),
            _serialize_embedding(chunk.get("embedding")),
            _to_optional_text(chunk.get("embedding_model")),
            _to_optional_text(chunk.get("chunk_hash")),
            _to_optional_text(chunk.get("source_url")),
            _to_optional_text(chunk.get("approval_date")),
            _to_optional_text(chunk.get("source_type")),
            _to_optional_text(chunk.get("source_date")),
            int(chunk.get("source_priority") or 300),
            "DONE" if chunk.get("embedding") is not None else "PENDING",
        )
        for chunk in chunks
    ]
    with _connect(database_path) as connection:
        connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        _executemany_in_chunks(
            connection,
            """
            INSERT INTO document_chunks (
                document_id, medicine_id, chunk_text, chunk_type, section,
                embedding, embedding_model, chunk_hash, source_url,
                approval_date, source_type, source_date, source_priority,
                embedding_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            prepared,
        )
        connection.execute(
            "UPDATE documents SET embedding_status = ?, status = ?, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE document_id = ?",
            (
                "DONE"
                if chunks and all(chunk.get("embedding") is not None for chunk in chunks)
                else "PENDING",
                "DONE"
                if chunks and all(chunk.get("embedding") is not None for chunk in chunks)
                else "PENDING",
                document_id,
            ),
        )
    return len(chunks)


def create_sync_run(
    mode: str,
    *,
    database_path: str | Path | None = None,
) -> int:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "UPDATE sync_runs SET status = 'interrupted', finished_at = CURRENT_TIMESTAMP "
            "WHERE status = 'running'"
        )
        cursor = connection.execute(
            "INSERT INTO sync_runs (mode) VALUES (?)", (mode.strip(),)
        )
        return int(cursor.lastrowid)


def finish_sync_run(
    sync_run_id: int,
    counters: Mapping[str, int],
    *,
    status: str,
    error_log: str | None = None,
    database_path: str | Path | None = None,
) -> None:
    allowed = (
        "products_seen",
        "products_added",
        "products_updated",
        "documents_downloaded",
        "documents_updated",
        "chunks_created",
        "errors",
    )
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            f"UPDATE sync_runs SET finished_at = CURRENT_TIMESTAMP, status = ?, "
            f"error_log = ?, last_checkpoint_at = CURRENT_TIMESTAMP, "
            f"downloaded = ?, parsed = ?, embedded = ?, failed = ?, "
            f"{', '.join(f'{field} = ?' for field in allowed)} "
            "WHERE sync_run_id = ?",
            (
                status,
                _to_optional_text(error_log),
                int(counters.get("downloaded", counters.get("documents_downloaded", 0))),
                int(counters.get("parsed", 0)),
                int(counters.get("embedded", 0)),
                int(counters.get("failed", counters.get("errors", 0))),
                *(int(counters.get(field, 0)) for field in allowed),
                sync_run_id,
            ),
        )


def get_extended_database_stats(
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM medicines) AS medicines,
                (SELECT COUNT(*) FROM medicine_aliases) AS aliases,
                (SELECT COUNT(*) FROM documents WHERE document_type = 'KUB') AS kub_documents,
                (SELECT COUNT(*) FROM documents WHERE document_type = 'KT') AS kt_documents,
                (SELECT COUNT(*) FROM document_chunks) AS chunks,
                (SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL) AS embedded_chunks,
                (SELECT finished_at FROM sync_runs WHERE status LIKE 'completed%'
                 ORDER BY sync_run_id DESC LIMIT 1) AS last_sync
            """
        ).fetchone()
    return dict(row) if row is not None else {}


def get_pipeline_status(
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return durable download/parse/embedding pipeline counters."""

    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM medicines) AS total_medicines,
                (SELECT COUNT(*) FROM documents) AS total_documents,
                (SELECT COUNT(*) FROM documents WHERE download_status != 'DONE'
                    AND download_status != 'FAILED_PERMANENT') AS pending_downloads,
                (SELECT COUNT(*) FROM documents WHERE download_status = 'DONE') AS completed_downloads,
                (SELECT COUNT(*) FROM documents WHERE parse_status != 'DONE'
                    AND parse_status != 'FAILED_PERMANENT') AS pending_parse,
                (SELECT COUNT(*) FROM documents WHERE parse_status = 'DONE') AS completed_parse,
                (SELECT COUNT(*) FROM documents WHERE embedding_status != 'DONE'
                    AND embedding_status != 'FAILED_PERMANENT') AS pending_embeddings,
                (SELECT COUNT(*) FROM documents WHERE embedding_status = 'DONE') AS completed_embeddings,
                (SELECT COUNT(*) FROM document_chunks) AS chunks,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE embedding_status = 'DONE' AND embedding IS NOT NULL) AS embedded_chunks,
                (SELECT COUNT(*) FROM documents WHERE document_type = 'KT') AS kt_total,
                (SELECT COUNT(*) FROM documents
                    WHERE document_type = 'KT' AND embedding_status = 'DONE') AS kt_completed,
                (SELECT COUNT(*) FROM documents WHERE document_type = 'KUB') AS kub_total,
                (SELECT COUNT(*) FROM documents
                    WHERE document_type = 'KUB' AND embedding_status = 'DONE') AS kub_completed,
                (SELECT COUNT(*) FROM documents
                    WHERE status = 'FAILED_RETRYABLE') AS failed_retryable,
                (SELECT COUNT(*) FROM documents
                    WHERE status = 'FAILED_PERMANENT') AS failed_permanent,
                (SELECT MAX(updated_at) FROM documents
                    WHERE download_status = 'DONE'
                       OR parse_status = 'DONE'
                       OR embedding_status = 'DONE') AS last_successful_task,
                (SELECT status FROM sync_runs ORDER BY sync_run_id DESC LIMIT 1) AS sync_status,
                (SELECT last_checkpoint_at FROM sync_runs
                    ORDER BY sync_run_id DESC LIMIT 1) AS last_checkpoint_at
            """
        ).fetchone()
    return dict(row) if row is not None else {}


def get_database_stats(
    database_path: str | Path | None = None,
) -> dict[str, int]:
    """Return medicine and document chunk counts for status displays."""

    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM medicines) AS medicine_count,
                (SELECT COUNT(*) FROM document_chunks) AS chunk_count
            """
        ).fetchone()
    if row is None:
        return {"medicine_count": 0, "chunk_count": 0}
    return {
        "medicine_count": int(row["medicine_count"]),
        "chunk_count": int(row["chunk_count"]),
    }


def get_medicine_data_status(
    medicine_ids: Sequence[int],
    *,
    database_path: str | Path | None = None,
) -> dict[str, int | bool]:
    """Summarize catalog existence separately from queryable RAG content."""

    clean_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
    empty: dict[str, int | bool] = {
        "medicine_exists": False,
        "documents_count": 0,
        "kt_count": 0,
        "kub_count": 0,
        "chunks_count": 0,
        "side_effect_chunks": 0,
        "dosage_chunks": 0,
        "embeddings_count": 0,
    }
    if not clean_ids:
        return empty
    placeholders = ", ".join("?" for _ in clean_ids)
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            f"""
            SELECT
                EXISTS(
                    SELECT 1 FROM medicines WHERE medicine_id IN ({placeholders})
                ) AS medicine_exists,
                (SELECT COUNT(*) FROM documents
                    WHERE medicine_id IN ({placeholders})) AS documents_count,
                (SELECT COUNT(*) FROM documents
                    WHERE medicine_id IN ({placeholders}) AND document_type = 'KT')
                    AS kt_count,
                (SELECT COUNT(*) FROM documents
                    WHERE medicine_id IN ({placeholders}) AND document_type = 'KUB')
                    AS kub_count,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE medicine_id IN ({placeholders})) AS chunks_count,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE medicine_id IN ({placeholders})
                      AND chunk_type IN (
                          'side_effects', 'common_side_effects',
                          'serious_side_effects'
                      )) AS side_effect_chunks,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE medicine_id IN ({placeholders})
                      AND chunk_type IN (
                          'dosage', 'frequency', 'usage',
                          'route_of_administration'
                      )) AS dosage_chunks,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE medicine_id IN ({placeholders})
                      AND embedding IS NOT NULL
                      AND embedding_status = 'DONE') AS embeddings_count
            """,
            clean_ids * 8,
        ).fetchone()
    if row is None:
        return empty
    return {
        "medicine_exists": bool(row["medicine_exists"]),
        "documents_count": int(row["documents_count"]),
        "kt_count": int(row["kt_count"]),
        "kub_count": int(row["kub_count"]),
        "chunks_count": int(row["chunks_count"]),
        "side_effect_chunks": int(row["side_effect_chunks"]),
        "dosage_chunks": int(row["dosage_chunks"]),
        "embeddings_count": int(row["embeddings_count"]),
    }


def get_medicines_by_ids(
    medicine_ids: Sequence[int],
    *,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return complete medicine rows for diagnostic tooling."""

    clean_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
    if not clean_ids:
        return []
    placeholders = ", ".join("?" for _ in clean_ids)
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM medicines WHERE medicine_id IN ({placeholders}) "
            "ORDER BY medicine_name, medicine_id",
            clean_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_diagnostics(
    medicine_ids: Sequence[int],
    *,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return document pipeline state plus per-document chunk/embedding counts."""

    clean_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
    if not clean_ids:
        return []
    placeholders = ", ".join("?" for _ in clean_ids)
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT d.document_id, d.medicine_id, m.medicine_name,
                   d.document_type, d.status, d.download_status, d.parse_status,
                   d.embedding_status, d.document_url, d.local_path,
                   d.approval_date, d.retry_count, d.last_error,
                   COUNT(dc.chunk_id) AS chunk_count,
                   SUM(CASE WHEN dc.embedding IS NOT NULL
                                  AND dc.embedding_status = 'DONE'
                            THEN 1 ELSE 0 END) AS embedded_chunk_count,
                   SUM(CASE WHEN dc.chunk_id IS NOT NULL
                                  AND (dc.embedding IS NULL
                                    OR dc.embedding_status != 'DONE')
                            THEN 1 ELSE 0 END) AS pending_embedding_count,
                   GROUP_CONCAT(DISTINCT dc.embedding_model) AS embedding_models
            FROM documents AS d
            JOIN medicines AS m ON m.medicine_id = d.medicine_id
            LEFT JOIN document_chunks AS dc ON dc.document_id = d.document_id
            WHERE d.medicine_id IN ({placeholders})
            GROUP BY d.document_id
            ORDER BY m.medicine_name, d.document_type, d.document_id
            """,
            clean_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def get_medicine_source_urls(
    medicine_ids: Sequence[int],
    *,
    database_path: str | Path | None = None,
) -> list[str]:
    clean_ids = list(dict.fromkeys(int(value) for value in medicine_ids))
    if not clean_ids:
        return []
    placeholders = ", ".join("?" for _ in clean_ids)
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT document_url AS url FROM documents
            WHERE medicine_id IN ({placeholders}) AND document_url IS NOT NULL
            UNION
            SELECT source_url AS url FROM document_chunks
            WHERE medicine_id IN ({placeholders}) AND source_url IS NOT NULL
            UNION
            SELECT source_url AS url FROM medicine_dosage_rules
            WHERE medicine_id IN ({placeholders}) AND source_url IS NOT NULL
            ORDER BY url
            """,
            clean_ids * 3,
        ).fetchall()
    return [str(row["url"]) for row in rows if row["url"]]


def get_data_coverage(
    *,
    missing_limit: int = 20,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report how much of the catalog has documents and usable embeddings."""

    if missing_limit < 0:
        raise ValueError("missing_limit cannot be negative")
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM medicines) AS total_medicines,
                (SELECT COUNT(*) FROM documents) AS total_documents,
                (SELECT COUNT(*) FROM document_chunks) AS total_chunks,
                (SELECT COUNT(*) FROM document_chunks
                    WHERE embedding IS NOT NULL AND embedding_status = 'DONE')
                    AS total_embedded_chunks,
                (SELECT COUNT(DISTINCT medicine_id) FROM documents)
                    AS medicines_with_documents,
                (SELECT COUNT(DISTINCT medicine_id) FROM documents
                    WHERE document_type = 'KT') AS medicines_with_kt,
                (SELECT COUNT(DISTINCT medicine_id) FROM documents
                    WHERE document_type = 'KUB') AS medicines_with_kub,
                (SELECT COUNT(DISTINCT medicine_id) FROM document_chunks)
                    AS medicines_with_chunks,
                (SELECT COUNT(DISTINCT medicine_id) FROM document_chunks
                    WHERE embedding IS NOT NULL AND embedding_status = 'DONE')
                    AS medicines_with_embeddings
            """
        ).fetchone()
        missing_rows = connection.execute(
            """
            SELECT m.medicine_id, m.medicine_name
            FROM medicines AS m
            WHERE NOT EXISTS (
                SELECT 1 FROM document_chunks AS dc
                WHERE dc.medicine_id = m.medicine_id
                  AND dc.embedding IS NOT NULL
                  AND dc.embedding_status = 'DONE'
            )
            ORDER BY m.medicine_name, m.medicine_id
            LIMIT ?
            """,
            (missing_limit,),
        ).fetchall()
    total = int(row["total_medicines"]) if row is not None else 0
    embedded = int(row["medicines_with_embeddings"]) if row is not None else 0
    with_documents = int(row["medicines_with_documents"]) if row is not None else 0
    with_kt = int(row["medicines_with_kt"]) if row is not None else 0
    with_kub = int(row["medicines_with_kub"]) if row is not None else 0
    with_chunks = int(row["medicines_with_chunks"]) if row is not None else 0

    def percentage(value: int) -> float:
        return round((value / total * 100.0) if total else 0.0, 2)

    return {
        "total_medicines": total,
        "total_documents": int(row["total_documents"]),
        "total_chunks": int(row["total_chunks"]),
        "total_embedded_chunks": int(row["total_embedded_chunks"]),
        "medicines_with_documents": with_documents,
        "medicines_with_kt": with_kt,
        "medicines_with_kub": with_kub,
        "medicines_with_chunks": with_chunks,
        "medicines_with_embeddings": embedded,
        "documents_percentage": percentage(with_documents),
        "kt_percentage": percentage(with_kt),
        "kub_percentage": percentage(with_kub),
        "chunks_percentage": percentage(with_chunks),
        "embeddings_percentage": percentage(embedded),
        "coverage_percentage": percentage(embedded),
        "missing_examples": [dict(item) for item in missing_rows],
    }


def find_duplicate_database_paths(
    *,
    search_root: str | Path = BASE_DIR,
    canonical_path: str | Path = DATABASE_PATH,
) -> list[Path]:
    """Return medicines.db files other than the configured canonical database."""

    canonical = Path(canonical_path).resolve()
    return sorted(
        path.resolve()
        for path in Path(search_root).resolve().rglob("medicines.db")
        if path.resolve() != canonical
    )


def get_medicine_names(
    *,
    limit: int = 100,
    database_path: str | Path | None = None,
) -> list[str]:
    """Return registered medicine names for transparent UI feedback."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT medicine_name FROM medicines ORDER BY medicine_name LIMIT ?",
            (limit,),
        ).fetchall()
    return [str(row["medicine_name"]) for row in rows]


def get_all_medicines(
    *,
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    with _read_connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM medicines ORDER BY medicine_id"
        ).fetchall()
    return [dict(row) for row in rows]


def mark_unseen_titck_products(
    normalized_names: Sequence[str],
    *,
    database_path: str | Path | None = None,
) -> int:
    """Mark master-list products absent from a complete update, without deleting them."""

    clean_names = list(dict.fromkeys(name for name in normalized_names if name))
    if not clean_names:
        return 0
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS current_titck_products "
            "(normalized_name TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM current_titck_products")
        _executemany_in_chunks(
            connection,
            "INSERT INTO current_titck_products (normalized_name) VALUES (?)",
            ((name,) for name in clean_names),
        )
        cursor = connection.execute(
            """
            UPDATE medicines
            SET status = 'not_in_latest_list', updated_at = CURRENT_TIMESTAMP
            WHERE source = 'TİTCK'
              AND normalized_name NOT IN (
                  SELECT normalized_name FROM current_titck_products
              )
            """
        )
        return int(cursor.rowcount)


def require_database_content(
    database_path: str | Path | None = None,
) -> dict[str, int]:
    """Fail loudly when the selected database cannot support retrieval."""

    stats = get_database_stats(database_path)
    if stats["medicine_count"] <= 0 or stats["chunk_count"] <= 0:
        path = _database_path(database_path).resolve()
        raise RuntimeError(
            "İlaç veritabanı boş veya embedding chunk'ları eksik. "
            "Önce `python -m src.ingestion` komutunu çalıştırın. "
            f"Kullanılan veritabanı: {path}"
        )
    return stats


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Mapping):
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        text = "\n".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value).strip()
    return text or None


def _serialize_embedding(embedding: Sequence[float] | str | None) -> str | None:
    if embedding is None:
        return None
    if isinstance(embedding, str):
        try:
            parsed = json.loads(embedding)
        except json.JSONDecodeError as exc:
            raise ValueError("embedding must be a JSON array or a numeric sequence") from exc
    else:
        parsed = list(embedding)

    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in parsed
        )
    ):
        raise ValueError("embedding must contain finite numeric values")
    return json.dumps([float(value) for value in parsed], separators=(",", ":"))


def _deserialize_embedding(value: str | None) -> list[float] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(
                isinstance(item, Real)
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in parsed
            )
        ):
            return None
        return [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
