"""SQLite schema and data-access helpers for medicine documents.

Embeddings are stored as JSON arrays.  The public read functions deserialize
them back to Python lists so callers do not need to know about the storage
format.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any

from config import DATABASE_PATH, EMBEDDING_MODEL_NAME


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


def _database_path(database_path: str | Path | None = None) -> Path:
    """Resolve the configured database path and ensure its directory exists."""

    path = Path(database_path) if database_path is not None else Path(DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(
    database_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Yield a transactional SQLite connection and always close it."""

    connection = sqlite3.connect(_database_path(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: str | Path | None = None) -> None:
    """Create the medicine and document chunk tables when they do not exist."""

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
                UNIQUE (medicine_name, source_name, source_reference)
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                embedding TEXT,
                embedding_model TEXT,
                FOREIGN KEY (medicine_id)
                    REFERENCES medicines (medicine_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_medicines_name
                ON medicines (medicine_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_medicine_id
                ON document_chunks (medicine_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_type
                ON document_chunks (chunk_type);
            """
        )
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(document_chunks)")
        }
        if "embedding_model" not in columns:
            connection.execute(
                "ALTER TABLE document_chunks ADD COLUMN embedding_model TEXT"
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
        # ``IS ?`` provides null-safe matching: unlike a UNIQUE constraint,
        # repeated imports with missing source values still find the same row.
        existing = connection.execute(
            """
            SELECT medicine_id
            FROM medicines
            WHERE medicine_name = ? COLLATE NOCASE
              AND source_name IS ?
              AND source_reference IS ?
            LIMIT 1
            """,
            (
                values["medicine_name"],
                values["source_name"],
                values["source_reference"],
            ),
        ).fetchone()

        if existing is not None:
            update_fields = tuple(
                field for field in MEDICINE_FIELDS if field != "medicine_name"
            )
            assignments = ", ".join(f"{field} = ?" for field in update_fields)
            connection.execute(
                f"UPDATE medicines SET {assignments} WHERE medicine_id = ?",
                tuple(values[field] for field in update_fields)
                + (existing["medicine_id"],),
            )
            return int(existing["medicine_id"])

        placeholders = ", ".join("?" for _ in MEDICINE_FIELDS)
        columns = ", ".join(MEDICINE_FIELDS)
        cursor = connection.execute(
            f"INSERT INTO medicines ({columns}) VALUES ({placeholders})",
            tuple(values[field] for field in MEDICINE_FIELDS),
        )
        return int(cursor.lastrowid)


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
                (medicine_id, chunk_text, chunk_type, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                medicine_id,
                clean_text,
                clean_type,
                serialized_embedding,
                serialized_model,
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

    prepared: list[tuple[int, str, str, str | None, str | None]] = []
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
            )
        )

    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE medicine_id = ?", (medicine_id,)
        )
        chunk_ids: list[int] = []
        for values in prepared:
            cursor = connection.execute(
                """
                INSERT INTO document_chunks
                    (medicine_id, chunk_text, chunk_type, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?)
                """,
                values,
            )
            chunk_ids.append(int(cursor.lastrowid))
    return chunk_ids


def get_chunks(
    medicine_id: int | None = None,
    *,
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
    if chunk_type is not None:
        conditions.append("dc.chunk_type = ?")
        parameters.append(chunk_type)

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        "SELECT dc.chunk_id, dc.medicine_id, dc.chunk_text, dc.chunk_type, "
        "dc.embedding, dc.embedding_model, m.medicine_name, m.source_name, "
        "m.source_reference "
        "FROM document_chunks AS dc "
        "JOIN medicines AS m ON m.medicine_id = dc.medicine_id"
        f"{where_clause} ORDER BY dc.chunk_id"
    )

    with _connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    chunks = [dict(row) for row in rows]
    for chunk in chunks:
        chunk["embedding"] = _deserialize_embedding(chunk["embedding"])
    return chunks


def get_medicine_by_name(
    medicine_name: str,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find a medicine by exact name, case-insensitively."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
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


def get_database_stats(
    database_path: str | Path | None = None,
) -> dict[str, int]:
    """Return medicine and document chunk counts for status displays."""

    initialize_database(database_path)
    with _connect(database_path) as connection:
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
