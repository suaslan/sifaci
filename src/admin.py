"""Validation and ingestion helpers used by the administrator medicine form."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config import DATABASE_PATH
from src.database import MEDICINE_FIELDS
from src.ingestion import ingest_medicine_record


ADMIN_MEDICINE_FIELDS = MEDICINE_FIELDS


def validate_medicine_form(values: Mapping[str, Any]) -> dict[str, str | None]:
    """Validate form values without inventing values for empty fields."""

    unknown_fields = set(values) - set(ADMIN_MEDICINE_FIELDS)
    if unknown_fields:
        raise ValueError(f"Desteklenmeyen alanlar: {sorted(unknown_fields)}")

    record: dict[str, str | None] = {}
    for field in ADMIN_MEDICINE_FIELDS:
        value = values.get(field)
        if value is None:
            record[field] = None
            continue
        if not isinstance(value, str):
            raise TypeError(f"{field} metin olmalıdır")
        clean_value = value.strip()
        record[field] = clean_value or None

    if not record["medicine_name"]:
        raise ValueError("İlaç adı zorunludur.")
    return record


def save_medicine_from_form(
    values: Mapping[str, Any],
    *,
    database_path: str | Path = DATABASE_PATH,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
) -> tuple[int, int]:
    """Validate and save one medicine with its embedded document chunks."""

    record = validate_medicine_form(values)
    return ingest_medicine_record(
        record,
        database_path=database_path,
        embedding_function=embedding_function,
    )
