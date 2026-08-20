"""Import user-provided JSON medicine records into SQLite."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config import DATABASE_PATH, MEDICINE_DOCUMENTS_DIR
from src.database import (
    get_chunks,
    initialize_database,
    insert_medicine,
    replace_chunks,
)


DEFAULT_MAX_CHUNK_CHARS = 1_200

# JSON input key -> (database field, chunk category)
SECTION_FIELDS: dict[str, tuple[str, str]] = {
    "active_ingredient": ("active_ingredient", "active_ingredient"),
    "indications": ("indications", "indications"),
    "usage_information": ("usage_information", "usage"),
    "dosage_information": ("dosage_information", "dosage"),
    "dosage": ("dosage_information", "dosage"),
    "frequency_information": ("frequency_information", "frequency"),
    "frequency": ("frequency_information", "frequency"),
    "route_of_administration": (
        "route_of_administration",
        "route_of_administration",
    ),
    "common_side_effects": ("common_side_effects", "side_effects"),
    "serious_side_effects": ("serious_side_effects", "serious_side_effects"),
    "side_effects": ("common_side_effects", "side_effects"),
    "warnings": ("warnings", "warnings"),
    "contraindications": ("contraindications", "contraindications"),
    "interactions": ("interactions", "interactions"),
}


def discover_json_files(directory: str | Path = MEDICINE_DOCUMENTS_DIR) -> list[Path]:
    """Return all JSON files under the medicine directory in stable order."""

    return sorted(Path(directory).rglob("*.json"))


def load_medicine_file(path: str | Path) -> list[dict[str, Any]]:
    """Load either one JSON object or an array of medicine objects."""

    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc

    records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{file_path} must contain an object or an array of objects")
    return records


def split_text_safely(text: Any, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split long text at paragraph or sentence boundaries, never mid-sentence.

    A single sentence longer than ``max_chars`` remains intact.  This is
    deliberate for dosage and other safety-critical instructions.
    """

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    normalized = _as_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue

        # A line boundary (frequently used for leaflet bullet lists) is also a
        # semantic boundary. Within each long line, split only after terminal
        # punctuation and before a likely new sentence. Decimal values such as
        # 2.5 remain untouched.
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            if len(line) <= max_chars:
                units.append(line)
                continue
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])", line)
            units.extend(
                sentence.strip() for sentence in sentences if sentence.strip()
            )

    chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if "\n" in unit or "\n" in current else " "
        candidate = f"{current}{separator if current else ''}{unit}"
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def normalize_medicine(record: Mapping[str, Any]) -> dict[str, str | None]:
    """Map the friendly JSON format to fields used by the database."""

    name = _as_text(record.get("medicine_name"))
    if not name:
        raise ValueError("Each medicine record must contain medicine_name")

    normalized: dict[str, str | None] = {
        "medicine_name": name,
        "active_ingredient": None,
        "indications": None,
        "usage_information": None,
        "dosage_information": None,
        "frequency_information": None,
        "route_of_administration": None,
        "common_side_effects": None,
        "serious_side_effects": None,
        "warnings": None,
        "contraindications": None,
        "interactions": None,
        "source_name": None,
        "source_reference": None,
    }

    for input_field, (database_field, _) in SECTION_FIELDS.items():
        value = _as_text(record.get(input_field))
        if value and not normalized[database_field]:
            normalized[database_field] = value

    source = record.get("source")
    if isinstance(source, Mapping):
        normalized["source_name"] = _as_text(source.get("name"))
        normalized["source_reference"] = _as_text(
            source.get("reference") or source.get("url")
        )
    elif source is not None:
        normalized["source_name"] = _as_text(source)

    # Explicit source fields take precedence over the compact source format.
    normalized["source_name"] = (
        _as_text(record.get("source_name")) or normalized["source_name"]
    )
    normalized["source_reference"] = (
        _as_text(record.get("source_reference"))
        or normalized["source_reference"]
    )
    return normalized


def create_chunks(
    record: Mapping[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[dict[str, str]]:
    """Convert meaningful medicine sections into typed text chunks."""

    medicine_name = _as_text(record.get("medicine_name"))
    if not medicine_name:
        raise ValueError("Each medicine record must contain medicine_name")

    chunks: list[dict[str, str]] = []
    used_database_fields: set[str] = set()
    for input_field, (database_field, chunk_type) in SECTION_FIELDS.items():
        # Alias keys (for example dosage/dosage_information) should not create
        # duplicate chunks when both are present.
        if database_field in used_database_fields:
            continue
        value = record.get(input_field)
        if not _as_text(value):
            continue
        used_database_fields.add(database_field)

        for part in split_text_safely(value, max_chars=max_chars):
            chunks.append(
                {
                    "chunk_text": f"İlaç: {medicine_name}\nKategori: {chunk_type}\n{part}",
                    "chunk_type": chunk_type,
                }
            )
    return chunks


def ingest_medicine_record(
    record: Mapping[str, Any],
    *,
    database_path: str | Path = DATABASE_PATH,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
) -> tuple[int, int]:
    """Insert one medicine and its chunks; return ``(medicine_id, count)``."""

    normalized = normalize_medicine(record)
    medicine_id = insert_medicine(normalized, database_path=database_path)
    chunks = create_chunks(record, max_chars=max_chunk_chars)

    existing_chunks = get_chunks(medicine_id, database_path=database_path)
    if _chunks_are_current(existing_chunks, chunks):
        return medicine_id, len(chunks)

    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    if embedding_function is None:
        # Lazy import keeps JSON discovery and parsing usable even before the
        # optional native Foundry Local runtime has been installed.
        from src.embeddings import generate_embeddings

        embeddings = generate_embeddings(chunk_texts)
    else:
        embeddings = [embedding_function(text) for text in chunk_texts]

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match the document chunk count")
    chunks_with_embeddings = [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    replace_chunks(
        medicine_id,
        chunks_with_embeddings,
        database_path=database_path,
    )
    return medicine_id, len(chunks)


def ingest_directory(
    directory: str | Path = MEDICINE_DOCUMENTS_DIR,
    *,
    database_path: str | Path = DATABASE_PATH,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
) -> dict[str, int]:
    """Import every JSON file and return ingestion counters."""

    initialize_database(database_path)
    stats = {"files": 0, "medicines": 0, "chunks": 0, "skipped_examples": 0}
    for path in discover_json_files(directory):
        stats["files"] += 1
        for record in load_medicine_file(path):
            if record.get("is_example") is True:
                stats["skipped_examples"] += 1
                continue
            _, chunk_count = ingest_medicine_record(
                record,
                database_path=database_path,
                max_chunk_chars=max_chunk_chars,
                embedding_function=embedding_function,
            )
            stats["medicines"] += 1
            stats["chunks"] += chunk_count
    return stats


def _chunks_are_current(
    existing_chunks: Sequence[Mapping[str, Any]],
    new_chunks: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true when unchanged chunks already have stored embeddings."""

    if len(existing_chunks) != len(new_chunks):
        return False
    return all(
        existing.get("chunk_text") == new.get("chunk_text")
        and existing.get("chunk_type") == new.get("chunk_type")
        and bool(existing.get("embedding"))
        for existing, new in zip(existing_chunks, new_chunks, strict=True)
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON ilaç kayıtlarını SQLite'a aktarır.")
    parser.add_argument("--directory", type=Path, default=MEDICINE_DOCUMENTS_DIR)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS)
    args = parser.parse_args()

    result = ingest_directory(
        args.directory,
        database_path=args.database,
        max_chunk_chars=args.max_chars,
    )
    print(
        f"{result['files']} dosya tarandı; {result['medicines']} ilaç ve "
        f"{result['chunks']} parça aktarıldı."
    )
    if result["skipped_examples"]:
        print(f"{result['skipped_examples']} örnek kayıt atlandı.")


if __name__ == "__main__":
    main()
