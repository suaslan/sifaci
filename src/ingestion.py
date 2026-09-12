"""Bulk-import user-provided JSON and CSV medicine records into SQLite."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from config import (
    DATABASE_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_CHARS,
    MEDICINE_DOCUMENTS_DIR,
    MEDICINE_FILE_EXTENSIONS,
)
from src.database import (
    MEDICINE_FIELDS,
    get_chunks,
    get_database_stats,
    initialize_database,
    replace_chunks,
    save_chunk_embedding_batch,
    upsert_document,
    upsert_medicine,
)
from src.embeddings import iter_embedding_batches, load_embedding_model
from tqdm.auto import tqdm


# Input key -> (database field, semantic chunk category)
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
    "common_side_effects": ("common_side_effects", "common_side_effects"),
    "serious_side_effects": ("serious_side_effects", "serious_side_effects"),
    "side_effects": ("common_side_effects", "side_effects"),
    "warnings": ("warnings", "warnings"),
    "contraindications": ("contraindications", "contraindications"),
    "interactions": ("interactions", "interactions"),
}

IngestionStatus = Literal["inserted", "updated", "duplicate"]


def discover_medicine_files(
    directory: str | Path = MEDICINE_DOCUMENTS_DIR,
) -> list[Path]:
    """Return supported JSON/CSV files recursively in stable order."""

    extensions = {suffix.casefold() for suffix in MEDICINE_FILE_EXTENSIONS}
    return sorted(
        path
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    )


def discover_json_files(directory: str | Path = MEDICINE_DOCUMENTS_DIR) -> list[Path]:
    """Backward-compatible JSON-only discovery helper."""

    return [path for path in discover_medicine_files(directory) if path.suffix.casefold() == ".json"]


def load_medicine_file(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON object/array or a header-based CSV medicine file."""

    file_path = Path(path)
    suffix = file_path.suffix.casefold()
    if suffix == ".json":
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
        records = payload if isinstance(payload, list) else [payload]
    elif suffix == ".csv":
        text = file_path.read_text(encoding="utf-8-sig")
        try:
            dialect = csv.Sniffer().sniff(text[:8_192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"CSV header is missing in {file_path}")
        records = [
            {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
    else:
        raise ValueError(f"Unsupported medicine file type: {file_path.suffix}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{file_path} must contain medicine objects")
    return records


def split_text_safely(text: Any, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split at paragraphs or sentence boundaries, never in mid-sentence."""

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
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            if len(line) <= max_chars:
                units.append(line)
                continue
            # A safety-critical sentence longer than max_chars remains intact.
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])", line)
            units.extend(sentence.strip() for sentence in sentences if sentence.strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def normalize_medicine(record: Mapping[str, Any]) -> dict[str, str | None]:
    """Map friendly JSON/CSV fields to the canonical database structure."""

    name = _as_text(record.get("medicine_name"))
    if not name:
        raise ValueError("Each medicine record must contain medicine_name")

    normalized: dict[str, str | None] = {field: None for field in MEDICINE_FIELDS}
    normalized["medicine_name"] = name
    for input_field, (database_field, _) in SECTION_FIELDS.items():
        value = _as_text(record.get(input_field))
        if value and not normalized[database_field]:
            normalized[database_field] = value

    source = record.get("source")
    if isinstance(source, Mapping):
        normalized["source_name"] = _as_text(source.get("name")) or None
        normalized["source_reference"] = _as_text(
            source.get("reference") or source.get("url")
        ) or None
    elif source is not None:
        normalized["source_name"] = _as_text(source) or None

    normalized["source_name"] = (
        _as_text(record.get("source_name")) or normalized["source_name"]
    )
    normalized["source_reference"] = (
        _as_text(record.get("source_reference")) or normalized["source_reference"]
    )
    return normalized


def create_chunks(
    record: Mapping[str, Any],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict[str, str]]:
    """Convert populated medicine sections into typed semantic chunks."""

    medicine_name = _as_text(record.get("medicine_name"))
    if not medicine_name:
        raise ValueError("Each medicine record must contain medicine_name")

    chunks: list[dict[str, str]] = []
    used_database_fields: set[str] = set()
    for input_field, (database_field, chunk_type) in SECTION_FIELDS.items():
        if database_field in used_database_fields:
            continue
        value = record.get(input_field)
        if not _as_text(value):
            continue
        used_database_fields.add(database_field)
        for part in split_text_safely(value, max_chars=max_chars):
            chunks.append(
                {
                    "chunk_text": (
                        f"İlaç: {medicine_name}\nKategori: {chunk_type}\n{part}"
                    ),
                    "chunk_type": chunk_type,
                }
            )
    return chunks


def _ingest_medicine_record_details(
    record: Mapping[str, Any],
    *,
    database_path: str | Path,
    max_chunk_chars: int,
    embedding_model_name: str,
    embedding_function: Callable[[str], Sequence[float]] | None,
    embedding_progress: tqdm[Any] | None = None,
) -> tuple[int, int, int, IngestionStatus]:
    normalized = normalize_medicine(record)
    medicine_id, status = upsert_medicine(normalized, database_path=database_path)
    _register_source_document_metadata(
        medicine_id,
        normalized,
        database_path=database_path,
    )
    chunks = create_chunks(record, max_chars=max_chunk_chars)
    existing_chunks = get_chunks(medicine_id, database_path=database_path)
    if _chunks_are_current(existing_chunks, chunks, embedding_model_name):
        return medicine_id, len(chunks), 0, status

    chunk_ids = replace_chunks(
        medicine_id,
        chunks,
        database_path=database_path,
    )
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    batch_function = None
    if embedding_function is not None:
        batch_function = lambda batch: [
            list(embedding_function(text)) for text in batch
        ]
    for offset, vectors in iter_embedding_batches(
        chunk_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        embedding_function=batch_function,
    ):
        save_chunk_embedding_batch(
            list(
                zip(
                    chunk_ids[offset : offset + len(vectors)],
                    vectors,
                    strict=True,
                )
            ),
            embedding_model=embedding_model_name,
            database_path=database_path,
        )
        if embedding_progress is not None:
            embedding_progress.update(len(vectors))
    return medicine_id, len(chunks), len(chunks), status


def ingest_medicine_record(
    record: Mapping[str, Any],
    *,
    database_path: str | Path = DATABASE_PATH,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
) -> tuple[int, int]:
    """Insert/upsert one medicine and return ``(medicine_id, chunk_count)``."""

    medicine_id, chunk_count, _, _ = _ingest_medicine_record_details(
        record,
        database_path=database_path,
        max_chunk_chars=max_chunk_chars,
        embedding_model_name=embedding_model_name,
        embedding_function=embedding_function,
    )
    return medicine_id, chunk_count


def ingest_directory(
    directory: str | Path = MEDICINE_DOCUMENTS_DIR,
    *,
    database_path: str | Path = DATABASE_PATH,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
    show_progress: bool = False,
) -> dict[str, int]:
    """Import every JSON/CSV record and return detailed counters."""

    initialize_database(database_path)
    if show_progress and embedding_function is None:
        load_embedding_model()
    stats = {
        "files": 0,
        "medicines": 0,
        "inserted": 0,
        "updated": 0,
        "duplicates": 0,
        "errors": 0,
        "chunks": 0,
        "skipped_examples": 0,
    }
    embedding_progress = tqdm(
        desc="Embedding",
        unit="chunk",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for path in discover_medicine_files(directory):
        stats["files"] += 1
        try:
            records = load_medicine_file(path)
        except (OSError, UnicodeError, ValueError) as error:
            stats["errors"] += 1
            print(f"Hatalı dosya: {path} ({error})", file=sys.stderr)
            continue

        for row_number, record in enumerate(records, start=1):
            if _is_example_record(record):
                stats["skipped_examples"] += 1
                continue
            try:
                _, _, written_chunks, status = _ingest_medicine_record_details(
                    record,
                    database_path=database_path,
                    max_chunk_chars=max_chunk_chars,
                    embedding_model_name=embedding_model_name,
                    embedding_function=embedding_function,
                    embedding_progress=embedding_progress,
                )
            except Exception as error:
                stats["errors"] += 1
                print(
                    f"Hatalı kayıt: {path} satır/kayıt {row_number} ({error})",
                    file=sys.stderr,
                )
                continue
            stats["medicines"] += 1
            stats[status + "s" if status == "duplicate" else status] += 1
            stats["chunks"] += written_chunks
    embedding_progress.close()
    return stats


def _chunks_are_current(
    existing_chunks: Sequence[Mapping[str, Any]],
    new_chunks: Sequence[Mapping[str, Any]],
    embedding_model_name: str,
) -> bool:
    if len(existing_chunks) != len(new_chunks):
        return False
    return all(
        existing.get("chunk_text") == new.get("chunk_text")
        and existing.get("chunk_type") == new.get("chunk_type")
        and bool(existing.get("embedding"))
        and existing.get("embedding_model") == embedding_model_name
        for existing, new in zip(existing_chunks, new_chunks, strict=True)
    )


def _is_example_record(record: Mapping[str, Any]) -> bool:
    value = record.get("is_example")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _register_source_document_metadata(
    medicine_id: int,
    medicine: Mapping[str, Any],
    *,
    database_path: str | Path,
) -> None:
    """Register an official PDF reference without claiming it was downloaded."""

    source_name = _as_text(medicine.get("source_name"))
    source_reference = _as_text(medicine.get("source_reference"))
    if not source_reference.casefold().split("?", 1)[0].endswith(".pdf"):
        return
    normalized_source = source_name.casefold()
    if "kullanma talimat" in normalized_source or re.search(r"\bkt\b", normalized_source):
        document_type = "KT"
    elif "küb" in normalized_source or "kub" in normalized_source:
        document_type = "KUB"
    else:
        return
    approval_match = re.search(r"\b\d{2}[.]\d{2}[.]\d{4}\b", source_name)
    upsert_document(
        medicine_id,
        document_type,
        source_reference,
        approval_date=approval_match.group(0) if approval_match else None,
        source="TİTCK",
        database_path=database_path,
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


def _print_summary(result: Mapping[str, int], database_path: str | Path) -> None:
    database_stats = get_database_stats(database_path)
    print(f"Toplam dosya: {result['files']}")
    print(f"Başarıyla işlenen ilaç: {result['medicines']}")
    print(f"Yeni eklenen ilaç: {result['inserted']}")
    print(f"Güncellenen ilaç: {result['updated']}")
    print(f"Atlanan duplicate: {result['duplicates']}")
    print(f"Hatalı kayıt: {result['errors']}")
    print(f"Toplam oluşturulan chunk: {result['chunks']}")
    print(f"SELECT COUNT(*) FROM medicines: {database_stats['medicine_count']}")
    print(f"SELECT COUNT(*) FROM document_chunks: {database_stats['chunk_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JSON/CSV ilaç kayıtlarını SQLite'a topluca aktarır."
    )
    parser.add_argument("--directory", type=Path, default=MEDICINE_DOCUMENTS_DIR)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--max-chars", type=int, default=MAX_CHUNK_CHARS)
    args = parser.parse_args()

    result = ingest_directory(
        args.directory,
        database_path=args.database,
        max_chunk_chars=args.max_chars,
        show_progress=True,
    )
    _print_summary(result, args.database)


if __name__ == "__main__":
    main()
