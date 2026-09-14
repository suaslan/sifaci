"""Medicine lookup facade used by the RAG and API layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import DATABASE_PATH, RETRIEVAL_TOP_K
from src.database import find_candidate_medicines
from src.retrieval import get_top_chunks


def find_medicine(
    query: str,
    *,
    database_path: str | Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    """Resolve a brand name or active ingredient from a natural-language query."""

    candidates = find_candidate_medicines(query, limit=20, database_path=database_path)
    return candidates[0] if candidates else None


def search_medicine_information(
    medicine_id: int,
    question: str,
    *,
    database_path: str | Path = DATABASE_PATH,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    """Return question-relevant chunks and retain only the requested medicine."""

    return [
        chunk
        for chunk in get_relevant_chunks(
            question, database_path=database_path, top_k=top_k
        )
        if int(chunk.get("medicine_id") or -1) == int(medicine_id)
    ]


def get_relevant_chunks(
    question: str,
    *,
    database_path: str | Path = DATABASE_PATH,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    """Resolve the medicine and retrieve its FTS5/vector-ranked source chunks."""

    return get_top_chunks(question, top_k=top_k, database_path=database_path)
