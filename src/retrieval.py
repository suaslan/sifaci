"""Retrieve medicine chunks with cosine similarity over local embeddings."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from config import (
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    MEDICINE_NAME_BOOST,
    MINIMUM_SIMILARITY_SCORE,
    RETRIEVAL_TOP_K,
)
from src.database import get_chunks
from src.embeddings import generate_embedding


_GENERIC_NAME_WORDS = {
    "ampul",
    "film",
    "flakon",
    "kapsul",
    "kapsül",
    "mg",
    "ml",
    "saşe",
    "surup",
    "şurup",
    "tablet",
}


def get_top_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    *,
    database_path: str | Path = DATABASE_PATH,
    embedding_function: Callable[[str], Sequence[float]] | None = None,
    minimum_score: float = MINIMUM_SIMILARITY_SCORE,
    debug_trace: MutableMapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the most relevant trustworthy chunks for ``query``.

    An empty list means no chunk met ``minimum_score``. An explicitly named
    medicine receives a small ranking boost, while a semantically stronger
    result can still outrank it.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query cannot be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not -1.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between -1 and 1")

    if debug_trace is not None:
        debug_trace.update(
            {
                "query_embedding_created": False,
                "query_embedding_dimension": None,
                "query_embedding_preview": [],
                "minimum_similarity_score": minimum_score,
                "requested_top_k": top_k,
                "database_chunk_count": 0,
                "stale_embedding_count": 0,
                "top_chunks": [],
            }
        )

    database_chunks = get_chunks(database_path=database_path)
    if debug_trace is not None:
        debug_trace["database_chunk_count"] = len(database_chunks)

    candidate_chunks = []
    for chunk in database_chunks:
        raw_embedding = chunk.get("embedding")
        if raw_embedding is None:
            continue
        stored_model = chunk.get("embedding_model")
        if stored_model and stored_model != EMBEDDING_MODEL_NAME:
            if debug_trace is not None:
                debug_trace["stale_embedding_count"] += 1
            continue
        candidate_chunks.append(chunk)

    if not candidate_chunks:
        return []

    embed = embedding_function or generate_embedding
    query_embedding = _validated_vector(embed(query.strip()), "query embedding")
    if debug_trace is not None:
        debug_trace["query_embedding_created"] = True
        debug_trace["query_embedding_dimension"] = len(query_embedding)
        debug_trace["query_embedding_preview"] = query_embedding[:8]
    normalized_query = _normalize_for_match(query)
    ranked: list[tuple[float, float, dict[str, Any]]] = []

    for chunk in candidate_chunks:
        raw_embedding = chunk["embedding"]
        try:
            chunk_embedding = _validated_vector(raw_embedding, "chunk embedding")
            similarity = cosine_similarity(query_embedding, chunk_embedding)
        except ValueError:
            # One corrupt/stale vector must not make all retrieval unavailable.
            continue
        if similarity < minimum_score:
            continue

        name_match = _medicine_name_is_explicit(
            normalized_query, str(chunk.get("medicine_name") or "")
        )
        ranking_score = similarity + (MEDICINE_NAME_BOOST if name_match else 0.0)
        result = {
            "medicine_name": chunk.get("medicine_name"),
            "chunk_type": chunk.get("chunk_type"),
            "chunk_text": chunk.get("chunk_text"),
            "similarity_score": float(similarity),
            "source_name": chunk.get("source_name"),
        }
        ranked.append((ranking_score, similarity, result))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results = [item[2] for item in ranked[:top_k]]
    if debug_trace is not None:
        debug_trace["top_chunks"] = [dict(result) for result in results]
    return results


def cosine_similarity(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Calculate cosine similarity for two equal-length numerical vectors."""

    left = _validated_vector(first, "first vector")
    right = _validated_vector(second, "second vector")
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, dot_product / (left_norm * right_norm)))


def _validated_vector(vector: Sequence[float], label: str) -> list[float]:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numerical") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite values")
    return values


def _normalize_for_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"\w+", without_marks, flags=re.UNICODE))


def _medicine_name_is_explicit(normalized_query: str, medicine_name: str) -> bool:
    normalized_name = _normalize_for_match(medicine_name)
    if not normalized_name:
        return False
    if re.search(rf"(?<!\w){re.escape(normalized_name)}(?!\w)", normalized_query):
        return True

    # Product names commonly continue with strength/form information. Matching
    # their leading brand token catches queries such as "Parol yan etkileri"
    # for a record named "PAROL 500 mg tablet".
    name_tokens = [
        token
        for token in normalized_name.split()
        if len(token) >= 3 and not token.isdigit() and token not in _GENERIC_NAME_WORDS
    ]
    if not name_tokens:
        return False
    brand_token = name_tokens[0]
    return bool(re.search(rf"(?<!\w){re.escape(brand_token)}(?!\w)", normalized_query))
