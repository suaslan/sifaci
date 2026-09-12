"""Retrieve medicine chunks with cosine similarity over local embeddings."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from config import (
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    MEDICINE_CANDIDATE_LIMIT,
    MEDICINE_NAME_BOOST,
    MINIMUM_SIMILARITY_SCORE,
    RETRIEVAL_TOP_K,
)
from src.database import (
    find_candidate_medicines,
    get_chunks,
    get_dosage_rules,
    get_medicine_data_status,
)
from src.dosage_parser import extract_query_age
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

_INTENT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "SIDE_EFFECTS",
        (
            "yan etki",
            "yan etkiler",
            "yan etkisi",
            "yan etkileri",
            "istenmeyen etki",
            "istenmeyen etkiler",
            "advers etkiler",
        ),
    ),
    ("FREQUENCY", ("günde kaç", "kaç kez", "kullanım sıklığı", "sıklık")),
    ("DOSAGE", ("doz", "pozoloji", "kaç mg")),
    ("INDICATION", ("ne için", "endikasyon", "hangi durumda")),
    ("ACTIVE_INGREDIENT", ("etken madde", "etkin madde", "aktif madde")),
    ("CONTRAINDICATION", ("kontrendikasyon", "kimler kullanmamalı")),
    ("INTERACTION", ("etkileşim", "birlikte kullan")),
    ("WARNING", ("uyarı", "dikkat")),
    ("STORAGE", ("saklama", "nasıl saklan")),
)

_INTENT_CHUNK_TYPES = {
    "SIDE_EFFECTS": {"common_side_effects", "side_effects", "serious_side_effects"},
    "FREQUENCY": {"frequency", "usage", "dosage"},
    "DOSAGE": {"dosage", "usage", "frequency"},
    "INDICATION": {"indications"},
    "ACTIVE_INGREDIENT": {"active_ingredient"},
    "CONTRAINDICATION": {"contraindications"},
    "INTERACTION": {"interactions"},
    "WARNING": {"warnings", "serious_side_effects"},
    "STORAGE": {"storage"},
}

_GENERIC_QUERY_WORDS = {
    "bu", "ciddi", "doz", "etken", "etkileri", "gunde", "hangi", "ilac",
    "kac", "kimler", "kullanim", "maddesi", "nasil", "ne", "nedir",
    "nelerdir", "uyari", "yan",
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
                "retrieval_state": None,
                "fallback_reason": None,
            }
        )

    intent = detect_intent(query)
    candidates = find_candidate_medicines(
        query,
        limit=MEDICINE_CANDIDATE_LIMIT,
        database_path=database_path,
    )
    matched_ids = [int(item["medicine_id"]) for item in candidates]
    if debug_trace is not None:
        debug_trace["detected_intent"] = intent
        debug_trace["matched_medicine_ids"] = matched_ids
        debug_trace["medicine_candidates"] = [
            {
                "medicine_id": item["medicine_id"],
                "medicine_name": item["medicine_name"],
                "alias": item["alias"],
                "name_score": item["name_score"],
                "match_type": item.get("match_type"),
                "matched_alias": item.get("matched_alias") or item.get("alias"),
            }
            for item in candidates
        ]
        debug_trace["detected_medicine"] = (
            str(candidates[0].get("matched_alias") or candidates[0].get("alias"))
            if candidates
            else None
        )

    if not candidates:
        if debug_trace is not None:
            suggestions = find_candidate_medicines(
                query,
                limit=5,
                minimum_fuzzy_score=0.48,
                database_path=database_path,
            )
            debug_trace["similar_medicines"] = [
                str(item["medicine_name"]) for item in suggestions[:5]
            ]
            debug_trace["retrieval_state"] = "medicine_not_found"
            debug_trace["fallback_reason"] = "medicine_not_found"
        return []

    data_status = get_medicine_data_status(matched_ids, database_path=database_path)
    database_chunks = get_chunks(
        medicine_ids=matched_ids,
        database_path=database_path,
    )
    if debug_trace is not None:
        debug_trace["database_chunk_count"] = len(database_chunks)
        debug_trace["medicine_data_status"] = data_status

    structured_results: list[dict[str, Any]] = []
    if candidates and intent in {"DOSAGE", "FREQUENCY"}:
        query_age = extract_query_age(query)
        dosage_rules = get_dosage_rules(
            [int(item["medicine_id"]) for item in candidates],
            age=query_age[0] if query_age else None,
            age_unit=query_age[1] if query_age else "year",
            database_path=database_path,
        )
        names_by_id = {
            int(item["medicine_id"]): str(item["medicine_name"])
            for item in candidates
        }
        structured_results = _structured_rule_results(
            dosage_rules,
            names_by_id,
            intent=intent,
            top_k=top_k,
        )
        if debug_trace is not None:
            debug_trace["query_age"] = (
                {"value": query_age[0], "unit": query_age[1]}
                if query_age
                else None
            )
            debug_trace["structured_dosage_rules"] = [
                {
                    "medicine_name": item["medicine_name"],
                    "chunk_type": item["chunk_type"],
                    "source_name": item["source_name"],
                }
                for item in structured_results
            ]

    intent_types = _INTENT_CHUNK_TYPES.get(intent, set())
    filtered_chunks = (
        [
            chunk
            for chunk in database_chunks
            if str(chunk.get("chunk_type") or "") in intent_types
        ]
        if intent_types
        else database_chunks
    )
    if debug_trace is not None:
        debug_trace["filtered_sections"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "medicine_id": chunk.get("medicine_id"),
                "document_id": chunk.get("document_id"),
                "chunk_type": chunk.get("chunk_type"),
                "section": chunk.get("section"),
                "document_type": chunk.get("document_type"),
            }
            for chunk in filtered_chunks
        ]

    if not filtered_chunks and not structured_results:
        if debug_trace is not None:
            has_queryable_data = (
                int(data_status["chunks_count"]) > 0
                and int(data_status["embeddings_count"]) > 0
            )
            state = "section_missing" if has_queryable_data else "rag_not_processed"
            debug_trace["retrieval_state"] = state
            debug_trace["fallback_reason"] = state
        return []

    candidate_chunks = []
    for chunk in filtered_chunks:
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
        if debug_trace is not None:
            debug_trace["top_chunks"] = [dict(item) for item in structured_results]
            if structured_results:
                debug_trace["retrieval_state"] = "chunks_found"
            else:
                debug_trace["retrieval_state"] = "rag_not_processed"
                debug_trace["fallback_reason"] = "embeddings_missing"
        return structured_results

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
        intent_match = str(chunk.get("chunk_type") or "") in intent_types
        source_rank = _source_rank(chunk, intent)
        section_rank = _section_rank(chunk, intent)
        ranking_score = (
            similarity
            + (MEDICINE_NAME_BOOST if name_match else 0.0)
            + (0.18 if intent_match else 0.0)
            + section_rank * 10.0
            + source_rank * 0.001
        )
        result = {
            "chunk_id": chunk.get("chunk_id"),
            "medicine_id": chunk.get("medicine_id"),
            "document_id": chunk.get("document_id"),
            "medicine_name": chunk.get("medicine_name"),
            "document_type": chunk.get("document_type"),
            "chunk_type": chunk.get("chunk_type"),
            "section": chunk.get("section"),
            "chunk_text": chunk.get("chunk_text"),
            "similarity_score": float(similarity),
            "ranking_score": float(ranking_score),
            "source_name": chunk.get("source_name"),
            "source_type": chunk.get("source_type"),
            "source_url": chunk.get("source_url") or chunk.get("source_reference"),
            "source_date": chunk.get("source_date") or chunk.get("approval_date"),
        }
        ranked.append((ranking_score, similarity, result))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    vector_results = [item[2] for item in ranked]
    results = _merge_results(structured_results, vector_results, top_k)
    if debug_trace is not None:
        debug_trace["top_chunks"] = [dict(result) for result in results]
        if results:
            debug_trace["retrieval_state"] = "chunks_found"
        else:
            debug_trace["retrieval_state"] = "low_similarity"
            debug_trace["fallback_reason"] = "low_similarity"
    return results


def _structured_rule_results(
    rules: Sequence[dict[str, Any]],
    names_by_id: dict[int, str],
    *,
    intent: str,
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for rule in rules:
        medicine_id = int(rule["medicine_id"])
        if intent == "FREQUENCY" and rule.get("frequency_text"):
            chunk_type = "frequency"
            text = str(rule["frequency_text"])
        else:
            chunk_type = "dosage"
            text = str(rule.get("raw_source_text") or rule.get("dose_text") or "").strip()
        if not text:
            continue
        key = (medicine_id, text)
        if key in seen:
            continue
        seen.add(key)
        source_type = str(rule.get("source_type") or "")
        source_name = {
            "TITCK": "TİTCK yapılandırılmış doz kuralı",
            "MANUFACTURER": "Üretici resmi doz bilgisi",
            "ILACABAK": "İlacabak prospektüsü",
        }.get(source_type, source_type or "Kayıtlı doz kaynağı")
        results.append(
            {
                "medicine_name": names_by_id.get(medicine_id, "Belirtilmemiş"),
                "chunk_type": chunk_type,
                "chunk_text": text,
                "similarity_score": 1.0,
                "source_name": source_name,
                "source_type": source_type,
                "source_url": rule.get("source_url"),
                "source_date": rule.get("source_date"),
            }
        )
        if len(results) >= top_k:
            break
    return results


def _source_rank(chunk: dict[str, Any], intent: str) -> int:
    """Encode the deterministic evidence order without changing similarity."""

    source_type = str(chunk.get("source_type") or "").upper()
    document_type = str(chunk.get("document_type") or "").upper()
    chunk_type = str(chunk.get("chunk_type") or "")
    if intent in {"DOSAGE", "FREQUENCY"}:
        if source_type == "TITCK" and document_type == "KUB" and chunk_type in {
            "dosage",
            "frequency",
            "usage",
        }:
            return 4
        if source_type == "TITCK" and document_type == "KT" and chunk_type in {
            "dosage",
            "frequency",
            "usage",
        }:
            return 3
        if source_type == "MANUFACTURER" and chunk_type in {
            "dosage",
            "frequency",
            "usage",
        }:
            return 3
        if source_type == "ILACABAK" and chunk_type in {"dosage", "frequency"}:
            return 2
        return 1
    try:
        return max(0, int(chunk.get("source_priority") or 0))
    except (TypeError, ValueError):
        return 0


def _section_rank(chunk: dict[str, Any], intent: str) -> int:
    document_type = str(chunk.get("document_type") or "").upper()
    section = _normalize_for_match(str(chunk.get("section") or ""))
    chunk_type = str(chunk.get("chunk_type") or "")
    if intent == "SIDE_EFFECTS":
        if document_type == "KT" and chunk_type in {
            "side_effects",
            "common_side_effects",
            "serious_side_effects",
        }:
            return 5
        if document_type == "KUB" and chunk_type in {
            "side_effects",
            "common_side_effects",
            "serious_side_effects",
        }:
            return 4
        if "yan etki" in section:
            return 3
        if "advers etki" in section:
            return 2
        return 1
    if intent in {"DOSAGE", "FREQUENCY"}:
        if document_type == "KT" and "nasil kullan" in section:
            return 4
        if document_type == "KT" and (
            "uygun kullanim" in section or "uygulama sikligi" in section
        ):
            return 3
        if document_type == "KUB" and (
            "4 2" in section or "pozoloji" in section
        ):
            return 2
        return 1
    return 2 if chunk_type in _INTENT_CHUNK_TYPES.get(intent, set()) else 1


def _merge_results(
    structured: Sequence[dict[str, Any]],
    semantic: Sequence[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in [*structured, *semantic]:
        key = (
            str(result.get("medicine_name") or ""),
            str(result.get("chunk_type") or ""),
            str(result.get("chunk_text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged


def detect_intent(query: str) -> str:
    """Detect the retrieval intent without using an LLM."""

    normalized = _normalize_for_match(query)
    compact = normalized.replace(" ", "")
    if (
        "yanetki" in compact
        or "istenmeyenetki" in compact
        or "adversetki" in compact
    ):
        return "SIDE_EFFECTS"
    for intent, terms in _INTENT_TERMS:
        if any(_normalize_for_match(term) in normalized for term in terms):
            return intent
    query_tokens = normalized.split()
    for intent, terms in _INTENT_TERMS:
        for term in terms:
            normalized_term = _normalize_for_match(term)
            term_size = len(normalized_term.split())
            windows = [
                " ".join(query_tokens[index : index + term_size])
                for index in range(max(1, len(query_tokens) - term_size + 1))
            ]
            if any(fuzz.ratio(window, normalized_term) >= 86 for window in windows):
                return intent
    return "GENERAL"


def _query_has_explicit_medicine_name(query: str) -> bool:
    tokens = _normalize_for_match(query).split()
    if not tokens:
        return False
    return tokens[0] not in _GENERIC_QUERY_WORDS


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
