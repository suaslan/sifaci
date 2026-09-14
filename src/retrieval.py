"""Retrieve medicine chunks with cosine similarity over local embeddings."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

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
    get_medicines_by_ids,
    search_medicine_chunks_fts,
)
from src.dosage_parser import extract_query_age
from src.embeddings import generate_embedding
from src.intents import (
    INTENT_CHUNK_TYPES,
    SUGGESTION_SUFFIXES,
    chunk_supports_intent,
    detect_intent,
    intent_search_text,
    normalize_text,
)
from src.medicine_resolution import medicine_signature


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
                    "name_score": item.get("name_score", 0.0),
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
            suggestion_candidates = find_candidate_medicines(
                query,
                limit=MEDICINE_CANDIDATE_LIMIT,
                minimum_fuzzy_score=0.48,
                database_path=database_path,
            )
            suggestion_chunks = (
                get_chunks(
                    medicine_ids=[
                        int(item["medicine_id"]) for item in suggestion_candidates
                    ],
                    database_path=database_path,
                )
                if suggestion_candidates
                else []
            )
            processed_suggestions = _processed_candidates(
                suggestion_candidates, suggestion_chunks
            )
            debug_trace["similar_medicines"] = _suggestion_queries(
                processed_suggestions, intent=intent
            )
            debug_trace["retrieval_state"] = "medicine_not_found"
            debug_trace["fallback_reason"] = "medicine_not_found"
        return []

    if len({int(item["medicine_id"]) for item in candidates}) > 1:
        same_brand_candidates = _same_commercial_brand_candidates(candidates)
        if same_brand_candidates:
            candidates = same_brand_candidates
            matched_ids = [int(item["medicine_id"]) for item in candidates]
            if debug_trace is not None:
                debug_trace["matched_medicine_ids"] = matched_ids

    if len({int(item["medicine_id"]) for item in candidates}) > 1:
        narrowed_candidates = _narrow_candidates_by_explicit_product(query, candidates)
        if narrowed_candidates:
            candidates = narrowed_candidates
            matched_ids = [int(item["medicine_id"]) for item in candidates]
            if debug_trace is not None:
                debug_trace["matched_medicine_ids"] = matched_ids

    database_chunks = get_chunks(
        medicine_ids=matched_ids,
        database_path=database_path,
    )

    # A short brand may represent several strengths or dosage forms. Never mix
    # their evidence in one medical answer. Prefer the sole processed match, or
    # ask the user to select an exact processed product when several remain.
    if len({int(item["medicine_id"]) for item in candidates}) > 1:
        processed_candidates = _processed_candidates(candidates, database_chunks)
        if len(processed_candidates) > 1:
            if debug_trace is not None:
                debug_trace["similar_medicines"] = _suggestion_queries(
                    processed_candidates, intent=intent
                )
                debug_trace["retrieval_state"] = "medicine_ambiguous"
                debug_trace["fallback_reason"] = "medicine_ambiguous"
                debug_trace["database_chunk_count"] = len(database_chunks)
            return []
        if len(processed_candidates) == 1:
            candidates = processed_candidates
            matched_ids = [int(processed_candidates[0]["medicine_id"])]
            database_chunks = [
                chunk
                for chunk in database_chunks
                if int(chunk.get("medicine_id") or -1) == matched_ids[0]
            ]
            if debug_trace is not None:
                debug_trace["selected_medicine_id"] = matched_ids[0]
                debug_trace["detected_medicine"] = str(
                    processed_candidates[0]["medicine_name"]
                )

    data_status = get_medicine_data_status(matched_ids, database_path=database_path)
    if debug_trace is not None:
        debug_trace["database_chunk_count"] = len(database_chunks)
        debug_trace["medicine_data_status"] = data_status

    structured_results = (
        _structured_metadata_results(
            get_medicines_by_ids(matched_ids, database_path=database_path),
            intent=intent,
        )
        if candidates and all(candidate.get("match_type") for candidate in candidates)
        else []
    )
    if candidates and intent in {"DOSAGE", "FREQUENCY", "USAGE"}:
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
        structured_results = _merge_results(
            _structured_rule_results(
                dosage_rules,
                names_by_id,
                intent=intent,
                top_k=top_k,
            ),
            structured_results,
            top_k,
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

    intent_types = set(INTENT_CHUNK_TYPES.get(intent, ()))
    filtered_chunks = [
        chunk
        for chunk in database_chunks
        if chunk_supports_intent(
            str(chunk.get("chunk_type") or ""),
            f"{chunk.get('section') or ''}\n{chunk.get('chunk_text') or ''}",
            intent,
        )
    ]
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

    fts_chunks = search_medicine_chunks_fts(
        intent_search_text(intent, query),
        medicine_ids=matched_ids,
        chunk_types=None,
        limit=max(top_k * 3, top_k),
        database_path=database_path,
    )
    fts_results = [
        _retrieval_result(
            chunk,
            similarity=1.0 / (1.0 + index),
            ranking_score=(20.0 - index) + _source_rank(chunk, intent) * 0.001,
            retrieval_method="fts5",
        )
        for index, chunk in enumerate(fts_chunks)
        if chunk_supports_intent(
            str(chunk.get("chunk_type") or ""),
            f"{chunk.get('section') or ''}\n{chunk.get('chunk_text') or ''}",
            intent,
        )
    ]
    if debug_trace is not None:
        debug_trace["fts_match_count"] = len(fts_results)

    section_results = [
        _retrieval_result(
            chunk,
            similarity=1.0,
            ranking_score=(
                100.0
                + _section_rank(chunk, intent) * 10.0
                + _source_rank(chunk, intent) * 0.001
            ),
            retrieval_method="section",
            retrieval_intent=intent,
        )
        for chunk in filtered_chunks
        if intent != "GENERAL"
    ]
    section_results.sort(
        key=lambda item: float(item.get("ranking_score") or 0.0), reverse=True
    )

    # In the interactive production path an exact stored section or FTS hit is
    # already deterministic medicine-scoped evidence. Avoid waking the local
    # embedding model solely to re-rank evidence that will remain above vector
    # results. Explicit embedding functions (tests/diagnostics) still exercise
    # the full hybrid path.
    if embedding_function is None and (
        structured_results or section_results or fts_results
    ):
        deterministic_results = _merge_results(
            structured_results,
            sorted(
                [*section_results, *fts_results],
                key=lambda item: float(item.get("ranking_score") or 0.0),
                reverse=True,
            ),
            top_k,
        )
        if debug_trace is not None:
            debug_trace["embedding_skipped_reason"] = "deterministic_evidence_found"
            debug_trace["top_chunks"] = [dict(item) for item in deterministic_results]
            debug_trace["retrieval_state"] = "chunks_found"
        return deterministic_results

    candidate_chunks = []
    embedding_pool = filtered_chunks
    for chunk in embedding_pool:
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
        results = _merge_results(
            structured_results,
            _merge_results(section_results, fts_results, top_k),
            top_k,
        )
        if debug_trace is not None:
            debug_trace["top_chunks"] = [dict(item) for item in results]
            if results:
                debug_trace["retrieval_state"] = "chunks_found"
            else:
                has_queryable_data = int(data_status["chunks_count"]) > 0
                state = "section_missing" if has_queryable_data else "rag_not_processed"
                debug_trace["retrieval_state"] = state
                debug_trace["fallback_reason"] = state
        return results

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
        exact_semantic_section = (
            str(chunk.get("chunk_type") or "") in intent_types
            and chunk_supports_intent(
                str(chunk.get("chunk_type") or ""),
                f"{chunk.get('section') or ''}\n{chunk.get('chunk_text') or ''}",
                intent,
            )
        )
        if similarity < minimum_score and not exact_semantic_section:
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
        result = _retrieval_result(
            chunk,
            similarity=similarity,
            ranking_score=ranking_score,
            retrieval_method="embedding",
            retrieval_intent=intent,
        )
        ranked.append((ranking_score, similarity, result))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    vector_results = [item[2] for item in ranked]
    hybrid_results = sorted(
        [*section_results, *fts_results, *vector_results],
        key=lambda item: float(item.get("ranking_score") or 0.0),
        reverse=True,
    )
    results = _merge_results(structured_results, hybrid_results, top_k)
    if debug_trace is not None:
        debug_trace["top_chunks"] = [dict(result) for result in results]
        if results:
            debug_trace["retrieval_state"] = "chunks_found"
        else:
            has_queryable_data = int(data_status["chunks_count"]) > 0
            state = "section_missing" if has_queryable_data else "rag_not_processed"
            debug_trace["retrieval_state"] = state
            debug_trace["fallback_reason"] = state
    return results


def _retrieval_result(
    chunk: dict[str, Any],
    *,
    similarity: float,
    ranking_score: float,
    retrieval_method: str,
    retrieval_intent: str | None = None,
) -> dict[str, Any]:
    return {
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
        "retrieval_method": retrieval_method,
        "retrieval_intent": retrieval_intent,
        "source_name": chunk.get("source_name"),
        "source_type": chunk.get("source_type"),
        "source_url": chunk.get("source_url") or chunk.get("source_reference"),
        "source_date": chunk.get("source_date") or chunk.get("approval_date"),
    }


_METADATA_FIELDS = {
    "ACTIVE_INGREDIENT": ("active_ingredient", "active_ingredient"),
    "INDICATION": ("indications", "indications"),
    "USAGE": ("usage_information", "usage"),
    "DOSAGE": ("dosage_information", "dosage"),
    "FREQUENCY": ("frequency_information", "frequency"),
    "ROUTE_OF_ADMINISTRATION": ("route_of_administration", "route_of_administration"),
    "SIDE_EFFECTS": ("common_side_effects", "common_side_effects"),
    "SERIOUS_SIDE_EFFECTS": ("serious_side_effects", "serious_side_effects"),
    "WARNING": ("warnings", "warnings"),
    "CONTRAINDICATION": ("contraindications", "contraindications"),
    "INTERACTION": ("interactions", "interactions"),
}


def _structured_metadata_results(
    medicines: Sequence[dict[str, Any]], *, intent: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for medicine in medicines:
        if intent == "WHAT_IS":
            facts = [
                f"Ürün adı: {medicine.get('medicine_name')}",
                f"Etken madde: {medicine.get('active_ingredient')}",
                f"Firma: {medicine.get('company')}",
                f"Farmasötik form: {medicine.get('pharmaceutical_form')}",
                f"Yitilik: {medicine.get('strength')}",
            ]
            text = "\n".join(item for item in facts if not item.endswith("None"))
            chunk_type = "general"
        else:
            mapping = _METADATA_FIELDS.get(intent)
            if mapping is None:
                continue
            field, chunk_type = mapping
            text = str(medicine.get(field) or "").strip()
        if not text:
            continue
        results.append(
            {
                "medicine_id": medicine.get("medicine_id"),
                "medicine_name": medicine.get("medicine_name"),
                "chunk_type": chunk_type,
                "chunk_text": text,
                "similarity_score": 1.0,
                "ranking_score": 200.0,
                "retrieval_method": "metadata",
                "retrieval_intent": intent,
                "source_name": medicine.get("source_name") or medicine.get("source") or "Yerel ürün kaydı",
                "source_type": medicine.get("source") or "LOCAL_METADATA",
                "source_url": medicine.get("source_reference"),
                "source_date": medicine.get("license_date"),
            }
        )
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
    if intent in {"DOSAGE", "FREQUENCY", "USAGE"}:
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
    if intent in {"DOSAGE", "FREQUENCY", "USAGE"}:
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
    return 2 if chunk_type in INTENT_CHUNK_TYPES.get(intent, ()) else 1


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


def _processed_candidates(
    candidates: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique candidates that have current, queryable embeddings."""

    queryable_ids = {
        int(chunk["medicine_id"])
        for chunk in chunks
        if chunk.get("medicine_id") is not None
        and chunk.get("embedding") is not None
        and (
            not chunk.get("embedding_model")
            or chunk.get("embedding_model") == EMBEDDING_MODEL_NAME
        )
    }
    processed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        medicine_id = int(candidate["medicine_id"])
        if medicine_id in queryable_ids and medicine_id not in seen:
            seen.add(medicine_id)
            processed.append(candidate)
    return processed


def _same_commercial_brand_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Exclude PLUS/FORTE/DUO sub-brands from an exact base-brand query."""

    if not candidates:
        return []
    matched_alias = str(
        candidates[0].get("matched_alias") or candidates[0].get("alias") or ""
    )
    requested_brand = _compact_commercial_brand(matched_alias)
    if not requested_brand:
        return list(candidates)
    exact_brand = [
        candidate
        for candidate in candidates
        if _compact_commercial_brand(str(candidate.get("medicine_name") or ""))
        == requested_brand
    ]
    return exact_brand or list(candidates)


def _narrow_candidates_by_explicit_product(
    query: str, candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Use an explicitly written strength/form to avoid mixing brand variants."""

    normalized_query = normalize_text(query)
    exact_mentions = [
        candidate
        for candidate in candidates
        if re.search(
            rf"(?:^|\s){re.escape(normalize_text(str(candidate.get('medicine_name') or '')))}(?:\s|$)",
            normalized_query,
        )
    ]
    if len(exact_mentions) == 1:
        return exact_mentions

    query_signature = medicine_signature(query)
    query_strengths = set(query_signature["strengths"])
    query_forms = set(query_signature["forms"])
    compatible: list[dict[str, Any]] = []
    for candidate in candidates:
        signature = medicine_signature(str(candidate.get("medicine_name") or ""))
        candidate_strengths = set(signature["strengths"])
        candidate_forms = set(signature["forms"])
        if query_strengths and candidate_strengths != query_strengths:
            continue
        if query_forms and candidate_forms and query_forms.isdisjoint(candidate_forms):
            continue
        compatible.append(candidate)
    return compatible or list(candidates)


def _compact_commercial_brand(value: str) -> str:
    prefix: list[str] = []
    for token in _normalize_for_match(value).split():
        if any(character.isdigit() for character in token):
            break
        if token in _GENERIC_NAME_WORDS:
            break
        prefix.append(token)
    return "".join(prefix)


def _suggestion_queries(
    candidates: Sequence[dict[str, Any]], *, intent: str, limit: int = 5
) -> list[str]:
    """Build ready-to-run questions while preserving the original intent."""

    suffix = SUGGESTION_SUFFIXES.get(intent, "hakkında kayıtlı bilgi verir misin?")
    queries: list[str] = []
    for candidate in candidates:
        query = f"{str(candidate['medicine_name']).strip()} {suffix}"
        if query not in queries:
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries


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
    return normalize_text(text)


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
