"""Conservative canonical medicine matching for KÜB/KT document records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rapidfuzz import fuzz

from src.medicine_names import normalize_medicine_name


_FORM_TOKENS = {
    "ampul", "damla", "draje", "efervesan", "film", "flakon", "jel",
    "kapsul", "krem", "losyon", "merhem", "oral", "pastil", "saşe",
    "solusyon", "sprey", "surup", "suspansiyon", "tablet", "toz",
}
_UNIT_TOKENS = {"g", "iu", "mcg", "mg", "mikrogram", "ml", "miligram", "unit"}


def medicine_signature(name: str, pharmaceutical_form: str | None = None) -> dict[str, Any]:
    normalized = normalize_medicine_name(name)
    tokens = normalized.split()
    brand_tokens: list[str] = []
    for token in tokens:
        if any(character.isdigit() for character in token) or token in _FORM_TOKENS:
            break
        brand_tokens.append(token)
    strengths: list[str] = []
    for index, token in enumerate(tokens):
        compact = re.sub(r"[^a-z0-9]", "", token)
        if not any(character.isdigit() for character in compact):
            continue
        unit_match = re.search(r"(mcg|mg|ml|iu|g)$", compact)
        if unit_match:
            strengths.append(compact)
        elif index + 1 < len(tokens) and tokens[index + 1] in _UNIT_TOKENS:
            strengths.append(f"{compact}{tokens[index + 1]}")
    form_text = normalize_medicine_name(pharmaceutical_form or "")
    forms = sorted({token for token in [*tokens, *form_text.split()] if token in _FORM_TOKENS})
    return {
        "normalized_name": normalized,
        "brand": " ".join(brand_tokens),
        "strengths": tuple(strengths),
        "forms": tuple(forms),
    }


def resolve_canonical_medicine(
    document_product: Mapping[str, Any],
    canonical_medicines: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve a document product without guessing between ambiguous products."""

    product_name = str(
        document_product.get("product_name")
        or document_product.get("medicine_name")
        or document_product.get("name")
        or ""
    ).strip()
    if not product_name:
        return {"status": "unresolved", "reason": "document_product_name_missing", "candidates": []}
    wanted = medicine_signature(
        product_name,
        str(document_product.get("pharmaceutical_form") or ""),
    )
    wanted_ingredient = normalize_medicine_name(str(document_product.get("active_ingredient") or document_product.get("element") or ""))
    wanted_company = normalize_medicine_name(str(document_product.get("company") or document_product.get("firmName") or ""))

    scored: list[dict[str, Any]] = []
    for candidate in canonical_medicines:
        medicine_id = candidate.get("medicine_id")
        candidate_name = str(candidate.get("medicine_name") or candidate.get("product_name") or "").strip()
        if medicine_id is None or not candidate_name:
            continue
        signature = medicine_signature(candidate_name, str(candidate.get("pharmaceutical_form") or ""))
        exact_name = signature["normalized_name"] == wanted["normalized_name"]
        brand_score = fuzz.ratio(wanted["brand"], signature["brand"]) / 100.0 if wanted["brand"] and signature["brand"] else 0.0
        if not exact_name and brand_score < 0.86:
            continue
        wanted_strengths = set(wanted["strengths"])
        candidate_strengths = set(signature["strengths"])
        strength_conflict = bool(wanted_strengths and candidate_strengths and wanted_strengths != candidate_strengths)
        wanted_forms = set(wanted["forms"])
        candidate_forms = set(signature["forms"])
        form_conflict = bool(wanted_forms and candidate_forms and wanted_forms.isdisjoint(candidate_forms))
        if strength_conflict or form_conflict:
            continue
        name_score = fuzz.ratio(wanted["normalized_name"], signature["normalized_name"]) / 100.0
        ingredient = normalize_medicine_name(str(candidate.get("active_ingredient") or ""))
        company = normalize_medicine_name(str(candidate.get("company") or ""))
        ingredient_score = fuzz.token_set_ratio(wanted_ingredient, ingredient) / 100.0 if wanted_ingredient and ingredient else 0.0
        company_score = fuzz.token_set_ratio(wanted_company, company) / 100.0 if wanted_company and company else 0.0
        strength_match = bool(wanted_strengths and wanted_strengths == candidate_strengths)
        form_match = bool(wanted_forms and candidate_forms and not wanted_forms.isdisjoint(candidate_forms))
        score = (
            name_score * 0.58
            + brand_score * 0.20
            + ingredient_score * 0.12
            + company_score * 0.05
            + (0.03 if strength_match else 0.0)
            + (0.02 if form_match else 0.0)
        )
        if exact_name:
            score = 1.0
        scored.append(
            {
                "medicine_id": int(medicine_id),
                "medicine_name": candidate_name,
                "score": round(score, 6),
                "name_score": round(name_score, 6),
                "brand_score": round(brand_score, 6),
                "ingredient_score": round(ingredient_score, 6),
                "company_score": round(company_score, 6),
                "strength_match": strength_match,
                "form_match": form_match,
                "exact_name": exact_name,
            }
        )
    scored.sort(key=lambda item: (item["score"], item["name_score"]), reverse=True)
    if not scored:
        return {"status": "unresolved", "reason": "no_compatible_candidate", "candidates": []}

    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    high_confidence = (
        best["exact_name"]
        or (best["brand_score"] == 1.0 and best["strength_match"] and best["name_score"] >= 0.82)
        or best["name_score"] >= 0.95
        or (
            best["name_score"] >= 0.88
            and best["ingredient_score"] >= 0.95
            and best["company_score"] >= 0.75
        )
    )
    ambiguous = second is not None and not best["exact_name"] and best["score"] - second["score"] < 0.03
    if not high_confidence:
        return {"status": "unresolved", "reason": "confidence_too_low", "candidates": scored[:5]}
    if ambiguous:
        return {"status": "ambiguous", "reason": "multiple_high_confidence_candidates", "candidates": scored[:5]}
    return {
        "status": "matched",
        "reason": "exact_normalized_name" if best["exact_name"] else "compatible_high_confidence_match",
        "medicine_id": best["medicine_id"],
        "medicine_name": best["medicine_name"],
        "score": best["score"],
        "candidates": scored[:5],
    }
