"""Turkish-aware medicine name normalization and alias generation."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


_FORM_TOKENS = {
    "ampul",
    "draje",
    "efervesan",
    "film",
    "flakon",
    "jel",
    "kapli",
    "kapsul",
    "krem",
    "losyon",
    "merhem",
    "oral",
    "saşe",
    "solusyon",
    "sprey",
    "surup",
    "tablet",
}
_STRENGTH_TOKENS = {"g", "mcg", "mg", "ml", "mikrogram", "miligram"}


def normalize_medicine_name(value: str) -> str:
    """Normalize Turkish names for exact and fuzzy matching."""

    translated = value.casefold().translate(str.maketrans({"ı": "i", "ş": "s"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def generate_medicine_aliases(product_name: str) -> list[str]:
    """Generate conservative aliases without merging strength/form variants."""

    tokens = product_name.split()
    aliases = [product_name.strip()]
    prefix: list[str] = []
    for token in tokens:
        normalized = normalize_medicine_name(token)
        if (
            not normalized
            or any(char.isdigit() for char in normalized)
            or normalized in _FORM_TOKENS
            or normalized in _STRENGTH_TOKENS
        ):
            break
        prefix.append(token)
        aliases.append(" ".join(prefix))

    # Keep the shortest brand alias and a useful multi-token commercial alias.
    unique: list[str] = []
    for alias in aliases:
        clean = alias.strip()
        if clean and normalize_medicine_name(clean) not in {
            normalize_medicine_name(item) for item in unique
        }:
            unique.append(clean)
    return unique


def fuzzy_name_score(query: str, candidate: str) -> float:
    """Return a 0..1 fuzzy score, using RapidFuzz when available."""

    left = normalize_medicine_name(query)
    right = normalize_medicine_name(candidate)
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(left, right)) / 100.0
    except ImportError:
        return SequenceMatcher(None, left, right).ratio()
