"""Deterministic age, frequency and route extraction from source dosage text."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any


_POPULATION_PATTERN = re.compile(
    r"(?P<range>\b\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?\s*(?:yaş|ay))"
    r"|(?P<over>\b\d+(?:[.,]\d+)?\s*(?:yaş|ay)(?:ından)?\s*(?:ve\s+)?(?:büyük|üzeri))"
    r"|(?P<under>\b\d+(?:[.,]\d+)?\s*(?:yaş|ay)(?:ından)?\s*(?:küçük|altı))"
    r"|(?P<adult>\b(?:yetişkinler|erişkinler)\b)"
    r"|(?P<elderly>\byaşlılar\b)",
    flags=re.IGNORECASE,
)

_FREQUENCY_PATTERNS = (
    re.compile(r"\bgünde\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:defa|kez|doz)\b", re.I),
    re.compile(r"\b\d+(?:\s*[-–]\s*\d+)?\s*saatte\s+bir\b", re.I),
    re.compile(r"\b\d+\s*saat\s+ara\s+ile\b", re.I),
    re.compile(r"\bgünde\s+(?:bir|iki|üç|dört|beş|altı)\s+eşit\s+doz\b", re.I),
    re.compile(r"\b24\s*saatte\s+en\s+fazla\s+\d+\s+doz\b", re.I),
)

_ROUTE_TERMS = re.compile(
    r"\b(?:oral|ağızdan|çiğnenmeden|yutularak|göze|damlatılarak|buruna|"
    r"intravenöz|intramüsküler|subkutan|deri altına|cilde|topikal|rektal)\b",
    re.IGNORECASE,
)
_DURATION_TERMS = re.compile(
    r"\b\d+(?:\s*[-–]\s*\d+)?\s*(?:gün|hafta|ay|saat)\b",
    re.IGNORECASE,
)


def parse_dosage_rules(
    dosage_text: str,
    *,
    source_type: str,
    source_url: str,
    source_date: str | None = None,
) -> list[dict[str, Any]]:
    """Preserve source wording while extracting only explicitly stated bounds."""

    clean = dosage_text.strip()
    if not clean:
        return []
    matches = list(_POPULATION_PATTERN.finditer(clean))
    blocks: list[tuple[str | None, str]] = []
    if not matches:
        blocks.append((None, clean))
    else:
        preamble = clean[: matches[0].start()].strip(" \n:-")
        if preamble and _has_dose_signal(preamble):
            blocks.append((None, preamble))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
            blocks.append((match.group(0).strip(), clean[match.start() : end].strip()))

    rules: list[dict[str, Any]] = []
    for population, block in blocks:
        if not block or not _has_dose_signal(block):
            continue
        min_age, max_age, age_unit = _age_bounds(population)
        frequencies = _find_all_preserving_order(_FREQUENCY_PATTERNS, block)
        sentences = _sentences(block)
        route_text = " ".join(sentence for sentence in sentences if _ROUTE_TERMS.search(sentence))
        duration_text = " ".join(
            sentence for sentence in sentences if _DURATION_TERMS.search(sentence)
        )
        rule_hash = hashlib.sha256(
            f"{source_url}\n{population or ''}\n{block}".encode("utf-8")
        ).hexdigest()
        rules.append(
            {
                "population": population or "genel",
                "min_age": min_age,
                "max_age": max_age,
                "age_unit": age_unit,
                "dose_text": block,
                "frequency_text": "; ".join(frequencies) or None,
                "route_text": route_text or None,
                "duration_text": duration_text or None,
                "instruction_text": block,
                "source_type": source_type,
                "source_url": source_url,
                "source_date": source_date,
                "raw_source_text": block,
                "rule_hash": rule_hash,
            }
        )
    return rules


def extract_query_age(text: str) -> tuple[float, str] | None:
    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(yaşında|yaşındaki|yaş|aylık|ay)\b", text, re.I)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = "month" if match.group(2).casefold().startswith("ay") else "year"
    return value, unit


def _age_bounds(population: str | None) -> tuple[float | None, float | None, str | None]:
    if not population:
        return None, None, None
    numbers = [float(value.replace(",", ".")) for value in re.findall(r"\d+(?:[.,]\d+)?", population)]
    unit = "month" if re.search(r"\bay\b", population, re.I) else "year"
    if re.search(r"[-–]", population) and len(numbers) >= 2:
        return numbers[0], numbers[1], unit
    if re.search(r"(?:büyük|üzeri)", population, re.I) and numbers:
        return numbers[0], None, unit
    if re.search(r"(?:küçük|altı)", population, re.I) and numbers:
        return None, numbers[0], unit
    return None, None, unit if numbers else None


def _has_dose_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:doz|tablet|kapsül|damla|ml|mg|gram|günde|saatte|uygulan|alın|veril)\b",
            text,
            re.IGNORECASE,
        )
    )


def _find_all_preserving_order(patterns: Iterable[re.Pattern[str]], text: str) -> list[str]:
    matches = [match for pattern in patterns for match in pattern.finditer(text)]
    matches.sort(key=lambda item: item.start())
    return list(dict.fromkeys(match.group(0).strip() for match in matches))


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if item.strip()
    ]
