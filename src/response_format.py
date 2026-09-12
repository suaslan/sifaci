"""Shared parsing helpers for formatted RAG responses."""

from __future__ import annotations

import re

from config import MISSING_INFORMATION_RESPONSE, SOURCE_SECTION_MARKER
from src.rag import DISCLAIMER


MAX_DISPLAY_ANSWER_CHARS = 6_000
_SVG_BLOCK = re.compile(r"<svg\b[^>]*>.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
_ESCAPED_SVG_BLOCK = re.compile(
    r"&lt;svg\b.*?&lt;/svg\s*&gt;", re.IGNORECASE | re.DOTALL
)
_SVG_ELEMENT = re.compile(
    r"</?(?:svg|path|circle|rect|line|polyline|polygon|ellipse|g)\b[^>]*>",
    re.IGNORECASE,
)
_ESCAPED_SVG_ELEMENT = re.compile(
    r"&lt;/?(?:svg|path|circle|rect|line|polyline|polygon|ellipse|g)\b.*?&gt;",
    re.IGNORECASE,
)
_STANDALONE_SVG_WORD = re.compile(r"(?im)^\s*svg\s*$")


def sanitize_answer_text(answer: str) -> str:
    """Remove leaked SVG markup and reject oversized catalog-like output."""

    cleaned = _SVG_BLOCK.sub("", answer)
    cleaned = _ESCAPED_SVG_BLOCK.sub("", cleaned)
    cleaned = _SVG_ELEMENT.sub("", cleaned)
    cleaned = _ESCAPED_SVG_ELEMENT.sub("", cleaned)
    cleaned = _STANDALONE_SVG_WORD.sub("", cleaned).strip()
    if len(cleaned) > MAX_DISPLAY_ANSWER_CHARS:
        return MISSING_INFORMATION_RESPONSE
    return cleaned or MISSING_INFORMATION_RESPONSE


def split_rag_response(response: str) -> tuple[str, str, str]:
    """Split a RAG response into answer, source names, and disclaimer."""

    if SOURCE_SECTION_MARKER not in response:
        return sanitize_answer_text(response), "Belirtilmemiş", DISCLAIMER

    answer, source_section = response.rsplit(SOURCE_SECTION_MARKER, maxsplit=1)
    source_names, separator, disclaimer = source_section.partition("\n\n")
    return (
        sanitize_answer_text(answer),
        source_names.strip() or "Belirtilmemiş",
        disclaimer.strip() if separator and disclaimer.strip() else DISCLAIMER,
    )


def parse_source_names(source_names: str) -> list[str]:
    """Convert the RAG source line into a stable JSON-friendly list."""

    if source_names in {"", "Belirtilmemiş", "Bulunamadı"}:
        return []
    return [source.strip() for source in source_names.split(",") if source.strip()]
