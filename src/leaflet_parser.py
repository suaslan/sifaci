"""Section-aware parser for Turkish TİTCK KÜB and KT documents."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable

from config import SYNC_CHUNK_MAX_CHARS, SYNC_CHUNK_MIN_CHARS


_KT_HEADINGS = (
    (r"^\s*1[.)]?\s+.*?nedir\s+ve\s+ne\s+için\s+kullanılır", "Ne için kullanılır?", "indications"),
    (r"^\s*2[.)]?\s+.*?kullanmadan\s+önce", "Kullanmadan önce dikkat edilmesi gerekenler", "warnings"),
    (r"^\s*3[.)]?\s+.*?nasıl\s+kullanılır", "Nasıl kullanılır?", "usage"),
    (r"^\s*4[.)]?\s+olası\s+yan\s+etkiler", "Olası yan etkiler", "common_side_effects"),
    (r"^\s*5[.)]?\s+.*?saklanması", "Saklanması", "storage"),
)

_KUB_HEADINGS = (
    (r"^\s*2[.)]?\s+kalitatif", "Etkin madde", "active_ingredient"),
    (r"^\s*4[.]1\s+terapötik\s+endikasyonlar", "Terapötik endikasyonlar", "indications"),
    (r"^\s*4[.]2\s+pozoloji", "Pozoloji ve uygulama şekli", "dosage"),
    (r"^\s*4[.]3\s+kontrendikasyonlar", "Kontrendikasyonlar", "contraindications"),
    (r"^\s*4[.]4\s+özel\s+kullanım", "Özel kullanım uyarıları", "warnings"),
    (r"^\s*4[.]5\s+diğer\s+tıbbi", "İlaç etkileşimleri", "interactions"),
    (r"^\s*4[.]6\s+gebelik", "Gebelik ve emzirme", "pregnancy"),
    (r"^\s*4[.]7\s+araç", "Araç ve makine kullanımı", "driving"),
    (r"^\s*4[.]8\s+istenmeyen", "İstenmeyen etkiler", "common_side_effects"),
    (r"^\s*4[.]9\s+doz\s+aşımı", "Doz aşımı", "overdose"),
    (r"^\s*6[.]4\s+saklamaya", "Saklama koşulları", "storage"),
)


def clean_extracted_text(raw_text: str) -> str:
    """Remove repeated headers/page numbers while preserving medical content."""

    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
    counts = Counter(line for line in raw_lines if line and len(line) <= 100)
    cleaned: list[str] = []
    for line in raw_lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if counts[line] >= 4 and not re.search(r"(?:mg|ml|uyarı|doz)", line, re.I):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def parse_leaflet_sections(raw_text: str, document_type: str) -> list[dict[str, str]]:
    """Split a KÜB/KT by official heading structure."""

    text = clean_extracted_text(raw_text)
    patterns = _KUB_HEADINGS if document_type.upper() == "KUB" else _KT_HEADINGS
    matches: list[tuple[int, int, str, str]] = []
    for pattern, section, chunk_type in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            matches.append((match.start(), match.end(), section, chunk_type))
    matches.sort(key=lambda item: item[0])

    sections: list[dict[str, str]] = []
    for index, (start, _, section, chunk_type) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.extend(_split_side_effect_section(section, chunk_type, body))

    if not sections and text:
        sections.append({"section": "Belge metni", "chunk_type": "general", "text": text})

    # Leaflets usually repeat all headings in a table of contents. Keep the
    # longest occurrence for each semantic section so TOC fragments do not
    # become duplicate chunks.
    order: list[tuple[str, str]] = []
    best: dict[tuple[str, str], dict[str, str]] = {}
    for item in sections:
        key = (item["section"], item["chunk_type"])
        if key not in best:
            order.append(key)
            best[key] = item
        elif len(item["text"]) > len(best[key]["text"]):
            best[key] = item
    return [best[key] for key in order]


def create_document_chunks(
    medicine_name: str,
    sections: Iterable[dict[str, str]],
    *,
    min_chars: int = SYNC_CHUNK_MIN_CHARS,
    max_chars: int = SYNC_CHUNK_MAX_CHARS,
) -> list[dict[str, str]]:
    """Create 300–800-token-like chunks without mixing document sections."""

    chunks: list[dict[str, str]] = []
    for section in sections:
        heading = section["section"]
        chunk_type = section["chunk_type"]
        for part in _semantic_parts(section["text"], min_chars=min_chars, max_chars=max_chars):
            chunk_text = f"İlaç: {medicine_name}\nBölüm: {heading}\n{part}"
            chunks.append(
                {
                    "section": heading,
                    "chunk_type": chunk_type,
                    "chunk_text": chunk_text,
                    "chunk_hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                }
            )
    return chunks


def _split_side_effect_section(
    section: str, chunk_type: str, body: str
) -> list[dict[str, str]]:
    if chunk_type != "common_side_effects":
        return [{"section": section, "chunk_type": chunk_type, "text": body}]

    serious_start = re.search(
        r"aşağıdakilerden\s+herhangi\s+birini.*?(?:hemen|derhal)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    frequency_start = re.search(
        r"(?:klinik\s+çalışmalarda|çok\s+yaygın\s*:?)",
        body,
        flags=re.IGNORECASE,
    )
    if not serious_start or not frequency_start or frequency_start.start() <= serious_start.start():
        return [{"section": section, "chunk_type": chunk_type, "text": body}]

    serious = body[serious_start.start() : frequency_start.start()].strip()
    common = (body[: serious_start.start()] + "\n" + body[frequency_start.start() :]).strip()
    result: list[dict[str, str]] = []
    if serious:
        result.append(
            {"section": f"{section} — ciddi", "chunk_type": "serious_side_effects", "text": serious}
        )
    if common:
        result.append({"section": section, "chunk_type": chunk_type, "text": common})
    return result


def _semantic_parts(text: str, *, min_chars: int, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9•])|\n{2,}", text)
        if item.strip()
    ]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars and len(current) >= min_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        if parts and len(current) < min_chars and len(parts[-1]) + len(current) + 1 <= max_chars:
            parts[-1] = f"{parts[-1]} {current}"
        else:
            parts.append(current)
    return parts
