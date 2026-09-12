"""Section-aware parser for İlacabak's accessible prospectus HTML."""

from __future__ import annotations

import re
from collections import defaultdict

from bs4 import BeautifulSoup


_HEADINGS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"^END[İI]KASYONLARI?\b\s*:?[ \t]*(.*)$", re.I), "Endikasyonları", "indications"),
    (re.compile(r"^KONTREND[İI]KASYONLARI?\b\s*:?[ \t]*(.*)$", re.I), "Kontrendikasyonları", "contraindications"),
    (
        re.compile(r"^(?:UYARILAR(?:\s*/\s*ÖNLEMLER)?|ÖZEL KULLANIM UYARILARI)\b\s*:?[ \t]*(.*)$", re.I),
        "Uyarılar / Önlemler",
        "warnings",
    ),
    (
        re.compile(r"^(?:YAN ETK[İI]LER|ADVERS ETK[İI]LER)(?:\s*/\s*(?:YAN ETK[İI]LER|ADVERS ETK[İI]LER))?\b\s*:?[ \t]*(.*)$", re.I),
        "Yan etkiler / Advers etkiler",
        "common_side_effects",
    ),
    (
        re.compile(r"^(?:[İI]LAÇ ETK[İI]LEŞ[İI]MLER[İI](?: VE D[İI]ĞER ETK[İI]LEŞ[İI]MLER)?|ETK[İI]LEŞ[İI]MLER)\b\s*:?[ \t]*(.*)$", re.I),
        "İlaç etkileşimleri",
        "interactions",
    ),
    (
        re.compile(r"^(?:KULLANIM ŞEKL[İI] VE DOZU|POZOLOJ[İI] VE UYGULAMA ŞEKL[İI])\b\s*:?[ \t]*(.*)$", re.I),
        "Kullanım Şekli ve Dozu",
        "dosage",
    ),
)


def parse_prospektus_html(html: str) -> list[dict[str, str]]:
    """Extract only named medical sections; never index navigation or summaries."""

    soup = BeautifulSoup(html, "html.parser")
    candidates = soup.select("#iceriksol .kutucuksol, #iceriksol, main, article")
    containers = [item for item in candidates if _contains_medical_heading(item.get_text("\n"))]
    if not containers:
        return []
    container = min(containers, key=lambda item: len(item.get_text("\n", strip=True)))
    text = container.get_text("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    sections: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    body: list[str] = []
    for line in lines:
        if not line:
            if body and body[-1] != "":
                body.append("")
            continue
        heading = _match_heading(line)
        if heading is not None:
            if current is not None:
                _append_section(sections, current, body)
            section, chunk_type, remainder = heading
            current = {"section": section, "chunk_type": chunk_type}
            body = [remainder] if remainder else []
        elif current is not None:
            body.append(line)
    if current is not None:
        _append_section(sections, current, body)

    # Repeated headings occur in some archived leaflets. Merge them by type
    # without crossing semantic section boundaries.
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for section in sections:
        key = (section["section"], section["chunk_type"])
        if key not in grouped:
            order.append(key)
        if section["text"] not in grouped[key]:
            grouped[key].append(section["text"])
    return [
        {"section": section, "chunk_type": chunk_type, "text": "\n\n".join(grouped[(section, chunk_type)])}
        for section, chunk_type in order
    ]


def _contains_medical_heading(text: str) -> bool:
    return any(_match_heading(line.strip()) is not None for line in text.splitlines())


def _match_heading(line: str) -> tuple[str, str, str] | None:
    for pattern, section, chunk_type in _HEADINGS:
        match = pattern.match(line)
        if match:
            return section, chunk_type, match.group(1).strip()
    return None


def _append_section(
    sections: list[dict[str, str]], current: dict[str, str], body: list[str]
) -> None:
    text = "\n".join(body).strip()
    text = re.split(
        r"\bBu sitede ve verilen linklerdeki bilgilerin\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if text:
        sections.append({**current, "text": text})
