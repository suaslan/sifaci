"""Safety-constrained retrieval-augmented medicine answers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from config import (
    CATALOG_NOT_FOUND_RESPONSE,
    MISSING_INFORMATION_RESPONSE,
    RAG_SYSTEM_PROMPT,
    RETRIEVAL_TOP_K,
    SAFETY_DISCLAIMER,
    SOURCE_SECTION_MARKER,
)
from src.foundry_client import complete_chat
from src.retrieval import get_top_chunks


DISCLAIMER = SAFETY_DISCLAIMER
SYSTEM_PROMPT = RAG_SYSTEM_PROMPT

_PERSONALIZED_PATTERNS = (
    r"\b\d+(?:[.,]\d+)?\s*(?:kg|kilo(?:yum|sun|dur|luk)?)\b",
    r"\b(?:çocuğum|çocuğuma|çocuğumun|bebeğim|bebeğime|bebek|çocuk)\b.*\b(?:doz|kaç|ne kadar|ver)",
    r"\b(?:kaç tane|kaç tablet|ne kadar)\b.*\b(?:almalıyım|alayım|içmeliyim|içeyim|vereyim)",
    r"\bdoz(?:u|umu)?\b.*\b(?:iki kat|artır|arttir|azalt|değiştir|degistir)",
    r"\b(?:benim için|bana uygun|kullanmam uygun mu)\b",
    r"\b(?:kaç gün|ne kadar süre)\b.*\b(?:kullanmalıyım|kullanayım|vereyim)",
    r"\b(?:birlikte|aynı anda)\b.*\b(?:alabilir miyim|kullanabilir miyim|içebilir miyim)",
)
_CRITICAL_TERMS = re.compile(
    r"\b(?:acil|derhal|hemen|hayati|ciddi|ambulans|hastaneye)\b", re.IGNORECASE
)
_MEDICAL_QUANTITY = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"
    r"(mikrogram|miligram|gram|mcg|µg|mg|g|mililitre|ml|tablet|kapsül|kapsul|"
    r"damla|ölçek|olcek|doz|saat|gün|gun|hafta)\b",
    re.IGNORECASE,
)

_QUESTION_INTENTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\b(?:yan\s+etki|yan\s+etkiler|istenmeyen\s+etki)", re.IGNORECASE),
        ("common_side_effects", "side_effects", "serious_side_effects", "warnings"),
    ),
    (
        re.compile(r"\b(?:ciddi|ağır)\b.*\b(?:yan\s+etki|reaksiyon)", re.IGNORECASE),
        ("serious_side_effects", "warnings"),
    ),
    (
        re.compile(r"\b(?:etken|etkin|aktif)\s+madde", re.IGNORECASE),
        ("active_ingredient",),
    ),
    (
        re.compile(r"\b(?:ne\s+için|hangi\s+durum|endikasyon)", re.IGNORECASE),
        ("indications",),
    ),
    (
        re.compile(r"\b(?:nasıl\s+kullan|kullanım\s+şekli)", re.IGNORECASE),
        ("usage", "route_of_administration", "dosage", "frequency"),
    ),
    (
        re.compile(r"\b(?:uygulama\s+yolu|ağızdan|oral|damardan)", re.IGNORECASE),
        ("route_of_administration",),
    ),
    (
        re.compile(r"\b(?:sıklık|kaç\s+kez|günde\s+kaç)", re.IGNORECASE),
        ("frequency",),
    ),
    (re.compile(r"\bdoz", re.IGNORECASE), ("dosage",)),
    (
        re.compile(r"\b(?:kontrendikasyon|kimler\s+kullanma|kullanılmamalı)", re.IGNORECASE),
        ("contraindications",),
    ),
    (
        re.compile(r"\b(?:etkileşim|birlikte\s+kullan)", re.IGNORECASE),
        ("interactions",),
    ),
    (re.compile(r"\b(?:uyarı|dikkat)", re.IGNORECASE), ("warnings",)),
    (re.compile(r"\b(?:saklama|nasıl\s+saklan)", re.IGNORECASE), ("storage",)),
)

_CHUNK_TYPE_LABELS = {
    "active_ingredient": "Etken madde",
    "indications": "Kullanım alanı",
    "usage": "Genel kullanım bilgisi",
    "dosage": "Genel doz bilgisi",
    "frequency": "Genel kullanım sıklığı",
    "route_of_administration": "Uygulama yolu",
    "side_effects": "Kaynakta bildirilen yan etkiler",
    "common_side_effects": "Kaynakta bildirilen yan etkiler",
    "serious_side_effects": "Ciddi yan etkiler",
    "warnings": "Önemli uyarılar",
    "contraindications": "Kontrendikasyonlar",
    "interactions": "Etkileşimler",
    "storage": "Saklama koşulları",
}

_MEDICINE_FORM_WORDS = {
    "ampul",
    "film",
    "flakon",
    "kapsul",
    "mg",
    "ml",
    "saşe",
    "surup",
    "tablet",
}
_TURKISH_NAME_SUFFIXES = {
    "i",
    "ı",
    "u",
    "ü",
    "in",
    "ın",
    "un",
    "ün",
    "a",
    "e",
    "da",
    "de",
    "dan",
    "den",
    "la",
    "le",
}
_GENERIC_QUERY_TOKENS = {
    "bu",
    "ciddi",
    "doz",
    "etken",
    "etkileri",
    "gunde",
    "hangi",
    "ilac",
    "ilacin",
    "ilacinin",
    "kac",
    "kimler",
    "kullanim",
    "kullanilir",
    "maddesi",
    "nasil",
    "ne",
    "nedir",
    "nelerdir",
    "uyarilar",
    "yan",
}

_GENERIC_ANSWER_TOKENS = _GENERIC_QUERY_TOKENS | {
    "bilgi",
    "bilgilere",
    "bulunabilmeniz",
    "gore",
    "icin",
    "kaynak",
    "kaynakta",
    "kullanilamadi",
    "kullanilabilir",
    "mevcut",
    "tabaninda",
    "veri",
}
_UNHELPFUL_MODEL_PHRASES = (
    "veri tabanında bulunabilmeniz",
    "kaynakta veri tabanında",
    "bu bilgilere göre kaynakta",
)
_FALLBACK_ITEM_LIMIT = 12


def answer_query(
    user_question: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    retrieval_function: Callable[..., list[dict[str, Any]]] = get_top_chunks,
    chat_function: Callable[[Sequence[Mapping[str, str]]], str] = complete_chat,
    debug_trace: MutableMapping[str, Any] | None = None,
) -> str:
    """Answer from retrieved medicine records while enforcing safety rules."""

    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("user_question cannot be empty")
    question = user_question.strip()
    retrieval_trace = debug_trace
    if retrieval_trace is None and retrieval_function is get_top_chunks:
        retrieval_trace = {}
    if retrieval_trace is not None:
        retrieval_trace.update(
            {
                "user_question": question,
                "retrieved_context": "",
                "model_called": False,
                "model_call_block_reason": None,
            }
        )
        chunks = retrieval_function(
            question,
            top_k=top_k,
            debug_trace=retrieval_trace,
        )
        retrieval_trace.setdefault("top_chunks", [dict(chunk) for chunk in chunks])
    else:
        chunks = retrieval_function(question, top_k=top_k)

    if not chunks:
        if retrieval_trace is not None:
            fallback_reason = str(
                retrieval_trace.get("fallback_reason") or "retrieval_empty"
            )
            retrieval_trace["model_call_block_reason"] = fallback_reason
        return _finalize(_retrieval_fallback(retrieval_trace), [])

    context = build_retrieved_context(chunks)
    if debug_trace is not None:
        debug_trace["retrieved_context"] = context

    if _is_personalized_treatment_request(question):
        if debug_trace is not None:
            debug_trace["model_call_block_reason"] = "Kişiye özel tedavi/doz isteği engellendi."
        return _personalized_safe_answer(chunks)

    if not _question_matches_retrieved_medicine(question, chunks):
        if debug_trace is not None:
            debug_trace["model_call_block_reason"] = (
                "Sorudaki ilaç adı retrieved sonuçlardaki kayıtlı ilaçlarla eşleşmedi."
            )
        return _finalize(MISSING_INFORMATION_RESPONSE, [])

    safety_notes = _build_safety_notes(chunks)
    user_prompt = (
        f"KULLANICI SORUSU:\n{question}\n\n"
        f"GÜVENLİK İŞARETLERİ:\n{safety_notes}\n\n"
        f"RETRIEVED CONTEXT:\n{context}"
    )
    if debug_trace is not None:
        debug_trace["llm_system_prompt"] = SYSTEM_PROMPT
        debug_trace["llm_user_prompt"] = user_prompt
        debug_trace["model_called"] = True
    grounded_fallback = _grounded_field_fallback(question, chunks)
    try:
        answer = chat_function(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        ).strip()
    except Exception:
        if not grounded_fallback:
            raise
        answer = grounded_fallback
        if debug_trace is not None:
            debug_trace["response_safety_action"] = (
                "Yerel model hatasında eşleşen kaynak alanları doğrudan gösterildi."
            )
    if not answer:
        answer = MISSING_INFORMATION_RESPONSE

    # Very small local chat models occasionally claim that information is
    # absent even when retrieval returned the exact requested field. Preserve
    # the LLM-based flow, but recover deterministically from the retrieved
    # evidence instead of presenting a false "not found" result.
    used_grounded_fallback = False
    if answer == grounded_fallback and grounded_fallback:
        used_grounded_fallback = True
    elif _looks_like_missing_information(answer) and grounded_fallback:
        answer = grounded_fallback
        used_grounded_fallback = True
        if debug_trace is not None:
            debug_trace["response_safety_action"] = (
                "Modelin hatalı bilgi-yok yanıtı, eşleşen kaynak alanlarıyla değiştirildi."
            )
    elif (
        grounded_fallback
        and not _contains_unsupported_quantity(answer, chunks)
        and _model_answer_needs_grounded_fallback(answer, chunks)
    ):
        answer = grounded_fallback
        used_grounded_fallback = True
        if debug_trace is not None:
            debug_trace["response_safety_action"] = (
                "Modelin anlamsız, temelsiz veya aşırı uzun yanıtı temiz kaynak "
                "özetiyle değiştirildi."
            )

    # A generated dose/duration quantity not present in the evidence is a
    # deterministic safety failure, regardless of the model's wording.
    if not used_grounded_fallback and _contains_unsupported_quantity(answer, chunks):
        answer = MISSING_INFORMATION_RESPONSE
        if debug_trace is not None:
            debug_trace["response_safety_action"] = (
                "Kaynakta bulunmayan tıbbi miktar nedeniyle model yanıtı reddedildi."
            )

    if _has_critical_source(chunks) and not _CRITICAL_TERMS.search(answer):
        answer = (
            "Önemli güvenlik uyarısı: Kaynaklarda ciddi yan etki veya acil "
            "değerlendirme uyarısı bulunmaktadır.\n\n" + answer
        )
    return _finalize(answer, chunks)


def _retrieval_fallback(trace: Mapping[str, Any] | None) -> str:
    if not trace:
        return MISSING_INFORMATION_RESPONSE
    state = str(trace.get("retrieval_state") or "")
    medicine = str(trace.get("detected_medicine") or "İlaç").strip()
    if state == "medicine_not_found":
        return CATALOG_NOT_FOUND_RESPONSE
    if state == "rag_not_processed":
        return (
            f"{medicine} kayıtlı ancak KÜB/Kullanma Talimatı henüz yerel "
            "bilgi tabanına işlenmemiş."
        )
    if state in {"section_missing", "low_similarity"}:
        return (
            f"{medicine} için kaynak mevcut ancak sorulan bilgi ilgili "
            "belgelerde bulunamadı."
        )
    return MISSING_INFORMATION_RESPONSE


def build_retrieved_context(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Render clearly separated, untrusted source blocks for the chat model."""

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source_type = chunk.get("source_type") or "Belirtilmemiş"
        source_date = chunk.get("source_date") or "Belirtilmemiş"
        blocks.append(
            f"[KAYNAK {index}]\n"
            f"İlaç: {chunk.get('medicine_name') or 'Belirtilmemiş'}\n"
            f"Kategori: {chunk.get('chunk_type') or 'Belirtilmemiş'}\n"
            f"Metin: {chunk.get('chunk_text') or ''}\n"
            f"Kaynak: {chunk.get('source_name') or 'Belirtilmemiş'}\n"
            f"Kaynak türü: {source_type}\n"
            f"Kaynak tarihi: {source_date}"
        )
    return "\n\n".join(blocks)


def _is_personalized_treatment_request(question: str) -> bool:
    lowered = question.casefold()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _PERSONALIZED_PATTERNS)


def _requested_chunk_types(question: str) -> tuple[str, ...]:
    """Map a direct Turkish medicine question to stored evidence categories."""

    requested: list[str] = []
    for pattern, chunk_types in _QUESTION_INTENTS:
        if pattern.search(question):
            for chunk_type in chunk_types:
                if chunk_type not in requested:
                    requested.append(chunk_type)
    return tuple(requested)


def _question_matches_retrieved_medicine(
    question: str, chunks: Sequence[Mapping[str, Any]]
) -> bool:
    """Require an exact, suffixed, or fuzzy registered brand mention."""

    question_tokens = _normalized_word_tokens(question)
    if not question_tokens:
        return False

    medicine_candidates: list[str] = []
    for index, token in enumerate(question_tokens):
        if token.startswith("ilac") and index > 0:
            previous = question_tokens[index - 1]
            if previous not in _GENERIC_QUERY_TOKENS:
                medicine_candidates.append(previous)
    first_token = question_tokens[0]
    if first_token not in _GENERIC_QUERY_TOKENS:
        medicine_candidates.append(first_token)
    medicine_candidates = list(dict.fromkeys(medicine_candidates))

    # Generic questions remain usable (especially with a single-record test
    # database). Once a likely product name is present, it must match a
    # retrieved registered brand before the LLM is allowed to answer.
    if not medicine_candidates:
        return True

    brands: list[str] = []
    for chunk in chunks:
        name_tokens = [
            token
            for token in _normalized_word_tokens(str(chunk.get("medicine_name") or ""))
            if len(token) >= 3 and not token.isdigit() and token not in _MEDICINE_FORM_WORDS
        ]
        if name_tokens and name_tokens[0] not in brands:
            brands.append(name_tokens[0])

    for brand in brands:
        for token in medicine_candidates:
            if token == brand:
                return True
            if token.startswith(brand) and token[len(brand) :] in _TURKISH_NAME_SUFFIXES:
                return True
            if len(brand) >= 4 and SequenceMatcher(None, token, brand).ratio() >= 0.8:
                return True
    return False


def _normalized_word_tokens(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.findall(r"\w+", without_marks, flags=re.UNICODE)


def _looks_like_missing_information(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return (
        MISSING_INFORMATION_RESPONSE.casefold() in normalized
        or "bulunmuyor" in normalized
        or "bulunamadı" in normalized
        or "veri tabanında bulunmuyor" in normalized
        or "veritabanında bulunmuyor" in normalized
    )


def _grounded_field_fallback(
    question: str, chunks: Sequence[Mapping[str, Any]]
) -> str | None:
    """Render exact retrieved fields when the local LLM falsely says no data."""

    requested_types = _requested_chunk_types(question)
    if not requested_types:
        return None

    matched = [
        chunk for chunk in chunks if str(chunk.get("chunk_type") or "") in requested_types
    ]
    if not matched:
        return None

    medicine_names: list[str] = []
    for chunk in matched:
        medicine_name = str(chunk.get("medicine_name") or "").strip()
        if medicine_name and medicine_name not in medicine_names:
            medicine_names.append(medicine_name)
    heading = ", ".join(medicine_names) or "İlaç"

    sections: list[str] = []
    for chunk_type in requested_types:
        bodies = [
            _source_body(str(chunk.get("chunk_text") or "")).strip()
            for chunk in matched
            if chunk.get("chunk_type") == chunk_type
        ]
        bodies = [body for body in bodies if body]
        if not bodies:
            continue
        label = _CHUNK_TYPE_LABELS.get(chunk_type, chunk_type)
        items: list[str] = []
        for body in bodies:
            items.extend(_source_items(body))
        unique_items = list(dict.fromkeys(items))[:_FALLBACK_ITEM_LIMIT]
        if not unique_items:
            continue
        rendered = "\n".join(f"- {item}" for item in unique_items)
        sections.append(f"{label}:\n{rendered}")

    if not sections:
        return None
    answer = f"{heading} için kayıtlı kaynak bilgileri:\n\n" + "\n\n".join(sections)
    if {"usage", "dosage", "frequency"} & set(requested_types):
        answer += (
            "\n\nBu kullanım bilgisi ürün/form ve kullanım amacına göre değişebilir; "
            "kişisel doz önerisi değildir."
        )
    return answer


def _personalized_safe_answer(chunks: Sequence[Mapping[str, Any]]) -> str:
    answer = (
        "Kişiye özel doz, kullanım süresi veya ilaç kombinasyonu hesaplayamam. "
        "Bu karar için doktorunuza veya eczacınıza danışın."
    )
    general_types = {
        "dosage",
        "frequency",
        "interactions",
        "route_of_administration",
        "usage",
        "warnings",
    }
    general_chunks = [
        chunk for chunk in chunks if chunk.get("chunk_type") in general_types
    ][:2]
    if general_chunks:
        answer += (
            "\n\nVeri tabanındaki genel ürün bilgisi (kişisel öneri değildir):\n"
            + "\n\n".join(str(chunk.get("chunk_text") or "") for chunk in general_chunks)
        )
    else:
        answer += f"\n\n{MISSING_INFORMATION_RESPONSE}"
    if _has_critical_source(chunks):
        answer = (
            "Önemli güvenlik uyarısı: Kaynaklarda ciddi yan etki veya acil "
            "değerlendirme uyarısı bulunmaktadır.\n\n" + answer
        )
    return _finalize(answer, chunks)


def _build_safety_notes(chunks: Sequence[Mapping[str, Any]]) -> str:
    critical = "EVET" if _has_critical_source(chunks) else "HAYIR"
    overlapping = "EVET" if _has_overlapping_categories(chunks) else "HAYIR"
    return (
        f"Ciddi/acil uyarı içeren kaynak var: {critical}.\n"
        f"Aynı ilaç ve kategoride karşılaştırılması gereken birden fazla kaynak var: "
        f"{overlapping}. Farklılık varsa bunu çelişki olarak açıkla; yoksa çelişki üretme."
    )


def _has_critical_source(chunks: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        chunk.get("chunk_type") == "serious_side_effects"
        or (
            chunk.get("chunk_type") == "warnings"
            and _CRITICAL_TERMS.search(str(chunk.get("chunk_text") or "")) is not None
        )
        for chunk in chunks
    )


def _has_overlapping_categories(chunks: Sequence[Mapping[str, Any]]) -> bool:
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (
            str(chunk.get("medicine_name") or "").casefold(),
            str(chunk.get("chunk_type") or "").casefold(),
        )
        if key in seen:
            return True
        seen.add(key)
    return False


def _contains_unsupported_quantity(
    answer: str, chunks: Sequence[Mapping[str, Any]]
) -> bool:
    answer_quantities = set(_normalized_quantities(answer))
    # Medicine names often contain a strength (for example "500 mg"). A
    # product-name strength alone is not evidence for a dose, so generated
    # quantities are compared only with the substantive source text.
    evidence_texts = [_source_body(str(chunk.get("chunk_text") or "")) for chunk in chunks]
    evidence_quantities = set(_normalized_quantities("\n".join(evidence_texts)))
    return bool(answer_quantities - evidence_quantities)


def _source_body(chunk_text: str) -> str:
    lines = chunk_text.splitlines()
    while lines and _normalized_header(lines[0]) in {"ilac", "kategori", "bolum"}:
        lines.pop(0)
    return "\n".join(lines)


def _model_answer_needs_grounded_fallback(
    answer: str, chunks: Sequence[Mapping[str, Any]]
) -> bool:
    """Reject output that does not communicate a concrete retrieved fact."""

    normalized = " ".join(answer.casefold().split())
    if len(answer) > 1_500:
        return True
    if any(phrase in normalized for phrase in _UNHELPFUL_MODEL_PHRASES):
        return True
    if any(
        marker in normalized
        for marker in ("retrieved context", "belge doğrulama kodu", "[kaynak ")
    ):
        return True

    answer_tokens = {
        token
        for token in _normalized_word_tokens(answer)
        if len(token) >= 4 and token not in _GENERIC_ANSWER_TOKENS
    }
    return not answer_tokens


def _source_items(source_text: str) -> list[str]:
    """Turn noisy PDF lines into readable, bounded source-backed bullets."""

    ignored_fragments = (
        "belge doğrulama kodu",
        "belge takip adresi",
        "yan etkilerin raporlanması",
        "www.titck.gov.tr",
        "tüm ilaçlar gibi",
        "yan etkiler aşağıdaki kategorilerde",
    )
    bullet = re.compile(r"^[\s\-•▪◦]+")
    page_number = re.compile(r"^\d+\s*/\s*\d+$")
    items: list[str] = []
    current = ""

    for raw_line in source_text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lowered = line.casefold()
        if page_number.fullmatch(line) or any(part in lowered for part in ignored_fragments):
            continue
        starts_item = bullet.match(line) is not None
        cleaned = bullet.sub("", line).strip()
        if not cleaned:
            continue
        if starts_item:
            if current:
                items.append(current)
            current = cleaned
        elif current:
            current = f"{current} {cleaned}"

    if current:
        items.append(current)

    if not items:
        compact = " ".join(
            " ".join(line.split())
            for line in source_text.splitlines()
            if line.strip()
            and not any(part in line.casefold() for part in ignored_fragments)
        )
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        items = [sentence.strip() for sentence in sentences if sentence.strip()]

    return [item for item in items if len(item) >= 3]


def _normalized_header(line: str) -> str:
    header = line.partition(":")[0].strip().casefold()
    decomposed = unicodedata.normalize("NFKD", header)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalized_quantities(text: str) -> list[tuple[str, str]]:
    unit_aliases = {
        "mikrogram": "mcg",
        "µg": "mcg",
        "miligram": "mg",
        "gram": "g",
        "mililitre": "ml",
        "kapsul": "kapsül",
        "olcek": "ölçek",
        "gun": "gün",
    }
    quantities = []
    for number, unit in _MEDICAL_QUANTITY.findall(text.casefold()):
        normalized_number = number.replace(",", ".").lstrip("0") or "0"
        normalized_unit = unit_aliases.get(unit, unit)
        quantities.append((normalized_number, normalized_unit))
    return quantities


def _finalize(answer: str, chunks: Sequence[Mapping[str, Any]]) -> str:
    sources = []
    for chunk in chunks:
        source = str(chunk.get("source_name") or "Belirtilmemiş")
        if source not in sources:
            sources.append(source)
    source_line = ", ".join(sources) if sources else "Bulunamadı"
    return f"{answer.strip()}{SOURCE_SECTION_MARKER}{source_line}\n\n{DISCLAIMER}"
