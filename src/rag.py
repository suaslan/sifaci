"""Safety-constrained retrieval-augmented medicine answers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from config import (
    CATALOG_NOT_FOUND_RESPONSE,
    MISSING_INFORMATION_RESPONSE,
    RAG_SYSTEM_PROMPT,
    RETRIEVAL_TOP_K,
    SAFETY_DISCLAIMER,
    SOURCE_SECTION_MARKER,
)
from src.foundry_client import FoundryLocalError, complete_chat
from src.intents import (
    INTENT_CHUNK_TYPES,
    INTENT_LABELS,
    chunk_supports_intent,
    detect_intent,
    normalize_text,
)
from src.medicine_resolution import medicine_signature
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

_GENERIC_QUERY_TOKENS = {
    "bu",
    "ciddi",
    "doz",
    "etken",
    "etkiler",
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
    "aktarilir",
    "belge",
    "belgeler",
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
    "ilaç bilgi asistanısın",
    "kaynakla sınırlandırılmış",
    "zorunlu kurallar",
)
_FALLBACK_ITEM_LIMIT = 12

_CHUNK_TYPE_LABELS = {
    "general": "Genel ürün bilgisi",
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
    "pregnancy": "Gebelik",
    "breastfeeding": "Emzirme",
    "lactation": "Emzirme",
    "driving": "Araç ve makine kullanımı",
    "overdose": "Doz aşımı",
    "storage": "Saklama koşulları",
    "special_populations": "Özel popülasyonlar",
    "missed_dose": "Unutulan doz",
    "stopping_treatment": "Tedavinin bırakılması",
}


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

    if retrieval_function is not get_top_chunks and not _custom_retrieval_matches_question(
        question, chunks
    ):
        if debug_trace is not None:
            debug_trace["model_call_block_reason"] = "Özel retrieval sonucu sorudaki ilaçla eşleşmedi."
        return _finalize(MISSING_INFORMATION_RESPONSE, [])

    grounded_fallback = _grounded_field_fallback(question, chunks)
    # Direct prospectus questions are safer and much faster when rendered from
    # the already-resolved local evidence. General/free-form questions still
    # use Foundry Local for composition, and injected test/model functions keep
    # exercising the complete model-safety path.
    if (
        retrieval_function is get_top_chunks
        and chat_function is complete_chat
        and grounded_fallback
    ):
        if debug_trace is not None:
            debug_trace["model_called"] = False
            debug_trace["model_call_block_reason"] = "deterministic_direct_answer"
            debug_trace["response_safety_action"] = (
                "Doğrudan prospektüs sorusu yerel kaynak kanıtından oluşturuldu."
            )
        return _finalize(grounded_fallback, chunks)

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
        answer = grounded_fallback or MISSING_INFORMATION_RESPONSE
        used_grounded_fallback = bool(grounded_fallback)
        if debug_trace is not None:
            debug_trace["response_safety_action"] = (
                "Kaynakta bulunmayan tıbbi miktar nedeniyle model yanıtı reddedildi."
            )

    if (
        answer != MISSING_INFORMATION_RESPONSE
        and _has_critical_source(chunks)
        and not _CRITICAL_TERMS.search(answer)
    ):
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
    if state == "medicine_ambiguous":
        return (
            "Birden fazla ürün formu bulundu. Kullanım ve doz bilgileri ürüne "
            "göre değişebileceği için lütfen aşağıdaki işlenmiş ürünlerden tam "
            "olanı seçin."
        )
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
    return tuple(INTENT_CHUNK_TYPES.get(detect_intent(question), ()))


def _normalized_word_tokens(text: str) -> list[str]:
    translated = text.casefold().translate(str.maketrans({"ı": "i", "ş": "s"}))
    decomposed = unicodedata.normalize("NFKD", translated)
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

    intent = detect_intent(question)
    requested_types = _requested_chunk_types(question)
    if intent == "GENERAL" or not requested_types:
        return None

    matched = [
        chunk
        for chunk in chunks
        if str(chunk.get("retrieval_intent") or "") == intent
        or chunk_supports_intent(
            str(chunk.get("chunk_type") or ""),
            f"{chunk.get('section') or ''}\n{chunk.get('chunk_text') or ''}",
            intent,
        )
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
        items: list[str] = []
        for chunk in matched:
            if str(chunk.get("chunk_type") or "") != chunk_type:
                continue
            body = _source_body(str(chunk.get("chunk_text") or "")).strip()
            if body:
                items.extend(_source_items(body))
        unique_items = list(dict.fromkeys(items))[:_FALLBACK_ITEM_LIMIT]
        if unique_items:
            label = _CHUNK_TYPE_LABELS.get(chunk_type, INTENT_LABELS.get(intent, chunk_type))
            sections.append(f"{label}:\n" + "\n".join(f"- {item}" for item in unique_items))
    if not sections:
        items = []
        for chunk in matched:
            body = _source_body(str(chunk.get("chunk_text") or "")).strip()
            if body:
                items.extend(_source_items(body))
        unique_items = list(dict.fromkeys(items))[:_FALLBACK_ITEM_LIMIT]
        if unique_items:
            sections.append(
                f"{INTENT_LABELS.get(intent, 'Kayıtlı kaynak bilgisi')}:\n"
                + "\n".join(f"- {item}" for item in unique_items)
            )
    if not sections:
        return None
    answer = f"{heading} için kayıtlı kaynak bilgileri:\n\n" + "\n\n".join(sections)
    if intent in {"USAGE", "DOSAGE", "FREQUENCY", "ROUTE_OF_ADMINISTRATION", "MISSED_DOSE", "STOPPING_TREATMENT"}:
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

    normalized = " ".join(_normalized_word_tokens(answer))
    if len(answer) > 1_500:
        return True
    if any(
        " ".join(_normalized_word_tokens(phrase)) in normalized
        for phrase in _UNHELPFUL_MODEL_PHRASES
    ):
        return True
    if any(
        marker in normalized
        for marker in ("retrieved context", "belge doğrulama kodu", "[kaynak ")
    ):
        return True

    all_answer_tokens = {token for token in _normalized_word_tokens(answer) if len(token) >= 4}
    all_evidence_tokens = {
        token
        for chunk in chunks
        for token in _normalized_word_tokens(_source_body(str(chunk.get("chunk_text") or "")))
        if len(token) >= 4
    }
    if all_answer_tokens & all_evidence_tokens:
        return False

    answer_tokens = {
        token
        for token in _normalized_word_tokens(answer)
        if len(token) >= 4 and token not in _GENERIC_ANSWER_TOKENS
    }
    if not answer_tokens:
        return True

    evidence_tokens = {
        token
        for chunk in chunks
        for token in _normalized_word_tokens(
            _source_body(str(chunk.get("chunk_text") or ""))
        )
        if len(token) >= 4 and token not in _GENERIC_ANSWER_TOKENS
    }
    return not bool(answer_tokens & evidence_tokens)


def _custom_retrieval_matches_question(
    question: str, chunks: Sequence[Mapping[str, Any]]
) -> bool:
    """Protect injected/test retrievers; production uses canonical medicine ids."""

    question_tokens = _normalized_word_tokens(question)
    if not question_tokens:
        return False
    generic = _GENERIC_QUERY_TOKENS | {"kullanma", "kullanmali", "etkilesir", "hamilelikte", "emzirirken"}
    likely_names = [token for token in question_tokens if len(token) >= 3 and token not in generic]
    if not likely_names:
        return True
    normalized_question = normalize_text(question)
    compact_question = normalized_question.replace(" ", "")
    for chunk in chunks:
        signature = medicine_signature(str(chunk.get("medicine_name") or ""))
        normalized_name = str(signature["normalized_name"])
        if normalized_name and normalized_name in normalized_question:
            return True
        brand = str(signature["brand"])
        if not brand:
            continue
        if brand in normalized_question or brand.replace(" ", "") in compact_question:
            return True
        brand_token = brand.split()[-1]
        for token in likely_names:
            if token == brand_token or token.startswith(brand_token):
                return True
            if len(brand_token) >= 4 and len(token) >= 4:
                from difflib import SequenceMatcher

                if SequenceMatcher(None, token, brand_token).ratio() >= 0.8:
                    return True
    return False


def _source_items(source_text: str) -> list[str]:
    """Turn noisy PDF lines into readable, bounded source-backed bullets."""

    ignored_prefixes = (
        "belge doğrulama kodu",
        "belge takip adresi",
        "www.titck.gov.tr",
    )
    ignored_lines = (
        "yan etkilerin raporlanması",
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
        if page_number.fullmatch(line) or any(
            lowered.startswith(part) for part in ignored_prefixes
        ) or any(lowered.strip(" .:") == part for part in ignored_lines):
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

    items = [
        item
        for item in items
        if len(item) >= 3 and not re.fullmatch(r"\d+\s*-?", item)
    ]
    if not items:
        compact = " ".join(
            " ".join(line.split())
            for line in source_text.splitlines()
            if line.strip()
            and not any(
                line.casefold().startswith(part) for part in ignored_prefixes
            )
            and not any(
                line.casefold().strip(" .:") == part for part in ignored_lines
            )
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
