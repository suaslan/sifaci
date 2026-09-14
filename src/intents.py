"""Shared Turkish question intents for retrieval and grounded rendering."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz


INTENT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SERIOUS_SIDE_EFFECTS", ("ciddi yan etki", "ağır yan etki", "acil yan etki")),
    ("MISSED_DOSE", ("dozu unuttum", "kullanmayı unut", "almayı unut", "unutulan doz")),
    ("STOPPING_TREATMENT", ("bırakırsam", "bırakınca", "tedaviyi bırak", "kullanmayı bırak")),
    ("BREASTFEEDING", ("emzirirken", "emzirme", "emzirme dönem", "laktasyon")),
    ("PREGNANCY", ("hamilelik", "hamileyken", "hamile", "gebelik", "gebe")),
    ("DRIVING", ("araç kullan", "araba kullan", "makine kullan", "sürüş")),
    ("OVERDOSE", ("fazla alın", "fazla kullan", "doz aşımı", "aşırı doz")),
    ("SIDE_EFFECTS", ("yan etki", "istenmeyen etki", "advers etki")),
    ("SPECIAL_POPULATIONS", ("özel popülasyon", "yaşlılarda", "böbrek yetmez", "karaciğer yetmez", "çocuklarda")),
    ("FREQUENCY", ("günde kaç", "kaç kez", "kaç defa", "kullanım sıklığı", "sıklık")),
    ("ROUTE_OF_ADMINISTRATION", ("uygulama yolu", "hangi yolla", "ağızdan", "oral yolla", "oral olarak", "damardan", "kas içine")),
    ("DOSAGE", ("doz", "pozoloji", "kaç mg")),
    ("USAGE", ("nasıl kullan", "nasıl alın", "nasıl iç", "kullanım şekli", "aç karnına", "tok karnına", "yemekten önce", "yemekten sonra")),
    ("INDICATION", ("ne için", "ne işe yarar", "endikasyon", "hangi durumda", "hangi amaçla", "kas gevşet")),
    ("ACTIVE_INGREDIENT", ("etken madde", "etkin madde", "aktif madde")),
    ("CONTRAINDICATION", ("kontrendikasyon", "kimler kullanmamalı", "kullanılmamalı")),
    ("INTERACTION", ("etkileşim", "hangi ilaçlarla", "birlikte kullan")),
    ("WARNING", ("uyarı", "dikkat", "önlem")),
    ("STORAGE", ("saklama", "nasıl saklan", "saklanmalı")),
    ("WHAT_IS", ("nedir", "hakkında bilgi", "hangi ilaç")),
)


INTENT_CHUNK_TYPES: dict[str, tuple[str, ...]] = {
    "WHAT_IS": ("general", "indications", "active_ingredient"),
    "ACTIVE_INGREDIENT": ("active_ingredient",),
    "INDICATION": ("indications",),
    "USAGE": ("usage", "route_of_administration", "dosage", "frequency"),
    "DOSAGE": ("dosage", "usage", "frequency"),
    "FREQUENCY": ("frequency", "dosage", "usage"),
    "ROUTE_OF_ADMINISTRATION": ("route_of_administration", "usage", "dosage"),
    "SIDE_EFFECTS": ("common_side_effects", "side_effects", "serious_side_effects"),
    "SERIOUS_SIDE_EFFECTS": ("serious_side_effects", "warnings", "common_side_effects", "side_effects"),
    "CONTRAINDICATION": ("contraindications",),
    "WARNING": ("warnings",),
    "INTERACTION": ("interactions",),
    "PREGNANCY": ("pregnancy",),
    "BREASTFEEDING": ("breastfeeding", "lactation", "pregnancy"),
    "DRIVING": ("driving",),
    "OVERDOSE": ("overdose",),
    "STORAGE": ("storage",),
    "SPECIAL_POPULATIONS": ("special_populations", "dosage", "warnings"),
    "MISSED_DOSE": ("missed_dose", "usage"),
    "STOPPING_TREATMENT": ("stopping_treatment", "usage", "warnings"),
}


INTENT_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "WHAT_IS": ("nedir", "etkin madde", "ne için kullanılır"),
    "ACTIVE_INGREDIENT": ("etkin madde", "etken madde", "bileşim"),
    "INDICATION": ("endikasyon", "ne için kullanılır"),
    "USAGE": ("nasıl kullanılır", "uygulama şekli", "kullanım şekli"),
    "DOSAGE": ("pozoloji", "doz", "uygun kullanım"),
    "FREQUENCY": ("uygulama sıklığı", "günde", "kez", "defa"),
    "ROUTE_OF_ADMINISTRATION": ("uygulama yolu", "uygulama şekli", "oral", "intravenöz", "intramüsküler"),
    "SIDE_EFFECTS": ("yan etkiler", "istenmeyen etkiler"),
    "SERIOUS_SIDE_EFFECTS": ("ciddi yan etki", "derhal", "hemen doktor", "acil"),
    "CONTRAINDICATION": ("kontrendikasyon", "kullanmayınız", "kullanılmamalı"),
    "WARNING": ("özel kullanım uyarıları", "dikkatli kullanınız", "uyarı"),
    "INTERACTION": ("etkileşim", "diğer ilaçlar", "birlikte kullanım"),
    "PREGNANCY": ("gebelik", "hamilelik", "gebe"),
    "BREASTFEEDING": ("emzirme", "emzirirken", "laktasyon", "anne sütü"),
    "DRIVING": ("araç ve makine", "araç kullan", "makine kullan"),
    "OVERDOSE": ("doz aşımı", "kullanmanız gerekenden fazlasını", "fazla kullan"),
    "STORAGE": ("saklama", "saklanması", "saklayınız"),
    "SPECIAL_POPULATIONS": ("özel popülasyon", "böbrek yetmez", "karaciğer yetmez", "yaşlılarda", "pediyatrik"),
    "MISSED_DOSE": ("unutulan doz", "kullanmayı unut", "almayı unut"),
    "STOPPING_TREATMENT": ("tedavi sonlandır", "kullanmayı bırak", "tedaviyi bırak"),
}


INTENT_LABELS = {
    "WHAT_IS": "Genel ürün bilgisi",
    "ACTIVE_INGREDIENT": "Etken madde",
    "INDICATION": "Kullanım alanı",
    "USAGE": "Genel kullanım bilgisi",
    "DOSAGE": "Genel doz bilgisi",
    "FREQUENCY": "Genel kullanım sıklığı",
    "ROUTE_OF_ADMINISTRATION": "Uygulama yolu",
    "SIDE_EFFECTS": "Kaynakta bildirilen yan etkiler",
    "SERIOUS_SIDE_EFFECTS": "Ciddi yan etkiler",
    "CONTRAINDICATION": "Kontrendikasyonlar",
    "WARNING": "Önemli uyarılar",
    "INTERACTION": "Etkileşimler",
    "PREGNANCY": "Gebelik",
    "BREASTFEEDING": "Emzirme",
    "DRIVING": "Araç ve makine kullanımı",
    "OVERDOSE": "Doz aşımı",
    "STORAGE": "Saklama koşulları",
    "SPECIAL_POPULATIONS": "Özel popülasyonlar",
    "MISSED_DOSE": "Unutulan doz",
    "STOPPING_TREATMENT": "Tedavinin bırakılması",
}


SUGGESTION_SUFFIXES = {
    "SIDE_EFFECTS": "yan etkileri nelerdir?",
    "SERIOUS_SIDE_EFFECTS": "ciddi yan etkileri nelerdir?",
    "FREQUENCY": "günde kaç kez kullanılır?",
    "DOSAGE": "doz bilgisi nedir?",
    "USAGE": "nasıl kullanılır?",
    "ROUTE_OF_ADMINISTRATION": "hangi yolla uygulanır?",
    "INDICATION": "ne için kullanılır?",
    "ACTIVE_INGREDIENT": "etken maddesi nedir?",
    "CONTRAINDICATION": "kimler kullanmamalı?",
    "INTERACTION": "hangi ilaçlarla etkileşir?",
    "WARNING": "uyarıları nelerdir?",
    "PREGNANCY": "hamilelikte kullanılır mı?",
    "BREASTFEEDING": "emzirirken kullanılır mı?",
    "DRIVING": "araç kullanmayı etkiler mi?",
    "OVERDOSE": "fazla alınırsa ne olur?",
    "STORAGE": "nasıl saklanır?",
    "MISSED_DOSE": "kullanmayı unutursam ne yapmalıyım?",
    "STOPPING_TREATMENT": "kullanmayı bırakınca ne olur?",
}


def normalize_text(value: str) -> str:
    translated = value.casefold().translate(str.maketrans({"ı": "i", "ş": "s"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def detect_intent(query: str) -> str:
    """Detect one stable intent vocabulary without an LLM."""

    normalized = normalize_text(query)
    compact = normalized.replace(" ", "")
    if "yanetki" in compact or "istenmeyenetki" in compact or "adversetki" in compact:
        query_tokens = set(normalized.split())
        if query_tokens.intersection({"ciddi", "agir", "acil"}):
            return "SERIOUS_SIDE_EFFECTS"
        return "SIDE_EFFECTS"
    for intent, terms in INTENT_TERMS:
        if any(_contains_normalized_phrase(normalized, term) for term in terms):
            return intent
    tokens = normalized.split()
    for intent, terms in INTENT_TERMS:
        if intent not in {"SIDE_EFFECTS", "USAGE", "INDICATION"}:
            continue
        for term in terms:
            expected = normalize_text(term)
            size = len(expected.split())
            windows = [" ".join(tokens[index:index + size]) for index in range(max(1, len(tokens) - size + 1))]
            if any(fuzz.ratio(window, expected) >= 86 for window in windows):
                return intent
    return "GENERAL"


def _contains_normalized_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    words = normalized_phrase.split()
    if len(words) == 1:
        suffix = r"[a-z0-9]*" if len(words[0]) >= 6 else ""
        pattern = rf"(?:^|\s){re.escape(words[0])}{suffix}(?:\s|$)"
    else:
        prefix = r"\s+".join(re.escape(word) for word in words[:-1])
        pattern = rf"(?:^|\s){prefix}\s+{re.escape(words[-1])}[a-z0-9]*(?:\s|$)"
    return bool(re.search(pattern, normalized_text))


def chunk_supports_intent(chunk_type: str, text: str, intent: str) -> bool:
    """Accept exact semantic sections or broad sections containing the subject."""

    if intent == "GENERAL":
        return True
    primary_types = {
        "WHAT_IS": {"general"},
        "ACTIVE_INGREDIENT": {"active_ingredient"},
        "INDICATION": {"indications"},
        "USAGE": {"usage"},
        "DOSAGE": {"dosage"},
        "FREQUENCY": {"frequency"},
        "ROUTE_OF_ADMINISTRATION": {"route_of_administration"},
        "SIDE_EFFECTS": {"common_side_effects", "side_effects", "serious_side_effects"},
        "SERIOUS_SIDE_EFFECTS": {"serious_side_effects"},
        "CONTRAINDICATION": {"contraindications"},
        "WARNING": {"warnings"},
        "INTERACTION": {"interactions"},
        "PREGNANCY": {"pregnancy"},
        "BREASTFEEDING": {"breastfeeding", "lactation"},
        "DRIVING": {"driving"},
        "OVERDOSE": {"overdose"},
        "STORAGE": {"storage"},
        "SPECIAL_POPULATIONS": {"special_populations"},
        "MISSED_DOSE": {"missed_dose"},
        "STOPPING_TREATMENT": {"stopping_treatment"},
    }
    if chunk_type in primary_types.get(intent, set()):
        return True
    if chunk_type in INTENT_CHUNK_TYPES.get(intent, ()):
        return any(term in normalize_text(text) for term in map(normalize_text, INTENT_EVIDENCE_TERMS.get(intent, ())))
    normalized = normalize_text(text)
    return any(normalize_text(term) in normalized for term in INTENT_EVIDENCE_TERMS.get(intent, ()))


def intent_search_text(intent: str, original_query: str) -> str:
    terms = INTENT_EVIDENCE_TERMS.get(intent)
    return " ".join(terms) if terms else original_query
