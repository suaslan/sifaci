from __future__ import annotations

import pytest

from src.intents import detect_intent


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("X ne işe yarar?", "INDICATION"),
        ("X'in etken maddesi ne?", "ACTIVE_INGREDIENT"),
        ("X nasıl kullanılır?", "USAGE"),
        ("X günde kaç kez kullanılır?", "FREQUENCY"),
        ("X'in yan etkileri neler?", "SIDE_EFFECTS"),
        ("X'in ciddi yan etkileri?", "SERIOUS_SIDE_EFFECTS"),
        ("X'i kimler kullanmamalı?", "CONTRAINDICATION"),
        ("X hangi ilaçlarla etkileşir?", "INTERACTION"),
        ("X hamilelikte kullanılır mı?", "PREGNANCY"),
        ("X emzirirken kullanılır mı?", "BREASTFEEDING"),
        ("X araç kullanmayı etkiler mi?", "DRIVING"),
        ("X'ten fazla alınırsa ne olur?", "OVERDOSE"),
        ("X nasıl saklanır?", "STORAGE"),
        ("X'i kullanmayı unutursam ne yapmalıyım?", "MISSED_DOSE"),
        ("X'i bırakınca ne olur?", "STOPPING_TREATMENT"),
    ],
)
def test_shared_intent_vocabulary(question, intent):
    assert detect_intent(question) == intent


def test_medicine_name_suffix_is_not_mistaken_for_urgency():
    question = "5-FLUOROURACIL 1000 MG/20 ML ENJEKTABL ÇÖZELTİ'nin yan etkileri neler?"
    assert detect_intent(question) == "SIDE_EFFECTS"


def test_product_form_word_is_not_mistaken_for_route_question():
    question = "ABEXTIN 2 MG/ML ORAL ÇÖZELTİ ne için kullanılır?"
    assert detect_intent(question) == "INDICATION"
