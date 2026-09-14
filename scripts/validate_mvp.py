"""Validate medicine resolution, intent, retrieval and grounded fallback for READY data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from config import MISSING_INFORMATION_RESPONSE, SOURCE_SECTION_MARKER
from src.database import get_chunks, get_ready_medicines
from src.intents import chunk_supports_intent, detect_intent
from src.rag import answer_query
from src.retrieval import get_top_chunks


SCENARIOS = {
    "general": ("WHAT_IS", "{name} nedir?"),
    "active_ingredient": ("ACTIVE_INGREDIENT", "{name}'in etken maddesi ne?"),
    "indications": ("INDICATION", "{name} ne için kullanılır?"),
    "usage": ("USAGE", "{name} nasıl kullanılır?"),
    "dosage": ("DOSAGE", "{name} doz bilgisi nedir?"),
    "frequency": ("FREQUENCY", "{name} günde kaç kez kullanılır?"),
    "route_of_administration": ("ROUTE_OF_ADMINISTRATION", "{name} hangi yolla uygulanır?"),
    "side_effects": ("SIDE_EFFECTS", "{name}'in yan etkileri neler?"),
    "common_side_effects": ("SIDE_EFFECTS", "{name}'in yan etkileri neler?"),
    "serious_side_effects": ("SERIOUS_SIDE_EFFECTS", "{name}'in ciddi yan etkileri neler?"),
    "contraindications": ("CONTRAINDICATION", "{name}'i kimler kullanmamalı?"),
    "warnings": ("WARNING", "{name} kullanırken nelere dikkat edilmeli?"),
    "interactions": ("INTERACTION", "{name} hangi ilaçlarla etkileşir?"),
    "pregnancy": ("PREGNANCY", "{name} hamilelikte kullanılır mı?"),
    "breastfeeding": ("BREASTFEEDING", "{name} emzirirken kullanılır mı?"),
    "lactation": ("BREASTFEEDING", "{name} emzirirken kullanılır mı?"),
    "driving": ("DRIVING", "{name} araç kullanmayı etkiler mi?"),
    "overdose": ("OVERDOSE", "{name}'ten fazla alınırsa ne olur?"),
    "storage": ("STORAGE", "{name} nasıl saklanır?"),
    "special_populations": ("SPECIAL_POPULATIONS", "{name} yaşlılarda nasıl kullanılır?"),
    "missed_dose": ("MISSED_DOSE", "{name}'i kullanmayı unutursam ne yapmalıyım?"),
    "stopping_treatment": ("STOPPING_TREATMENT", "{name}'i bırakınca ne olur?"),
}

INTENT_QUESTIONS: dict[str, str] = {}
for _chunk_type, (_intent, _template) in SCENARIOS.items():
    INTENT_QUESTIONS.setdefault(_intent, _template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-ready", type=int)
    parser.add_argument("--representative", type=int, default=10)
    parser.add_argument("--call-local-llm", action="store_true")
    args = parser.parse_args()

    ready = get_ready_medicines(limit=args.limit_ready)
    chunks_by_medicine: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in get_chunks(medicine_ids=[int(item["medicine_id"]) for item in ready]):
        chunks_by_medicine[int(chunk["medicine_id"])].append(chunk)

    tested = passed = wrong_medicine = false_no_information = source_less = 0
    failures: list[dict[str, Any]] = []
    representative_ids = {
        int(item["medicine_id"])
        for item in sorted(ready, key=lambda item: len(item["semantic_families"]), reverse=True)[: max(0, args.representative)]
    }
    local_answer_tested = local_answer_passed = 0

    for medicine in ready:
        medicine_id = int(medicine["medicine_id"])
        name = str(medicine["medicine_name"])
        chunks = chunks_by_medicine[medicine_id]
        for expected_intent, template in INTENT_QUESTIONS.items():
            expected_chunk = next(
                (
                    chunk
                    for chunk in chunks
                    if chunk_supports_intent(
                        str(chunk.get("chunk_type") or ""),
                        f"{chunk.get('section') or ''}\n{chunk.get('chunk_text') or ''}",
                        expected_intent,
                    )
                ),
                None,
            )
            if expected_chunk is None:
                continue
            question = template.format(name=name)
            tested += 1
            if detect_intent(question) != expected_intent:
                failures.append({"question": question, "reason": "wrong_intent", "actual": detect_intent(question)})
                continue
            expected_embedding = expected_chunk.get("embedding")
            trace: dict[str, Any] = {}
            results = get_top_chunks(
                question,
                database_path=None,
                embedding_function=lambda _text, vector=expected_embedding: vector,
                debug_trace=trace,
            )
            result_ids = {int(item["medicine_id"]) for item in results if item.get("medicine_id") is not None}
            if result_ids - {medicine_id}:
                wrong_medicine += 1
                failures.append({"question": question, "reason": "wrong_medicine", "ids": sorted(result_ids)})
                continue
            if not results or medicine_id not in result_ids:
                failures.append({"question": question, "reason": "no_expected_evidence"})
                continue
            if not any(str(item.get("source_name") or "").strip() for item in results):
                source_less += 1
                failures.append({"question": question, "reason": "source_missing"})
                continue

            def fail_model(_messages):
                raise RuntimeError("deterministic fallback validation")

            try:
                answer = answer_query(
                    question,
                    retrieval_function=lambda *_, selected=results, **__: selected,
                    chat_function=fail_model,
                )
            except Exception as error:
                failures.append({
                    "question": question,
                    "reason": "grounded_fallback_failed",
                    "error": str(error),
                    "chunk_types": [item.get("chunk_type") for item in results],
                })
                continue
            if answer.startswith(MISSING_INFORMATION_RESPONSE) or "ilgili belgelerde bulunamadı" in answer.casefold():
                false_no_information += 1
                failures.append({"question": question, "reason": "false_no_information"})
                continue
            if f"{SOURCE_SECTION_MARKER}Bulunamadı" in answer:
                source_less += 1
                failures.append({"question": question, "reason": "source_less_answer"})
                continue
            passed += 1

            if args.call_local_llm and medicine_id in representative_ids:
                local_answer_tested += 1
                local_trace: dict[str, Any] = {}
                try:
                    local_answer = answer_query(
                        question,
                        retrieval_function=lambda *_, selected=results, **__: selected,
                        debug_trace=local_trace,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "question": question,
                            "reason": "local_llm_answer_failed",
                            "error": str(error),
                        }
                    )
                    continue
                if not local_answer.startswith(MISSING_INFORMATION_RESPONSE) and f"{SOURCE_SECTION_MARKER}Bulunamadı" not in local_answer:
                    local_answer_passed += 1

    summary = {
        "ready_medicines": len(ready),
        "retrieval_scenarios_tested": tested,
        "retrieval_scenarios_passed": passed,
        "wrong_medicine_retrievals": wrong_medicine,
        "false_no_information_with_evidence": false_no_information,
        "source_less_answers": source_less,
        "representative_medicines": len(representative_ids),
        "local_llm_answer_scenarios_tested": local_answer_tested,
        "local_llm_answer_scenarios_passed": local_answer_passed,
        "failures": failures[:50],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures or passed != tested or local_answer_passed != local_answer_tested:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
