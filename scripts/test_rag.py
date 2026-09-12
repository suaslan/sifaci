"""Transparent end-to-end RAG diagnostic for one medicine question."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import find_candidate_medicines
from src.rag import answer_query, build_retrieved_context
from src.retrieval import detect_intent, get_top_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Şifacı RAG zincirini adım adım test eder.")
    parser.add_argument("question")
    args = parser.parse_args()

    candidates = find_candidate_medicines(args.question)
    print(
        "Detected medicine: "
        + (", ".join(item["medicine_name"] for item in candidates) or "NONE")
    )
    print(f"Detected intent: {detect_intent(args.question)}")

    trace: dict[str, object] = {}
    chunks = get_top_chunks(args.question, top_k=5, debug_trace=trace)
    print("Matched product:")
    for chunk in chunks:
        print(f"  - {chunk['medicine_name']}")
    print("Retrieved sections:")
    for chunk in chunks:
        print(
            f"  - {chunk.get('section') or chunk['chunk_type']} | "
            f"Similarity: {chunk['similarity_score']:.4f} | "
            f"Source: {chunk.get('source_name') or '-'}"
        )
    context = build_retrieved_context(chunks)
    print(f"Context:\n{context or 'NONE'}")

    answer_trace: dict[str, object] = {}
    answer = answer_query(args.question, debug_trace=answer_trace)
    print(f"LLM answer:\n{answer}")


if __name__ == "__main__":
    main()
