"""Trace medicine resolution, intent, cosine retrieval and final LLM context."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._console_table import print_table
from src.rag import answer_query
from src.retrieval import get_top_chunks


def _simulated_chat(_: Sequence[Mapping[str, str]]) -> str:
    return "DEBUG_SIMULATION: LLM çağrısı yapılmadı."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir sorgunun ilaç tespiti, intent, cosine retrieval ve LLM context izini gösterir."
    )
    parser.add_argument("question", help='Sorgu; örn. "Muscoflex yan etkileri neler?"')
    parser.add_argument("--top-k", type=int, default=5, help="Döndürülecek en fazla chunk sayısı.")
    parser.add_argument(
        "--call-llm", action="store_true",
        help="Simülasyon yerine yerel LLM'i gerçekten çağır.",
    )
    parser.add_argument("--database", type=Path, help="Varsayılan yerine kullanılacak SQLite yolu.")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k pozitif olmalıdır")

    trace: dict[str, object] = {}
    retrieval_function = (
        get_top_chunks
        if args.database is None
        else partial(get_top_chunks, database_path=args.database)
    )
    call_options = {
        "top_k": args.top_k,
        "retrieval_function": retrieval_function,
        "debug_trace": trace,
    }
    if args.call_llm:
        answer = answer_query(args.question, **call_options)
    else:
        answer = answer_query(args.question, chat_function=_simulated_chat, **call_options)

    candidates = list(trace.get("medicine_candidates") or [])
    matched_ids = list(trace.get("matched_medicine_ids") or [])
    filtered = list(trace.get("filtered_sections") or [])
    chunks = list(trace.get("top_chunks") or [])

    print_table(
        "1. Sorgu ve sınıflandırma",
        ("Sorgu", "Tespit edilen ilaç", "Intent", "Eşleşen ID'ler"),
        [(args.question, trace.get("detected_medicine"), trace.get("detected_intent") or "GENERAL", ", ".join(map(str, matched_ids)) or "-")],
        max_cell_width=100,
    )
    print_table(
        "2. İlaç/alias eşleşmeleri",
        ("ID", "Kayıtlı ad", "Alias", "Eşleşme", "Ad skoru"),
        [
            (item.get("medicine_id"), item.get("medicine_name"), item.get("matched_alias") or item.get("alias"), item.get("match_type"), f"{float(item.get('name_score') or 0):.4f}")
            for item in candidates
        ],
    )
    print_table(
        "3. Intent ile filtrelenen chunk'lar",
        ("Chunk ID", "Doc ID", "Belge", "Chunk türü", "Bölüm"),
        [
            (item.get("chunk_id"), item.get("document_id"), item.get("document_type"), item.get("chunk_type"), item.get("section"))
            for item in filtered
        ],
        max_cell_width=70,
    )
    print_table(
        "4. Seçilen chunk'lar ve cosine similarity",
        ("Sıra", "Chunk ID", "Doc ID", "İlaç", "Tür", "Cosine", "Kaynak", "Metin"),
        [
            (
                index, item.get("chunk_id"), item.get("document_id"),
                item.get("medicine_name"), item.get("chunk_type"),
                f"{float(item.get('similarity_score') or 0):.6f}",
                item.get("source_name"), item.get("chunk_text"),
            )
            for index, item in enumerate(chunks, start=1)
        ],
        max_cell_width=90,
    )
    print_table(
        "5. Embedding/retrieval durumu",
        ("Embedding", "Boyut", "DB chunk", "Eski model", "Eşik", "Retrieval durumu"),
        [(
            "oluşturuldu" if trace.get("query_embedding_created") else "oluşturulmadı",
            trace.get("query_embedding_dimension"), trace.get("database_chunk_count"),
            trace.get("stale_embedding_count"), trace.get("minimum_similarity_score"),
            trace.get("retrieval_state"),
        )],
    )

    print("\n6. LLM'e giden final context/prompt")
    print("-" * 80)
    print(trace.get("llm_user_prompt") or trace.get("retrieved_context") or "(context oluşmadı)")
    print("-" * 80)
    print(f"LLM modu: {'GERÇEK ÇAĞRI' if args.call_llm else 'SİMÜLASYON'}")
    print(f"Model çağrı noktasına ulaşıldı: {'evet' if trace.get('model_called') else 'hayır'}")
    print(f"Fallback nedeni: {trace.get('fallback_reason') or '-'}")
    print("\n7. Final cevap")
    print(answer)


if __name__ == "__main__":
    main()
