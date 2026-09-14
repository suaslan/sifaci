"""Prepare only enough existing local TİTCK documents to reach an MVP READY target."""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.database import (
    create_sync_run,
    finish_sync_run,
    get_documents,
    get_medicine_readiness,
    get_ready_medicines,
)
from src.titck_pipeline import run_document_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ready", type=int, default=200)
    parser.add_argument("--candidate-buffer", type=int, default=10)
    args = parser.parse_args()
    if args.target_ready <= 0:
        parser.error("--target-ready pozitif olmalıdır")

    ready_before = get_ready_medicines()
    print(f"READY başlangıç: {len(ready_before)} / hedef {args.target_ready}")
    if len(ready_before) >= args.target_ready:
        for item in ready_before[: args.target_ready]:
            print(f"READY\t{item['medicine_id']}\t{item['medicine_name']}")
        print("Hedef zaten karşılandı; hiçbir indirme, parse veya embedding yapılmadı.")
        return

    ready_ids = {int(item["medicine_id"]) for item in ready_before}
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for document in get_documents():
        medicine_id = int(document["medicine_id"])
        if medicine_id not in ready_ids and str(document.get("status")) != "FAILED_PERMANENT":
            grouped[medicine_id].append(document)
    candidates = sorted(
        grouped,
        key=lambda medicine_id: (
            len({str(item["document_type"]) for item in grouped[medicine_id]}) == 2,
            sum(str(item.get("download_status")) == "DONE" for item in grouped[medicine_id]),
            sum(str(item.get("parse_status")) == "DONE" for item in grouped[medicine_id]),
        ),
        reverse=True,
    )
    needed = args.target_ready - len(ready_before)
    selected = candidates[: needed + max(0, args.candidate_buffer)]
    if not selected:
        raise SystemExit("READY hedefi için işlenebilir yerel TİTCK belgesi bulunamadı.")

    run_id = create_sync_run("prepare_mvp")
    try:
        result = run_document_pipeline(
            sync_run_id=run_id,
            medicine_ids=selected,
            resume=True,
            limit=max(needed * 2, 10),
        )
        counters = {
            "documents_downloaded": int(result["downloaded"]),
            "chunks_created": int(result["chunks"]),
            "errors": int(result["failed_retryable"]) + int(result["failed_permanent"]),
            "downloaded": int(result["downloaded"]),
            "parsed": int(result["parsed"]),
            "embedded": int(result["embedded_documents"]),
            "failed": int(result["failed_retryable"]) + int(result["failed_permanent"]),
        }
        finish_sync_run(run_id, counters, status="completed")
    except Exception as error:
        finish_sync_run(run_id, {}, status="failed", error_log=str(error))
        raise

    ready_after = get_ready_medicines()
    new_ready_ids = {int(item["medicine_id"]) for item in ready_after} - ready_ids
    for item in ready_after:
        if int(item["medicine_id"]) in new_ready_ids:
            print(f"YENİ READY\t{item['medicine_id']}\t{item['medicine_name']}")
    for medicine_id in selected:
        if medicine_id not in new_ready_ids:
            readiness = get_medicine_readiness(medicine_id)
            print(f"RED\t{medicine_id}\t{','.join(readiness['reasons'])}")
    print(f"READY sonuç: {len(ready_after)} / hedef {args.target_ready}")
    if len(ready_after) < args.target_ready:
        raise SystemExit("READY hedefi mevcut işlenebilir adaylarla karşılanamadı.")


if __name__ == "__main__":
    main()
