"""Benchmark the resumable pipeline against a bounded set of pending documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATABASE_PATH, SYNC_MIN_FREE_GB
from src.database import create_sync_run, finish_sync_run, initialize_database
from src.titck_pipeline import run_document_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="TİTCK pipeline performansını ölçer.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-free-gb", type=float, default=SYNC_MIN_FREE_GB)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit pozitif olmalıdır")

    initialize_database()
    run_id = create_sync_run("benchmark")
    try:
        result = run_document_pipeline(
            sync_run_id=run_id,
            limit=args.limit,
            min_free_gb=args.min_free_gb,
        )
        failed = int(result["failed_retryable"]) + int(result["failed_permanent"])
        counters = {
            "documents_downloaded": int(result["downloaded"]),
            "chunks_created": int(result["chunks"]),
            "errors": failed,
            "downloaded": int(result["downloaded"]),
            "parsed": int(result["parsed"]),
            "embedded": int(result["embedded_documents"]),
            "failed": failed,
        }
        finish_sync_run(
            run_id,
            counters,
            status="paused" if result["safely_paused"] else "benchmark_completed",
        )
    except BaseException as error:
        finish_sync_run(run_id, {}, status="failed", error_log=str(error))
        raise

    elapsed = float(result["elapsed_seconds"])
    documents = int(result["embedded_documents"])
    chunks = int(result["chunks"])
    print("\nPipeline benchmark")
    print(f"Database: {DATABASE_PATH}")
    print(f"Queued documents: {result['queued']}")
    print(f"Downloaded documents: {result['downloaded']}")
    print(f"Parsed documents: {result['parsed']}")
    print(f"Embedded documents: {documents}")
    print(f"Generated chunks: {chunks}")
    print(f"Reused embeddings: {result['reused_embeddings']}")
    print(f"Download duration: {float(result['download_seconds']):.2f} s")
    print(f"Parse duration: {float(result['parse_seconds']):.2f} s")
    print(f"Embedding duration: {float(result['embedding_seconds']):.2f} s")
    print(f"DB write duration: {float(result['db_write_seconds']):.2f} s")
    print(f"Wall duration: {elapsed:.2f} s")
    print(f"Document throughput: {documents / elapsed if elapsed else 0.0:.2f} docs/s")
    print(f"Chunk throughput: {chunks / elapsed if elapsed else 0.0:.2f} chunks/s")


if __name__ == "__main__":
    main()
