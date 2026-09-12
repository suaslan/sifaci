"""Print durable TİTCK pipeline counters from the canonical SQLite database."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATABASE_PATH
from src.database import get_pipeline_status


def main() -> None:
    status = get_pipeline_status()
    print(f"Database: {DATABASE_PATH}")
    for label, key in (
        ("Total medicines", "total_medicines"),
        ("Total documents", "total_documents"),
        ("Pending downloads", "pending_downloads"),
        ("Completed downloads", "completed_downloads"),
        ("Pending parse", "pending_parse"),
        ("Completed parse", "completed_parse"),
        ("Pending embeddings", "pending_embeddings"),
        ("Completed embeddings", "completed_embeddings"),
        ("Chunks", "chunks"),
        ("Embedded chunks", "embedded_chunks"),
        ("KT completed / total", "kt_completed"),
        ("KUB completed / total", "kub_completed"),
        ("Failed retryable", "failed_retryable"),
        ("Failed permanent", "failed_permanent"),
        ("Last successful task", "last_successful_task"),
        ("Sync status", "sync_status"),
        ("Last checkpoint", "last_checkpoint_at"),
    ):
        if key == "kt_completed":
            value = f"{status.get(key, 0)} / {status.get('kt_total', 0)}"
        elif key == "kub_completed":
            value = f"{status.get(key, 0)} / {status.get('kub_total', 0)}"
        else:
            value = status.get(key, 0)
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
