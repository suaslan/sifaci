"""Synchronize İlacabak as a lower-priority secondary HTML provider."""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    ILACABAK_PARSER_VERSION,
    ILACABAK_WORKERS,
)
from src.database import (
    get_ilacabak_document,
    initialize_database,
    iter_ilacabak_candidates,
    mark_ilacabak_in_progress,
    refresh_ilacabak_metadata,
    save_ilacabak_documents_batch,
    save_ilacabak_failure,
    save_ilacabak_no_content,
)
from src.dosage_parser import parse_dosage_rules
from src.embeddings import (
    close_embedding_model,
    iter_embedding_batches,
    load_embedding_model,
)
from src.ilacabak_client import IlacabakClient, IlacabakError, RobotsPolicy, load_robots_policy
from src.leaflet_parser import create_document_chunks
from src.prospektus_parser import parse_prospektus_html


_THREAD_LOCAL = threading.local()
_WORKER_CLIENTS: list[IlacabakClient] = []
_WORKER_CLIENTS_LOCK = threading.Lock()


def synchronize(
    *,
    resume: bool,
    update: bool,
    limit: int | None = None,
    medicine_query: str | None = None,
) -> dict[str, int]:
    initialize_database()
    load_embedding_model()
    policy = load_robots_policy()
    counters = {
        "seen": 0,
        "matched": 0,
        "not_found": 0,
        "no_html_prospectus": 0,
        "embedded": 0,
        "chunks": 0,
        "dosage_rules": 0,
        "failed_retryable": 0,
        "failed_permanent": 0,
        "duplicates": 0,
    }
    pending_documents: list[dict[str, Any]] = []
    pending_chunk_count = 0
    embedding_progress = tqdm(desc="Embedding", unit="chunk", dynamic_ncols=True)
    candidate_iterator = iter(
        iter_ilacabak_candidates(
            update=update, limit=limit, medicine_query=medicine_query
        )
    )

    with ThreadPoolExecutor(
        max_workers=ILACABAK_WORKERS, thread_name_prefix="ilacabak"
    ) as pool:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        for _ in range(ILACABAK_WORKERS * 2):
            if not _submit_next(pool, futures, candidate_iterator, policy, update, counters):
                break
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    medicine = futures.pop(future)
                    try:
                        result = future.result()
                        if result["status"] == "not_found":
                            counters["not_found"] += 1
                            save_ilacabak_failure(
                                int(medicine["medicine_id"]),
                                "match",
                                "İlacabak'ta güvenilir ürün eşleşmesi bulunamadı.",
                                retryable=False,
                            )
                        elif result["status"] == "no_content":
                            counters["matched"] += 1
                            counters["no_html_prospectus"] += 1
                            save_ilacabak_no_content(
                                int(medicine["medicine_id"]),
                                result["metadata"],
                                "Robots tarafından erişilebilir prospektüs HTML bölümü bulunamadı.",
                            )
                        else:
                            existing = get_ilacabak_document(int(medicine["medicine_id"]))
                            if (
                                existing
                                and existing.get("content_hash") == result["metadata"]["content_hash"]
                                and existing.get("parser_version") == ILACABAK_PARSER_VERSION
                                and existing.get("embedding_status") == "DONE"
                            ):
                                counters["duplicates"] += 1
                                refresh_ilacabak_metadata(
                                    int(medicine["medicine_id"]), result["metadata"]
                                )
                            else:
                                pending_documents.append(result)
                                pending_chunk_count += len(result["chunks"])
                            counters["matched"] += 1
                    except IlacabakError as error:
                        _record_failure(medicine, error, counters)
                    except Exception as error:
                        _record_failure(
                            medicine,
                            IlacabakError(str(error), retryable=False),
                            counters,
                        )

                    if pending_chunk_count >= EMBEDDING_BATCH_SIZE:
                        _flush_documents(
                            pending_documents, counters, embedding_progress
                        )
                        pending_documents = []
                        pending_chunk_count = 0
                    _submit_next(pool, futures, candidate_iterator, policy, update, counters)
                    if counters["seen"] % 50 == 0:
                        _print_progress(counters)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print("Sync safely paused. Run --resume to continue.")
        finally:
            if pending_documents:
                _flush_documents(pending_documents, counters, embedding_progress)
            embedding_progress.close()
            close_embedding_model()
            with _WORKER_CLIENTS_LOCK:
                for client in _WORKER_CLIENTS:
                    client.close()
                _WORKER_CLIENTS.clear()

    _print_progress(counters)
    return counters


def _submit_next(
    pool: ThreadPoolExecutor,
    futures: dict[Future[dict[str, Any]], dict[str, Any]],
    candidates: Any,
    policy: RobotsPolicy,
    refresh: bool,
    counters: dict[str, int],
) -> bool:
    try:
        medicine = next(candidates)
    except StopIteration:
        return False
    medicine_id = int(medicine["medicine_id"])
    mark_ilacabak_in_progress(medicine_id)
    counters["seen"] += 1
    future = pool.submit(_fetch_and_parse, medicine, policy, refresh)
    futures[future] = medicine
    return True


def _fetch_and_parse(
    medicine: dict[str, Any], policy: RobotsPolicy, refresh: bool
) -> dict[str, Any]:
    client = _worker_client(policy)
    product_url = str(medicine.get("ilacabak_url") or "")
    if not product_url:
        try:
            match = client.search_medicine(
                str(medicine["medicine_name"]), refresh=refresh
            )
        except IlacabakError as error:
            raise IlacabakError(
                str(error),
                retryable=error.retryable,
                attempts=error.attempts,
                stage="match",
            ) from error
        if match is None:
            return {"status": "not_found"}
        product_url = match.product_url
    product = client.get_product_metadata(product_url, refresh=refresh)
    page = client.get_html(product.prospectus_url, refresh=refresh)
    metadata = {
        "product_url": product.product_url,
        "prospectus_url": product.prospectus_url,
        "hkt_url": product.hkt_url,
        "kub_url": product.kub_url,
        "hkt_source_type": product.hkt_source_type,
        "kub_source_type": product.kub_source_type,
        "source_date": product.source_date,
        "cache_path": str(page.cache_path),
        "content_hash": page.content_hash,
        "parser_version": ILACABAK_PARSER_VERSION,
        "retry_count": page.retries,
    }
    try:
        sections = parse_prospektus_html(page.html)
    except Exception as error:
        raise IlacabakError(
            f"Prospektüs ayrıştırılamadı: {error}",
            retryable=False,
            stage="parse",
        ) from error
    if not sections:
        return {"status": "no_content", "metadata": metadata}
    chunks = create_document_chunks(str(medicine["medicine_name"]), sections)
    for chunk in chunks:
        chunk.update(
            {
                "source_type": "ILACABAK",
                "source_url": product.prospectus_url,
                "source_date": product.source_date,
                "source_priority": 200,
            }
        )
    dosage_text = "\n\n".join(
        section["text"] for section in sections if section["chunk_type"] == "dosage"
    )
    rules = parse_dosage_rules(
        dosage_text,
        source_type="ILACABAK",
        source_url=product.prospectus_url,
        source_date=product.source_date,
    )
    return {
        "status": "ready",
        "medicine_id": int(medicine["medicine_id"]),
        "metadata": metadata,
        "chunks": chunks,
        "dosage_rules": rules,
    }


def _flush_documents(
    documents: list[dict[str, Any]],
    counters: dict[str, int],
    progress: tqdm[Any],
) -> None:
    chunks = [chunk for document in documents for chunk in document["chunks"]]
    try:
        for offset, vectors in iter_embedding_batches(
            [str(chunk["chunk_text"]) for chunk in chunks],
            batch_size=EMBEDDING_BATCH_SIZE,
        ):
            batch = chunks[offset : offset + len(vectors)]
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk["embedding"] = vector
                chunk["embedding_model"] = EMBEDDING_MODEL_NAME
        save_ilacabak_documents_batch(documents)
        progress.update(len(chunks))
        for document in documents:
            counters["embedded"] += 1
            counters["chunks"] += len(document["chunks"])
            counters["dosage_rules"] += len(document["dosage_rules"])
    except Exception as error:
        for document in documents:
            save_ilacabak_failure(
                int(document["medicine_id"]),
                "embedding",
                str(error),
                retryable=True,
            )
            counters["failed_retryable"] += 1


def _record_failure(
    medicine: dict[str, Any], error: IlacabakError, counters: dict[str, int]
) -> None:
    save_ilacabak_failure(
        int(medicine["medicine_id"]),
        error.stage,
        str(error),
        retryable=error.retryable,
        attempts=error.attempts,
    )
    counters["failed_retryable" if error.retryable else "failed_permanent"] += 1


def _worker_client(policy: RobotsPolicy) -> IlacabakClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = IlacabakClient(policy)
        _THREAD_LOCAL.client = client
        with _WORKER_CLIENTS_LOCK:
            _WORKER_CLIENTS.append(client)
    return client


def _print_progress(counters: dict[str, int]) -> None:
    print("\nİlacabak senkronizasyon durumu")
    for key, value in counters.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="İlacabak ikincil veri kaynağını eşitler.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--medicine", help="Yalnızca adı bu metni içeren ilaçları işle.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit pozitif olmalıdır")
    synchronize(
        resume=args.resume,
        update=args.update,
        limit=args.limit,
        medicine_query=args.medicine,
    )


if __name__ == "__main__":
    main()
