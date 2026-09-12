"""Bounded, resumable TİTCK document ingestion pipeline."""

from __future__ import annotations

import hashlib
import io
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import pymupdf as fitz
import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from config import (
    CONNECT_TIMEOUT,
    DATABASE_PATH,
    DB_WRITE_BATCH_SIZE,
    DB_WRITE_QUEUE_MAX,
    DOCUMENTS_DIR,
    DOWNLOAD_QUEUE_MAX,
    DOWNLOAD_WORKERS,
    EMBED_QUEUE_MAX,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_RETRIES,
    OCR_DPI,
    OCR_ENABLED,
    OCR_LANGUAGE,
    PARSE_QUEUE_MAX,
    PARSE_WORKERS,
    PIPELINE_CHECKPOINT_INTERVAL,
    PIPELINE_CHECKPOINT_SECONDS,
    READ_TIMEOUT,
    TESSERACT_CMD,
    TITCK_REQUEST_DELAY_SECONDS,
)
from src.database import (
    apply_pipeline_events,
    get_document_raw_text,
    get_embeddings_by_hashes,
    get_pipeline_status,
    iter_pipeline_documents,
)
from src.embeddings import close_embedding_model, generate_embeddings, load_embedding_model
from src.leaflet_parser import create_document_chunks, parse_leaflet_sections
from src.medicine_names import normalize_medicine_name


_SENTINEL = object()
_RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class PipelineError(RuntimeError):
    """Base error for one pipeline task."""

    def __init__(self, message: str, *, retryable: bool, attempts: int = 0) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts


@dataclass(slots=True)
class DocumentTask:
    document_id: int
    medicine_id: int
    medicine_name: str
    document_type: str
    document_url: str
    local_path: str
    approval_date: str
    content_hash: str
    download_status: str
    parse_status: str
    embedding_status: str
    status: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DocumentTask":
        medicine_id = int(row["medicine_id"])
        medicine_name = str(row["medicine_name"])
        document_type = str(row["document_type"])
        local_path = str(row.get("local_path") or _document_path(
            medicine_id, medicine_name, document_type
        ))
        return cls(
            document_id=int(row["document_id"]),
            medicine_id=medicine_id,
            medicine_name=medicine_name,
            document_type=document_type,
            document_url=str(row["document_url"]),
            local_path=local_path,
            approval_date=str(row.get("approval_date") or ""),
            content_hash=str(row.get("content_hash") or ""),
            download_status=str(row.get("download_status") or "PENDING"),
            parse_status=str(row.get("parse_status") or "PENDING"),
            embedding_status=str(row.get("embedding_status") or "PENDING"),
            status=str(row.get("status") or "PENDING"),
        )


@dataclass(slots=True)
class ParsedDocument:
    task: DocumentTask
    chunks: list[dict[str, Any]]


@dataclass
class PipelineMetrics:
    queued: int = 0
    downloaded: int = 0
    parsed: int = 0
    embedded_documents: int = 0
    chunks: int = 0
    embedded_chunks: int = 0
    reused_embeddings: int = 0
    retries: int = 0
    failed_retryable: int = 0
    failed_permanent: int = 0
    download_seconds: float = 0.0
    parse_seconds: float = 0.0
    embedding_seconds: float = 0.0
    db_write_seconds: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **values: int | float) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, getattr(self, name) + value)

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            snapshot = {
                name: getattr(self, name)
                for name in (
                    "queued",
                    "downloaded",
                    "parsed",
                    "embedded_documents",
                    "chunks",
                    "embedded_chunks",
                    "reused_embeddings",
                    "retries",
                    "failed_retryable",
                    "failed_permanent",
                    "download_seconds",
                    "parse_seconds",
                    "embedding_seconds",
                    "db_write_seconds",
                )
            }
            snapshot["elapsed_seconds"] = time.perf_counter() - self.started_at
            return snapshot


def _is_retryable_download_error(error: BaseException) -> bool:
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    if not isinstance(error, requests.HTTPError):
        return False
    response = error.response
    return response is None or response.status_code in _RETRYABLE_HTTP_STATUS_CODES


class PersistentPdfDownloader:
    """One pooled HTTP session, owned by one download worker."""

    def __init__(self, *, retry_wait: Any | None = None) -> None:
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=2,
            max_retries=0,
        )
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "SifaciAI/2.0 (+local official medicine index)",
                "Accept-Language": "tr-TR,tr;q=0.9",
            }
        )
        self.retry_wait = retry_wait or wait_exponential(multiplier=1, min=1, max=32)

    def close(self) -> None:
        self.session.close()

    def download(self, url: str) -> tuple[bytes, int]:
        _assert_official_url(url)
        attempts = 0

        def download_once() -> bytes:
            nonlocal attempts
            attempts += 1
            response = self.session.get(
                url,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            )
            _assert_official_url(response.url)
            if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                response.raise_for_status()
            if response.status_code >= 400:
                label = "Belge bulunamadı" if response.status_code == 404 else "Kalıcı HTTP hatası"
                raise PipelineError(
                    f"{label} (HTTP {response.status_code}): {url}",
                    retryable=False,
                    attempts=attempts,
                )
            content = response.content
            if not content.startswith(b"%PDF"):
                raise PipelineError(
                    f"Belge geçerli bir PDF değil: {url}",
                    retryable=False,
                    attempts=attempts,
                )
            return content

        retryer = Retrying(
            retry=retry_if_exception(_is_retryable_download_error),
            stop=stop_after_attempt(MAX_RETRIES),
            wait=self.retry_wait,
            reraise=True,
        )
        try:
            content = retryer(download_once)
        except PipelineError:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            raise PipelineError(
                f"{url}: {error}",
                retryable=True,
                attempts=attempts,
            ) from error
        if TITCK_REQUEST_DELAY_SECONDS:
            time.sleep(TITCK_REQUEST_DELAY_SECONDS)
        return content, max(0, attempts - 1)


def run_document_pipeline(
    *,
    sync_run_id: int,
    limit: int | None = None,
    medicine_ids: Sequence[int] | None = None,
    resume: bool = False,
    min_free_gb: float,
    database_path: str | Path = DATABASE_PATH,
    embedding_function: Callable[[Sequence[str]], list[list[float]]] = generate_embeddings,
    show_progress: bool = True,
) -> dict[str, int | float | bool]:
    """Run download -> parse -> embed -> single-writer stages."""

    download_queue: queue.Queue[DocumentTask | object] = queue.Queue(DOWNLOAD_QUEUE_MAX)
    parse_queue: queue.Queue[DocumentTask | object] = queue.Queue(PARSE_QUEUE_MAX)
    embedding_queue: queue.Queue[ParsedDocument | object] = queue.Queue(EMBED_QUEUE_MAX)
    database_queue: queue.Queue[dict[str, Any] | object] = queue.Queue(DB_WRITE_QUEUE_MAX)
    fatal_errors: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    metrics = PipelineMetrics()

    writer = threading.Thread(
        target=_database_writer,
        name="titck-db-writer",
        args=(database_queue, fatal_errors, stop_event, metrics, sync_run_id, database_path),
    )
    embedder = threading.Thread(
        target=_embedding_worker,
        name="titck-embedding",
        args=(
            embedding_queue,
            database_queue,
            fatal_errors,
            stop_event,
            metrics,
            database_path,
            embedding_function,
            show_progress,
            embedding_function is generate_embeddings,
        ),
    )
    writer.start()
    database_queue.put(_checkpoint_event(metrics.snapshot()))
    embedder.start()
    parse_pool = ThreadPoolExecutor(
        max_workers=PARSE_WORKERS, thread_name_prefix="titck-parser"
    )
    download_pool = ThreadPoolExecutor(
        max_workers=DOWNLOAD_WORKERS, thread_name_prefix="titck-download"
    )
    parser_futures = [
        parse_pool.submit(
            _parse_worker,
            parse_queue,
            embedding_queue,
            database_queue,
            fatal_errors,
            stop_event,
            metrics,
            database_path,
        )
        for _ in range(PARSE_WORKERS)
    ]
    downloader_futures = [
        download_pool.submit(
            _download_worker,
            download_queue,
            parse_queue,
            database_queue,
            fatal_errors,
            stop_event,
            metrics,
        )
        for _ in range(DOWNLOAD_WORKERS)
    ]

    safely_paused = False
    try:
        for row in iter_pipeline_documents(
            limit=limit,
            medicine_ids=medicine_ids,
            resume=resume,
            database_path=database_path,
        ):
            _raise_fatal(fatal_errors)
            if _free_gb(database_path) < min_free_gb:
                safely_paused = True
                stop_event.set()
                break
            task = DocumentTask.from_row(row)
            metrics.add(queued=1)
            if _cached_download_is_valid(task):
                _put_with_backpressure(parse_queue, task, stop_event)
            else:
                _put_with_backpressure(download_queue, task, stop_event)
    except KeyboardInterrupt:
        safely_paused = True
        stop_event.set()
    finally:
        for _ in downloader_futures:
            _put_sentinel(download_queue)
        for future in downloader_futures:
            future.result()
        download_pool.shutdown(wait=True)
        for _ in parser_futures:
            _put_sentinel(parse_queue)
        for future in parser_futures:
            future.result()
        parse_pool.shutdown(wait=True)
        _put_sentinel(embedding_queue)
        embedder.join()
        _put_sentinel(database_queue)
        writer.join()

    _raise_fatal(fatal_errors)
    result = metrics.snapshot()
    result["safely_paused"] = safely_paused
    result["elapsed_seconds"] = time.perf_counter() - metrics.started_at
    return result


def _download_worker(
    input_queue: queue.Queue[DocumentTask | object],
    output_queue: queue.Queue[DocumentTask | object],
    database_queue: queue.Queue[dict[str, Any] | object],
    fatal_errors: queue.Queue[BaseException],
    stop_event: threading.Event,
    metrics: PipelineMetrics,
) -> None:
    downloader = PersistentPdfDownloader()
    try:
        while True:
            item = input_queue.get()
            try:
                if item is _SENTINEL:
                    return
                if stop_event.is_set():
                    continue
                task = item
                assert isinstance(task, DocumentTask)
                database_queue.put(_status_event(task, "download", "IN_PROGRESS"))
                started = time.perf_counter()
                try:
                    content, retries = downloader.download(task.document_url)
                    content_hash = hashlib.sha256(content).hexdigest()
                    target = Path(task.local_path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + f".{threading.get_ident()}.part")
                    temporary.write_bytes(content)
                    os.replace(temporary, target)
                    if task.content_hash != content_hash:
                        task.parse_status = "PENDING"
                        task.embedding_status = "PENDING"
                    task.content_hash = content_hash
                    task.download_status = "DONE"
                    database_queue.put(
                        {
                            "kind": "download_done",
                            "document_id": task.document_id,
                            "local_path": str(target),
                            "content_hash": content_hash,
                            "downloaded_at": datetime.now(UTC).isoformat(),
                            "retry_count": retries,
                        }
                    )
                    metrics.add(
                        downloaded=1,
                        retries=retries,
                        download_seconds=time.perf_counter() - started,
                    )
                    _put_with_backpressure(output_queue, task, stop_event)
                except PipelineError as error:
                    metrics.add(
                        failed_retryable=int(error.retryable),
                        failed_permanent=int(not error.retryable),
                        retries=max(0, error.attempts - 1),
                        download_seconds=time.perf_counter() - started,
                    )
                    database_queue.put(_failure_event(task, "download", error))
            except BaseException as error:
                fatal_errors.put(error)
                stop_event.set()
            finally:
                input_queue.task_done()
    finally:
        downloader.close()


def _parse_worker(
    input_queue: queue.Queue[DocumentTask | object],
    output_queue: queue.Queue[ParsedDocument | object],
    database_queue: queue.Queue[dict[str, Any] | object],
    fatal_errors: queue.Queue[BaseException],
    stop_event: threading.Event,
    metrics: PipelineMetrics,
    database_path: str | Path,
) -> None:
    while True:
        item = input_queue.get()
        try:
            if item is _SENTINEL:
                return
            if stop_event.is_set():
                continue
            task = item
            assert isinstance(task, DocumentTask)
            database_queue.put(_status_event(task, "parse", "IN_PROGRESS"))
            started = time.perf_counter()
            try:
                raw_text = (
                    get_document_raw_text(task.document_id, database_path=database_path)
                    if task.parse_status == "DONE"
                    else _extract_pdf_text(Path(task.local_path))
                )
                if not raw_text.strip():
                    raise PipelineError(
                        "PDF'den OCR sonrasında da metin çıkarılamadı.",
                        retryable=True,
                        attempts=1,
                    )
                sections = parse_leaflet_sections(raw_text, task.document_type)
                chunks = create_document_chunks(task.medicine_name, sections)
                if not chunks:
                    raise PipelineError("Belgeden chunk oluşturulamadı.", retryable=False)
                for chunk in chunks:
                    chunk["source_url"] = task.document_url
                    chunk["approval_date"] = task.approval_date
                    chunk["source_type"] = "TITCK"
                    chunk["source_date"] = task.approval_date
                    chunk["source_priority"] = 500 if task.document_type == "KUB" else 450
                if task.parse_status != "DONE":
                    database_queue.put(
                        {
                            "kind": "parse_done",
                            "document_id": task.document_id,
                            "raw_text": raw_text,
                        }
                    )
                metrics.add(parsed=1, parse_seconds=time.perf_counter() - started)
                _put_with_backpressure(
                    output_queue,
                    ParsedDocument(task=task, chunks=chunks),
                    stop_event,
                )
            except PipelineError as error:
                metrics.add(
                    failed_retryable=int(error.retryable),
                    failed_permanent=int(not error.retryable),
                    parse_seconds=time.perf_counter() - started,
                )
                database_queue.put(_failure_event(task, "parse", error))
            except (OSError, RuntimeError, ValueError) as error:
                failure = PipelineError(str(error), retryable=False)
                metrics.add(
                    failed_permanent=1,
                    parse_seconds=time.perf_counter() - started,
                )
                database_queue.put(_failure_event(task, "parse", failure))
        except BaseException as error:
            fatal_errors.put(error)
            stop_event.set()
        finally:
            input_queue.task_done()


def _embedding_worker(
    input_queue: queue.Queue[ParsedDocument | object],
    database_queue: queue.Queue[dict[str, Any] | object],
    fatal_errors: queue.Queue[BaseException],
    stop_event: threading.Event,
    metrics: PipelineMetrics,
    database_path: str | Path,
    embedding_function: Callable[[Sequence[str]], list[list[float]]],
    show_progress: bool,
    manage_embedding_model: bool,
) -> None:
    if manage_embedding_model:
        try:
            load_embedding_model()
        except BaseException as error:
            fatal_errors.put(error)
            stop_event.set()
            return

    pending: list[ParsedDocument] = []
    pending_chunks = 0
    progress = tqdm(
        desc="Embedding",
        unit="chunk",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    while True:
        item = input_queue.get()
        try:
            if item is _SENTINEL:
                if pending and not stop_event.is_set():
                    _flush_embedding_batch(
                        pending,
                        database_queue,
                        metrics,
                        database_path,
                        embedding_function,
                        progress,
                    )
                progress.close()
                if manage_embedding_model:
                    close_embedding_model()
                return
            if stop_event.is_set():
                continue
            document = item
            assert isinstance(document, ParsedDocument)
            pending.append(document)
            pending_chunks += len(document.chunks)
            if pending_chunks >= EMBEDDING_BATCH_SIZE:
                _flush_embedding_batch(
                    pending,
                    database_queue,
                    metrics,
                    database_path,
                    embedding_function,
                    progress,
                )
                pending = []
                pending_chunks = 0
        except BaseException as error:
            if isinstance(item, ParsedDocument):
                failure = PipelineError(str(error), retryable=True)
                failed_documents = pending or [item]
                for failed_document in failed_documents:
                    database_queue.put(
                        _failure_event(failed_document.task, "embedding", failure)
                    )
                metrics.add(failed_retryable=len(failed_documents))
                pending = []
                pending_chunks = 0
            else:
                fatal_errors.put(error)
                stop_event.set()
        finally:
            input_queue.task_done()


def _flush_embedding_batch(
    documents: list[ParsedDocument],
    database_queue: queue.Queue[dict[str, Any] | object],
    metrics: PipelineMetrics,
    database_path: str | Path,
    embedding_function: Callable[[Sequence[str]], list[list[float]]],
    progress: tqdm[Any],
) -> None:
    started = time.perf_counter()
    all_chunks = [chunk for document in documents for chunk in document.chunks]
    hashes = [str(chunk["chunk_hash"]) for chunk in all_chunks]
    cached = get_embeddings_by_hashes(
        hashes,
        embedding_model=EMBEDDING_MODEL_NAME,
        database_path=database_path,
    )
    missing_by_hash: dict[str, dict[str, Any]] = {}
    for chunk in all_chunks:
        chunk_hash = str(chunk["chunk_hash"])
        if chunk_hash in cached:
            chunk["embedding"] = cached[chunk_hash]
            chunk["embedding_model"] = EMBEDDING_MODEL_NAME
        else:
            missing_by_hash.setdefault(chunk_hash, chunk)

    generated: dict[str, list[float]] = {}
    missing_items = list(missing_by_hash.items())
    for offset in range(0, len(missing_items), EMBEDDING_BATCH_SIZE):
        batch = missing_items[offset : offset + EMBEDDING_BATCH_SIZE]
        vectors = _generate_adaptive_embeddings(
            [chunk["chunk_text"] for _, chunk in batch], embedding_function
        )
        generated.update(
            (chunk_hash, vector)
            for (chunk_hash, _), vector in zip(batch, vectors, strict=True)
        )
    for chunk in all_chunks:
        chunk_hash = str(chunk["chunk_hash"])
        if "embedding" not in chunk:
            chunk["embedding"] = generated[chunk_hash]
            chunk["embedding_model"] = EMBEDDING_MODEL_NAME

    for document in documents:
        database_queue.put(
            {
                "kind": "chunks_done",
                "document_id": document.task.document_id,
                "medicine_id": document.task.medicine_id,
                "chunks": document.chunks,
            }
        )
    metrics.add(
        embedded_documents=len(documents),
        chunks=len(all_chunks),
        embedded_chunks=len(missing_by_hash),
        reused_embeddings=len(all_chunks) - len(missing_by_hash),
        embedding_seconds=time.perf_counter() - started,
    )
    progress.update(len(all_chunks))


def _generate_adaptive_embeddings(
    texts: Sequence[str],
    embedding_function: Callable[[Sequence[str]], list[list[float]]],
) -> list[list[float]]:
    """Split a cancelled long batch while keeping one Foundry model instance."""

    try:
        return embedding_function(texts)
    except Exception:
        if len(texts) <= 1:
            raise
        midpoint = len(texts) // 2
        return _generate_adaptive_embeddings(texts[:midpoint], embedding_function) + (
            _generate_adaptive_embeddings(texts[midpoint:], embedding_function)
        )


def _database_writer(
    input_queue: queue.Queue[dict[str, Any] | object],
    fatal_errors: queue.Queue[BaseException],
    stop_event: threading.Event,
    metrics: PipelineMetrics,
    sync_run_id: int,
    database_path: str | Path,
) -> None:
    pending: list[dict[str, Any]] = []
    last_checkpoint_progress = 0
    last_checkpoint_time = time.monotonic()
    while True:
        try:
            item = input_queue.get(timeout=0.5)
        except queue.Empty:
            item = None
        try:
            if item is _SENTINEL:
                if pending:
                    _write_event_batch(pending, metrics, sync_run_id, database_path)
                snapshot = metrics.snapshot()
                _write_event_batch(
                    [_checkpoint_event(snapshot)], metrics, sync_run_id, database_path
                )
                return
            if isinstance(item, dict):
                pending.append(item)
            if pending and (len(pending) >= DB_WRITE_BATCH_SIZE or item is None):
                _write_event_batch(pending, metrics, sync_run_id, database_path)
                pending = []
            snapshot = metrics.snapshot()
            finished = int(snapshot["embedded_documents"]) + int(
                snapshot["failed_retryable"]
            ) + int(snapshot["failed_permanent"])
            checkpoint_progress = max(
                int(snapshot["downloaded"]),
                int(snapshot["parsed"]),
                finished,
            )
            now = time.monotonic()
            progress_checkpoint_due = (
                checkpoint_progress - last_checkpoint_progress
                >= PIPELINE_CHECKPOINT_INTERVAL
            )
            time_checkpoint_due = (
                now - last_checkpoint_time >= PIPELINE_CHECKPOINT_SECONDS
            )
            if progress_checkpoint_due or time_checkpoint_due:
                checkpoint = _checkpoint_event(snapshot)
                _write_event_batch([checkpoint], metrics, sync_run_id, database_path)
                if checkpoint_progress != last_checkpoint_progress:
                    _print_progress(snapshot, database_path)
                last_checkpoint_progress = checkpoint_progress
                last_checkpoint_time = now
        except BaseException as error:
            fatal_errors.put(error)
            stop_event.set()
            return
        finally:
            if item is not None:
                input_queue.task_done()


def _write_event_batch(
    events: list[dict[str, Any]],
    metrics: PipelineMetrics,
    sync_run_id: int,
    database_path: str | Path,
) -> None:
    started = time.perf_counter()
    apply_pipeline_events(
        events,
        sync_run_id=sync_run_id,
        database_path=database_path,
    )
    metrics.add(db_write_seconds=time.perf_counter() - started)


def _checkpoint_event(snapshot: Mapping[str, int | float]) -> dict[str, Any]:
    failed = int(snapshot["failed_retryable"]) + int(snapshot["failed_permanent"])
    return {
        "kind": "checkpoint",
        "downloaded": int(snapshot["downloaded"]),
        "parsed": int(snapshot["parsed"]),
        "embedded": int(snapshot["embedded_documents"]),
        "failed": failed,
        "chunks": int(snapshot["chunks"]),
    }


def _print_progress(snapshot: Mapping[str, int | float], database_path: str | Path) -> None:
    status = get_pipeline_status(database_path)
    elapsed = float(snapshot["elapsed_seconds"])
    documents_per_minute = (
        float(snapshot["embedded_documents"]) * 60.0 / elapsed if elapsed else 0.0
    )
    chunks_per_minute = float(snapshot["chunks"]) * 60.0 / elapsed if elapsed else 0.0
    download_speed = float(snapshot["downloaded"]) / elapsed if elapsed else 0.0
    embedding_speed = float(snapshot["embedded_chunks"]) / elapsed if elapsed else 0.0
    average_per_document = (
        elapsed / float(snapshot["embedded_documents"])
        if snapshot["embedded_documents"]
        else 0.0
    )
    print(
        "\nPipeline checkpoint"
        f"\nMedicines: {status.get('total_medicines', 0)}"
        f"\nDownloaded: {snapshot['downloaded']} ({download_speed:.2f} docs/sec)"
        f"\nParsed: {snapshot['parsed']}"
        f"\nChunks: {snapshot['chunks']}"
        f"\nEmbedded: {snapshot['embedded_chunks']} ({embedding_speed:.2f} chunks/sec)"
        f"\nKT progress: {status.get('kt_completed', 0)} / {status.get('kt_total', 0)}"
        f"\nKUB progress: {status.get('kub_completed', 0)} / {status.get('kub_total', 0)}"
        f"\nDocuments/minute: {documents_per_minute:.2f}"
        f"\nChunks/minute: {chunks_per_minute:.2f}"
        f"\nAverage/document: {average_per_document:.2f} s"
        f"\nElapsed: {elapsed:.2f} s"
        f"\nRetry queue: {status.get('failed_retryable', 0)}"
        f"\nFailed permanent: {status.get('failed_permanent', 0)}",
        flush=True,
    )


def _status_event(task: DocumentTask, stage: str, status: str) -> dict[str, Any]:
    return {
        "kind": "stage_status",
        "document_id": task.document_id,
        "stage": stage,
        "status": status,
    }


def _failure_event(task: DocumentTask, stage: str, error: PipelineError) -> dict[str, Any]:
    return {
        "kind": "failure",
        "document_id": task.document_id,
        "stage": stage,
        "message": str(error),
        "error_type": type(error).__name__,
        "retryable": error.retryable,
        "attempts": error.attempts,
    }


def _put_with_backpressure(
    target: queue.Queue[Any], item: Any, stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        try:
            target.put(item, timeout=0.5)
            return
        except queue.Full:
            continue


def _put_sentinel(target: queue.Queue[Any]) -> None:
    while True:
        try:
            target.put(_SENTINEL, timeout=0.5)
            return
        except queue.Full:
            continue


def _raise_fatal(fatal_errors: queue.Queue[BaseException]) -> None:
    try:
        error = fatal_errors.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"Pipeline worker stopped: {error}") from error


def _cached_download_is_valid(task: DocumentTask) -> bool:
    if task.download_status != "DONE" or not task.content_hash:
        return False
    path = Path(task.local_path)
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == task.content_hash


def _extract_pdf_text(path: Path) -> str:
    with fitz.open(path) as document:
        pages: list[str] = []
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text", sort=True)
            if page_text.strip():
                pages.append(page_text)
                continue
            if OCR_ENABLED:
                pages.append(_extract_page_text_with_ocr(page, path, page_number))
        return "\n".join(pages)


def _extract_page_text_with_ocr(page: fitz.Page, path: Path, page_number: int) -> str:
    """Render one image-only PDF page and extract Turkish text with Tesseract."""

    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise PipelineError(
            "OCR bağımlılıkları eksik; Pillow ve pytesseract kurulmalıdır.",
            retryable=True,
            attempts=1,
        ) from error

    command = TESSERACT_CMD or shutil.which("tesseract")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    try:
        pixmap = page.get_pixmap(dpi=OCR_DPI, alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            return str(pytesseract.image_to_string(image, lang=OCR_LANGUAGE))
    except Exception as error:
        raise PipelineError(
            f"OCR çalıştırılamadı ({path}, sayfa {page_number}): {error}",
            retryable=True,
            attempts=1,
        ) from error


def _document_path(medicine_id: int, medicine_name: str, document_type: str) -> Path:
    slug = normalize_medicine_name(medicine_name).replace(" ", "-")[:90] or "medicine"
    return DOCUMENTS_DIR / f"{medicine_id}-{slug}" / f"{document_type.casefold()}.pdf"


def _assert_official_url(url: str) -> None:
    hostname = (urlparse(url).hostname or "").casefold()
    if hostname not in {"titck.gov.tr", "www.titck.gov.tr"}:
        raise PipelineError(f"TİTCK dışı URL reddedildi: {url}", retryable=False)


def _free_gb(database_path: str | Path) -> float:
    return shutil.disk_usage(Path(database_path).resolve().parent).free / (1024**3)
