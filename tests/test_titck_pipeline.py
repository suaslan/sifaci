from __future__ import annotations

import hashlib

import pymupdf as fitz
import pytest
import requests
from tenacity import wait_none

import src.titck_pipeline as titck_pipeline
from src.database import (
    apply_pipeline_events,
    create_sync_run,
    get_documents,
    get_pipeline_status,
    initialize_database,
    insert_medicine,
    iter_pipeline_documents,
    recover_interrupted_documents,
    requeue_ocr_documents,
    register_document_catalog,
)
from src.titck_pipeline import PersistentPdfDownloader, PipelineError, run_document_pipeline


def _create_leaflet_pdf(path) -> str:
    text = """1. TEST nedir ve ne için kullanılır?
Kayıtlı endikasyon bilgisi.
2. TEST kullanmadan önce dikkat edilmesi gerekenler
Kayıtlı uyarı bilgisi.
3. TEST nasıl kullanılır?
Kayıtlı kullanım bilgisi.
4. Olası yan etkiler nelerdir?
Çok yaygın: Baş ağrısı.
5. TEST saklanması
25 derecenin altında saklayınız.
"""
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(page.rect + (36, 36, -36, -36), text, fontsize=10)
    document.save(path)
    document.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_image_only_pdf_uses_ocr_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    calls: list[tuple[str, int]] = []

    def fake_ocr(page, path, page_number):
        calls.append((str(path), page_number))
        return "OCR ile çıkarılan Türkçe metin"

    monkeypatch.setattr(titck_pipeline, "_extract_page_text_with_ocr", fake_ocr)

    assert titck_pipeline._extract_pdf_text(pdf_path) == "OCR ile çıkarılan Türkçe metin"
    assert calls == [(str(pdf_path), 1)]


def test_pipeline_resumes_from_downloaded_pdf_and_skips_completed_work(tmp_path):
    database_path = tmp_path / "pipeline.db"
    pdf_path = tmp_path / "kt.pdf"
    content_hash = _create_leaflet_pdf(pdf_path)
    initialize_database(database_path)
    medicine_id = insert_medicine("TEST 10 MG", database_path=database_path)
    register_document_catalog(
        [
            {
                "medicine_id": medicine_id,
                "document_type": "KT",
                "document_url": "https://www.titck.gov.tr/test.pdf",
            }
        ],
        database_path=database_path,
    )
    document_id = int(get_documents(medicine_id, database_path=database_path)[0]["document_id"])
    run_id = create_sync_run("test", database_path=database_path)
    apply_pipeline_events(
        [
            {
                "kind": "download_done",
                "document_id": document_id,
                "local_path": str(pdf_path),
                "content_hash": content_hash,
                "downloaded_at": "2026-09-05T00:00:00+00:00",
                "retry_count": 0,
            }
        ],
        sync_run_id=run_id,
        database_path=database_path,
    )

    first = run_document_pipeline(
        sync_run_id=run_id,
        limit=1,
        min_free_gb=0.0,
        database_path=database_path,
        embedding_function=lambda texts: [[1.0, float(index)] for index, _ in enumerate(texts)],
    )
    status = get_pipeline_status(database_path)
    second = run_document_pipeline(
        sync_run_id=run_id,
        limit=1,
        min_free_gb=0.0,
        database_path=database_path,
        embedding_function=lambda texts: pytest.fail("Completed chunks must not be embedded again"),
    )

    assert first["downloaded"] == 0
    assert first["parsed"] == 1
    assert first["embedded_documents"] == 1
    assert first["chunks"] > 0
    assert status["completed_embeddings"] == 1
    assert status["embedded_chunks"] == first["chunks"]
    assert status["last_checkpoint_at"] is not None
    assert second["queued"] == 0


class _Response:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.url = "https://www.titck.gov.tr/test.pdf"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_downloader_does_not_retry_404(monkeypatch):
    downloader = PersistentPdfDownloader(retry_wait=wait_none())
    calls = 0

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(404)

    monkeypatch.setattr(downloader.session, "get", fake_get)
    with pytest.raises(PipelineError, match="404") as caught:
        downloader.download("https://www.titck.gov.tr/test.pdf")
    downloader.close()

    assert calls == 1
    assert caught.value.retryable is False


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_downloader_retries_transient_http_then_succeeds(monkeypatch, status_code):
    downloader = PersistentPdfDownloader(retry_wait=wait_none())
    responses = iter((_Response(status_code), _Response(200, b"%PDF-test")))
    monkeypatch.setattr(downloader.session, "get", lambda *args, **kwargs: next(responses))

    content, retries = downloader.download("https://www.titck.gov.tr/test.pdf")
    downloader.close()

    assert content == b"%PDF-test"
    assert retries == 1


@pytest.mark.parametrize(
    "network_error",
    [requests.ConnectionError("DNS failure"), requests.Timeout("timeout")],
)
def test_downloader_retries_network_error_then_succeeds(monkeypatch, network_error):
    downloader = PersistentPdfDownloader(retry_wait=wait_none())
    outcomes = iter((network_error, _Response(200, b"%PDF-test")))

    def fake_get(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(downloader.session, "get", fake_get)
    content, retries = downloader.download("https://www.titck.gov.tr/test.pdf")
    downloader.close()

    assert content == b"%PDF-test"
    assert retries == 1


def test_downloader_treats_unlisted_server_error_as_permanent(monkeypatch):
    downloader = PersistentPdfDownloader(retry_wait=wait_none())
    calls = 0

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(501)

    monkeypatch.setattr(downloader.session, "get", fake_get)
    with pytest.raises(PipelineError, match="501") as caught:
        downloader.download("https://www.titck.gov.tr/test.pdf")
    downloader.close()

    assert calls == 1
    assert caught.value.retryable is False


def test_resume_selects_only_pending_and_retryable_documents(tmp_path):
    database_path = tmp_path / "resume.db"
    initialize_database(database_path)
    document_ids: dict[str, int] = {}
    medicine_ids: dict[str, int] = {}

    for status_name in ("pending", "retryable", "permanent", "done"):
        medicine_id = insert_medicine(
            f"TEST {status_name}", database_path=database_path
        )
        medicine_ids[status_name] = medicine_id
        register_document_catalog(
            [
                {
                    "medicine_id": medicine_id,
                    "document_type": "KT",
                    "document_url": f"https://www.titck.gov.tr/{status_name}.pdf",
                }
            ],
            database_path=database_path,
        )
        document_ids[status_name] = int(
            get_documents(medicine_id, database_path=database_path)[0]["document_id"]
        )

    run_id = create_sync_run("resume-test", database_path=database_path)
    apply_pipeline_events(
        [
            {
                "kind": "failure",
                "document_id": document_ids["retryable"],
                "stage": "download",
                "retryable": True,
                "message": "timeout",
                "attempts": 5,
            },
            {
                "kind": "failure",
                "document_id": document_ids["permanent"],
                "stage": "download",
                "retryable": False,
                "message": "HTTP 404",
                "attempts": 1,
            },
            {
                "kind": "stage_status",
                "document_id": document_ids["done"],
                "stage": "download",
                "status": "DONE",
            },
            {
                "kind": "stage_status",
                "document_id": document_ids["done"],
                "stage": "parse",
                "status": "DONE",
            },
            {
                "kind": "chunks_done",
                "document_id": document_ids["done"],
                "medicine_id": medicine_ids["done"],
                "chunks": [],
            },
        ],
        sync_run_id=run_id,
        database_path=database_path,
    )

    resumed = list(iter_pipeline_documents(resume=True, database_path=database_path))

    assert {row["document_id"] for row in resumed} == {
        document_ids["pending"],
        document_ids["retryable"],
    }
    assert {row["status"] for row in resumed} == {"PENDING", "FAILED_RETRYABLE"}
    stored = {row["document_id"]: row for row in get_documents(database_path=database_path)}
    assert stored[document_ids["permanent"]]["status"] == "FAILED_PERMANENT"
    assert stored[document_ids["done"]]["status"] == "DONE"


def test_resume_recovers_interrupted_stage_from_checkpoint(tmp_path):
    database_path = tmp_path / "interrupted.db"
    initialize_database(database_path)
    medicine_id = insert_medicine("INTERRUPTED TEST", database_path=database_path)
    register_document_catalog(
        [
            {
                "medicine_id": medicine_id,
                "document_type": "KT",
                "document_url": "https://www.titck.gov.tr/interrupted.pdf",
            }
        ],
        database_path=database_path,
    )
    document_id = int(
        get_documents(medicine_id, database_path=database_path)[0]["document_id"]
    )
    run_id = create_sync_run("interrupted-test", database_path=database_path)
    apply_pipeline_events(
        [
            {
                "kind": "stage_status",
                "document_id": document_id,
                "stage": "download",
                "status": "IN_PROGRESS",
            }
        ],
        sync_run_id=run_id,
        database_path=database_path,
    )

    recovered = recover_interrupted_documents(database_path=database_path)
    stored = get_documents(medicine_id, database_path=database_path)[0]
    resumed = list(iter_pipeline_documents(resume=True, database_path=database_path))

    assert recovered == 1
    assert stored["download_status"] == "FAILED_RETRYABLE"
    assert stored["status"] == "FAILED_RETRYABLE"
    assert [row["document_id"] for row in resumed] == [document_id]


def test_resume_requeues_pre_ocr_permanent_failures(tmp_path):
    database_path = tmp_path / "ocr-retry.db"
    initialize_database(database_path)
    medicine_id = insert_medicine("OCR TEST", database_path=database_path)
    register_document_catalog(
        [
            {
                "medicine_id": medicine_id,
                "document_type": "KT",
                "document_url": "https://www.titck.gov.tr/ocr.pdf",
            }
        ],
        database_path=database_path,
    )
    document_id = int(
        get_documents(medicine_id, database_path=database_path)[0]["document_id"]
    )
    run_id = create_sync_run("pre-ocr", database_path=database_path)
    apply_pipeline_events(
        [
            {
                "kind": "failure",
                "document_id": document_id,
                "stage": "parse",
                "retryable": False,
                "message": "PDF'den metin çıkarılamadı; OCR gerekebilir.",
                "attempts": 1,
            }
        ],
        sync_run_id=run_id,
        database_path=database_path,
    )

    assert requeue_ocr_documents(database_path=database_path) == 1
    stored = get_documents(medicine_id, database_path=database_path)[0]
    resumed = list(iter_pipeline_documents(resume=True, database_path=database_path))

    assert stored["parse_status"] == "FAILED_RETRYABLE"
    assert stored["status"] == "FAILED_RETRYABLE"
    assert [row["document_id"] for row in resumed] == [document_id]
    assert requeue_ocr_documents(database_path=database_path) == 0
