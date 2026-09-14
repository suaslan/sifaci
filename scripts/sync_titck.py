"""Synchronize the official TİTCK product list and KÜB/KT documents locally."""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

# Support the documented `python scripts/sync_titck.py` command.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

from config import (
    CONNECT_TIMEOUT,
    DB_WRITE_BATCH_SIZE,
    READ_TIMEOUT,
    TITCK_BASE_URL,
    TITCK_KUBKT_DATA_URL,
    TITCK_KUBKT_PAGE_URL,
    TITCK_MAX_RETRIES,
    TITCK_PAGE_SIZE,
    TITCK_PRODUCTS_PAGE_URL,
    TITCK_PRODUCTS_XLSX_PATH,
    TITCK_REQUEST_DELAY_SECONDS,
    SYNC_MIN_FREE_GB,
)
from src.database import (
    create_sync_run,
    find_candidate_medicines,
    finish_sync_run,
    get_all_medicines,
    get_pipeline_status,
    get_sync_state,
    initialize_database,
    mark_unseen_titck_products,
    recover_interrupted_documents,
    requeue_ocr_documents,
    register_document_catalog,
    set_sync_state,
    upsert_product,
    upsert_products,
)
from src.medicine_names import normalize_medicine_name
from src.medicine_resolution import resolve_canonical_medicine
from src.titck_pipeline import run_document_pipeline


_COLUMN_ALIASES = {
    "product_name": (
        "urun adi",
        "urun adı",
        "ilac adi",
        "ilac adı",
        "ruhsatli urun adi",
        "ruhsatlı ürün adı",
        "ticari ad",
    ),
    "active_ingredient": ("etkin madde", "etkin madde adi", "etkin madde adı"),
    "company": ("firma", "firma adi", "firma adı", "ruhsat sahibi"),
    "license_number": ("ruhsat no", "ruhsat numarasi", "ruhsat numarası"),
    "license_date": ("ruhsat tarihi",),
    "barcode": ("barkod", "barkod no"),
    "pharmaceutical_form": ("farmasotik form", "farmasötik form", "form"),
    "strength": ("doz", "yitilik", "yitilik dozu", "guc", "güç"),
    "status": ("durum", "ruhsat durumu"),
}

_DATATABLE_COLUMNS = (
    "name",
    "element",
    "firmName",
    "confirmationDateKub",
    "confirmationDateKt",
    "documentPathKub",
    "documentPathKt",
)
_RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class TitckSyncError(RuntimeError):
    pass


class TitckClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "SifaciAI/1.0 (+local official medicine index)",
                "Accept-Language": "tr-TR,tr;q=0.9",
            }
        )

    @retry(
        retry=retry_if_exception(lambda error: _is_retryable_request_error(error)),
        stop=stop_after_attempt(TITCK_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        _assert_official_url(url)
        response = self.session.request(
            method,
            url,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            **kwargs,
        )
        response.raise_for_status()
        _assert_official_url(response.url)
        if TITCK_REQUEST_DELAY_SECONDS:
            time.sleep(TITCK_REQUEST_DELAY_SECONDS)
        return response

    def latest_product_xlsx_url(self) -> str:
        response = self.request("GET", TITCK_PRODUCTS_PAGE_URL)
        soup = BeautifulSoup(response.text, "html.parser")
        candidates: list[tuple[tuple[int, int, int], int, str]] = []
        for index, anchor in enumerate(soup.select("a[href]")):
            href = urljoin(TITCK_PRODUCTS_PAGE_URL, str(anchor.get("href") or ""))
            context = " ".join(
                filter(
                    None,
                    [anchor.get_text(" ", strip=True), anchor.parent.get_text(" ", strip=True)],
                )
            )
            if ".xlsx" not in href.casefold():
                continue
            normalized = normalize_medicine_name(context + " " + href)
            # TİTCK storage filenames sometimes omit Turkish characters
            # entirely (RuhsatlBeeriTbbirnlerListesi), so match the stable
            # ASCII fragments rather than requiring a perfectly encoded title.
            if "ruhsatl" not in normalized or "listesi" not in normalized:
                continue
            candidates.append((_date_key(context + " " + href), -index, href))
        if not candidates:
            raise TitckSyncError("TİTCK sayfasında ruhsatlı ürün XLSX bağlantısı bulunamadı.")
        return max(candidates)[2]

    def download_product_xlsx(self, destination: Path) -> str:
        url = self.latest_product_xlsx_url()
        content = self.request("GET", url).content
        if not content.startswith(b"PK"):
            raise TitckSyncError("İndirilen ruhsatlı ürün dosyası geçerli bir XLSX değil.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return url

    def kubkt_record_batches(
        self,
        *,
        start_offset: int = 0,
        limit: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield discovery pages so every page can be checkpointed in SQLite."""

        page = self.request("GET", TITCK_KUBKT_PAGE_URL)
        token_match = re.search(r"_token\s*:\s*[\"']([^\"']+)", page.text)
        if not token_match:
            raise TitckSyncError("KÜB/KT sayfası erişim belirteci bulunamadı.")
        token = token_match.group(1)
        start = max(0, start_offset)
        yielded = 0
        draw = 1
        total: int | None = None
        progress: tqdm[Any] | None = None
        try:
            while (total is None or start < total) and (
                limit is None or yielded < limit
            ):
                request_length = TITCK_PAGE_SIZE
                if limit is not None:
                    request_length = min(request_length, limit - yielded)
                payload: dict[str, Any] = {
                    "draw": draw,
                    "start": start,
                    "length": request_length,
                    "order[0][column]": 0,
                    "order[0][dir]": "asc",
                    "search[value]": "",
                    "search[regex]": "false",
                    "_token": token,
                }
                for index, column in enumerate(_DATATABLE_COLUMNS):
                    payload[f"columns[{index}][data]"] = column
                    payload[f"columns[{index}][name]"] = ""
                    payload[f"columns[{index}][searchable]"] = "true"
                    payload[f"columns[{index}][orderable]"] = "true"
                    payload[f"columns[{index}][search][value]"] = ""
                    payload[f"columns[{index}][search][regex]"] = "false"
                response = self.request(
                    "POST",
                    TITCK_KUBKT_DATA_URL,
                    data=payload,
                    headers={
                        "Referer": TITCK_KUBKT_PAGE_URL,
                        "X-Requested-With": "XMLHttpRequest",
                        "X-CSRF-TOKEN": token,
                    },
                )
                body = response.json()
                page_records = body.get("data") or []
                if not isinstance(page_records, list):
                    raise TitckSyncError("KÜB/KT endpointi beklenmeyen veri döndürdü.")
                total = int(body.get("recordsTotal") or len(page_records))
                if progress is None:
                    progress = tqdm(
                        total=(
                            min(total, start_offset + limit)
                            if limit is not None
                            else total
                        ),
                        initial=start_offset,
                        desc="KÜB/KT records",
                        unit="record",
                    )
                if limit is not None:
                    page_records = page_records[: limit - yielded]
                progress.update(len(page_records))
                if not page_records:
                    break
                yield page_records
                yielded += len(page_records)
                start += len(page_records)
                draw += 1
        finally:
            if progress is not None:
                progress.close()
    def kubkt_records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [
            record
            for batch in self.kubkt_record_batches(limit=limit)
            for record in batch
        ]


def _is_retryable_request_error(error: BaseException) -> bool:
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    if not isinstance(error, requests.HTTPError):
        return False
    response = error.response
    if response is None:
        return True
    return response.status_code in _RETRYABLE_HTTP_STATUS_CODES


def read_product_list(path: Path, source_url: str) -> list[dict[str, Any]]:
    """Read the XLSX after locating and normalizing its actual header row."""

    preview = pd.read_excel(path, header=None, nrows=30, dtype=object)
    header_index = _find_header_row(preview)
    frame = pd.read_excel(path, header=header_index, dtype=object)
    print("TİTCK XLSX kolonları:")
    for column in frame.columns:
        print(f"  - {column}")

    normalized_columns = {_normalize_header(column): column for column in frame.columns}
    mapping: dict[str, Any] = {}
    for target, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            source_column = normalized_columns.get(_normalize_header(alias))
            if source_column is not None:
                mapping[target] = source_column
                break
    if "product_name" not in mapping:
        raise TitckSyncError("XLSX içinde ürün/ilaç adı kolonu bulunamadı.")

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        product_name = _cell_text(row.get(mapping["product_name"]))
        if not product_name:
            continue
        record = {
            target: _cell_text(row.get(source_column))
            for target, source_column in mapping.items()
        }
        record.update(
            {
                "product_name": product_name,
                "source": "TİTCK",
                "source_name": "TİTCK Ruhsatlı Beşeri Tıbbi Ürünler Listesi",
                "source_reference": source_url,
            }
        )
        records.append(record)
    return records


def synchronize(
    mode: str,
    *,
    limit: int | None = None,
    medicine: str | None = None,
    min_free_gb: float = SYNC_MIN_FREE_GB,
) -> dict[str, int]:
    initialize_database()
    requeued_ocr = requeue_ocr_documents() if mode == "resume" else 0
    recovered = recover_interrupted_documents() if mode == "resume" else 0
    run_id = create_sync_run(mode)
    if requeued_ocr:
        print(f"OCR recovery: {requeued_ocr} taranmış belge yeniden kuyruğa alındı.")
    if recovered:
        print(f"Checkpoint recovery: {recovered} yarım kalmış belge yeniden kuyruğa alındı.")
    counters = {
        "products_seen": 0,
        "products_added": 0,
        "products_updated": 0,
        "documents_downloaded": 0,
        "documents_updated": 0,
        "chunks_created": 0,
        "errors": 0,
    }
    errors: list[str] = []
    client = TitckClient()
    try:
        if mode == "resume" and TITCK_PRODUCTS_XLSX_PATH.exists():
            product_url = TITCK_PRODUCTS_PAGE_URL
        else:
            product_url = client.download_product_xlsx(TITCK_PRODUCTS_XLSX_PATH)
        products = read_product_list(TITCK_PRODUCTS_XLSX_PATH, product_url)
        selected_products = products[:limit] if limit else products
        normalized_seen: list[str] = []
        existing_medicines = get_all_medicines()
        existing_names = {
            str(item.get("normalized_name") or normalize_medicine_name(item["medicine_name"]))
            for item in existing_medicines
        }
        selected_names = {
            normalize_medicine_name(product["product_name"])
            for product in selected_products
        }
        product_metadata_complete = mode == "resume" and selected_names <= existing_names
        if product_metadata_complete:
            counters["products_seen"] = len(selected_products)
            normalized_seen.extend(selected_names)
            print(
                f"Resume: {len(selected_products)} ürünün metadata kaydı mevcut; "
                "tekrar upsert atlandı."
            )
        else:
            progress = tqdm(total=len(selected_products), desc="Products", unit="product")
            for offset in range(0, len(selected_products), DB_WRITE_BATCH_SIZE):
                product_batch = selected_products[offset : offset + DB_WRITE_BATCH_SIZE]
                try:
                    batch_results = upsert_products(
                        product_batch,
                        batch_size=DB_WRITE_BATCH_SIZE,
                    )
                    for product, (_, status) in zip(
                        product_batch, batch_results, strict=True
                    ):
                        counters["products_seen"] += 1
                        normalized_seen.append(
                            normalize_medicine_name(product["product_name"])
                        )
                        if status == "inserted":
                            counters["products_added"] += 1
                        elif status == "updated":
                            counters["products_updated"] += 1
                except Exception:
                    # An invalid row rolls back its batch. Retry that batch
                    # record-by-record to retain valid rows and diagnostics.
                    for product in product_batch:
                        counters["products_seen"] += 1
                        try:
                            _, status = upsert_product(product)
                            normalized_seen.append(
                                normalize_medicine_name(product["product_name"])
                            )
                            if status == "inserted":
                                counters["products_added"] += 1
                            elif status == "updated":
                                counters["products_updated"] += 1
                        except Exception as row_error:
                            _record_error(
                                errors,
                                counters,
                                f"Ürün {product.get('product_name')}: {row_error}",
                            )
                finally:
                    progress.update(len(product_batch))
            progress.close()

        if mode == "update" and limit is None:
            marked = mark_unseen_titck_products(normalized_seen)
            print(f"En güncel listede görünmeyen olarak işaretlenen ürün: {marked}")

        all_medicines = get_all_medicines()
        canonical_medicines = [
            item
            for item in all_medicines
            if "ruhsatli beseri tibbi urunler listesi"
            in normalize_medicine_name(str(item.get("source_name") or ""))
        ]
        if not canonical_medicines:
            canonical_medicines = all_medicines
        medicine_index = {
            str(item.get("normalized_name") or normalize_medicine_name(item["medicine_name"])): int(
                item["medicine_id"]
            )
            for item in canonical_medicines
        }
        cached_status = get_pipeline_status()
        discovery_snapshot = get_sync_state("titck_url_discovery_complete")
        if mode == "resume" and discovery_snapshot:
            print(
                "Discovery cache kullanılıyor: "
                f"{cached_status['total_documents']} kayıtlı belge URL'si "
                f"({discovery_snapshot} ürünlük snapshot)."
            )
        else:
            if mode != "resume":
                set_sync_state("titck_url_discovery_complete", "")
                set_sync_state("titck_discovery_offset", "0")
            start_offset = int(get_sync_state("titck_discovery_offset") or 0)
            discovered = start_offset
            catalog_result = {"inserted": 0, "updated": 0, "duplicates": 0, "pending": 0}
            for records in client.kubkt_record_batches(
                start_offset=start_offset,
                limit=limit,
            ):
                page_result = register_document_catalog(
                    _build_document_catalog(
                        records,
                        medicine_index,
                        canonical_medicines,
                        counters,
                        errors,
                    )
                )
                for key in catalog_result:
                    catalog_result[key] += page_result[key]
                discovered += len(records)
                set_sync_state("titck_discovery_offset", str(discovered))
            print(
                "Belge kataloğu: "
                f"{catalog_result['inserted']} yeni, "
                f"{catalog_result['updated']} güncel, "
                f"{catalog_result['duplicates']} değişmemiş, "
                f"{catalog_result['pending']} belirsiz eşleşme beklemede."
            )
            if limit is None:
                set_sync_state("titck_url_discovery_complete", str(discovered))
                set_sync_state("titck_discovery_offset", "0")

        medicine_ids: list[int] | None = None
        if medicine:
            matches = find_candidate_medicines(medicine, limit=50)
            medicine_ids = [int(item["medicine_id"]) for item in matches]
            if not medicine_ids:
                raise TitckSyncError(f"İlaç bulunamadı: {medicine}")
            print(
                f"Hedefli resume: {medicine} için {len(medicine_ids)} ürün varyantı."
            )

        pipeline = run_document_pipeline(
            sync_run_id=run_id,
            limit=limit,
            medicine_ids=medicine_ids,
            resume=mode == "resume",
            min_free_gb=min_free_gb,
        )
        counters.update(
            {
                "documents_downloaded": int(pipeline["downloaded"]),
                "chunks_created": int(pipeline["chunks"]),
                "errors": int(pipeline["failed_retryable"])
                + int(pipeline["failed_permanent"]),
                "downloaded": int(pipeline["downloaded"]),
                "parsed": int(pipeline["parsed"]),
                "embedded": int(pipeline["embedded_documents"]),
                "failed": int(pipeline["failed_retryable"])
                + int(pipeline["failed_permanent"]),
            }
        )
        safely_paused = bool(pipeline["safely_paused"])

        finish_sync_run(
            run_id,
            counters,
            status="paused" if safely_paused else (
                "completed" if not counters["errors"] else "completed_with_errors"
            ),
            error_log="\n".join(errors),
        )
    except KeyboardInterrupt:
        finish_sync_run(
            run_id,
            counters,
            status="paused",
            error_log="Kullanıcı tarafından güvenli biçimde duraklatıldı.",
        )
        print("Sync safely paused. Run --resume to continue.")
        return counters
    except Exception as error:
        _record_error(errors, counters, f"Senkronizasyon durdu: {error}")
        finish_sync_run(
            run_id,
            counters,
            status="failed",
            error_log="\n".join(errors),
        )
        raise

    _print_summary(counters, errors)
    if safely_paused:
        print("Sync safely paused. Run --resume to continue.")
    return counters


def _build_document_catalog(
    records: list[dict[str, Any]],
    medicine_index: dict[str, int],
    canonical_medicines: list[dict[str, Any]],
    counters: dict[str, int],
    errors: list[str],
) -> list[dict[str, Any]]:
    """Convert one discovery snapshot to durable KT/KÜB catalog rows."""

    catalog: list[dict[str, Any]] = []
    for document_type, path_key, date_key in (
        ("KT", "documentPathKt", "confirmationDateKt"),
        ("KUB", "documentPathKub", "confirmationDateKub"),
    ):
        for record in records:
            product_name = _plain_text(record.get("name"))
            if not product_name:
                _record_error(errors, counters, "KÜB/KT kaydında ilaç adı yok.")
                continue
            document_url = _document_url(record.get(path_key))
            if not document_url:
                continue
            active_ingredient = _plain_text(record.get("element"))
            company = _plain_text(record.get("firmName"))
            normalized_name = normalize_medicine_name(product_name)
            medicine_id = medicine_index.get(normalized_name)
            if medicine_id is None:
                resolution = resolve_canonical_medicine(
                    {
                        "product_name": product_name,
                        "active_ingredient": active_ingredient,
                        "company": company,
                    },
                    canonical_medicines,
                )
                if resolution["status"] == "matched":
                    medicine_id = int(resolution["medicine_id"])
                    medicine_index[normalized_name] = medicine_id
                else:
                    reason = str(resolution["reason"])
                    _record_error(
                        errors,
                        counters,
                        f"KÜB/KT eşleşmesi beklemede: {product_name} ({reason}) {document_url}",
                    )
                    catalog.append(
                        {
                            "medicine_id": None,
                            "product_name": product_name,
                            "active_ingredient": active_ingredient,
                            "company": company,
                            "document_type": document_type,
                            "document_url": document_url,
                            "approval_date": _plain_text(record.get(date_key)),
                            "source": "TİTCK",
                            "link_reason": reason,
                            "link_candidates": resolution.get("candidates") or [],
                        }
                    )
                    continue
            catalog.append(
                {
                    "medicine_id": medicine_id,
                    "document_type": document_type,
                    "document_url": document_url,
                    "approval_date": _plain_text(record.get(date_key)),
                    "source": "TİTCK",
                }
            )
    return catalog


def _find_header_row(preview: pd.DataFrame) -> int:
    all_aliases = {
        _normalize_header(alias) for aliases in _COLUMN_ALIASES.values() for alias in aliases
    }
    best_index = 0
    best_score = -1
    for index, row in preview.iterrows():
        normalized = {_normalize_header(value) for value in row if _cell_text(value)}
        score = len(normalized & all_aliases)
        if score > best_score:
            best_index = int(index)
            best_score = score
    return best_index


def _normalize_header(value: Any) -> str:
    return normalize_medicine_name(_cell_text(value))


def _cell_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    return BeautifulSoup(html.unescape(str(value)), "html.parser").get_text(" ", strip=True)


def _document_url(value: Any) -> str:
    if not value:
        return ""
    raw = html.unescape(str(value))
    soup = BeautifulSoup(raw, "html.parser")
    anchor = soup.find("a", href=True)
    url = urljoin(TITCK_BASE_URL, str(anchor["href"] if anchor else raw).strip())
    if ".pdf" not in url.casefold():
        return ""
    _assert_official_url(url)
    return url


def _date_key(value: str) -> tuple[int, int, int]:
    matches = re.findall(r"(\d{1,2})[./-](\d{1,2})[./-](20\d{2})", value)
    return max(((int(year), int(month), int(day)) for day, month, year in matches), default=(0, 0, 0))


def _assert_official_url(url: str) -> None:
    hostname = (urlparse(url).hostname or "").casefold()
    if hostname not in {"titck.gov.tr", "www.titck.gov.tr"}:
        raise TitckSyncError(f"TİTCK dışı URL reddedildi: {url}")


def _record_error(errors: list[str], counters: dict[str, int], message: str) -> None:
    counters["errors"] += 1
    errors.append(message)
    print(f"HATA: {message}", file=sys.stderr)


def _print_summary(counters: dict[str, int], errors: list[str]) -> None:
    print("\nTİTCK senkronizasyon özeti")
    for label, key in (
        ("Görülen ürün", "products_seen"),
        ("Yeni ürün", "products_added"),
        ("Güncellenen ürün", "products_updated"),
        ("İndirilen belge", "documents_downloaded"),
        ("Güncellenen belge", "documents_updated"),
        ("Oluşturulan chunk", "chunks_created"),
        ("Hata", "errors"),
    ):
        print(f"{label}: {counters[key]}")
    if errors:
        print("Ayrıntılar sync_runs.error_log alanına kaydedildi.")


def main() -> None:
    parser = argparse.ArgumentParser(description="TİTCK verilerini Şifacı'ya aktarır.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--update", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Yalnızca PENDING ve FAILED_RETRYABLE belgelerden devam et.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Geliştirme testi için işlenecek ürün ve KÜB/KT kayıt sayısını sınırlar.",
    )
    parser.add_argument(
        "--medicine",
        help="Yalnızca eşleşen ilacın ve varyantlarının kayıtlı belgelerini işle.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=SYNC_MIN_FREE_GB,
        help="Bu boş disk alanına ulaşıldığında güvenli biçimde durur.",
    )
    args = parser.parse_args()
    selected_mode = "bootstrap" if args.bootstrap else "update" if args.update else "resume"
    if args.min_free_gb < 0.25:
        parser.error("--min-free-gb en az 0.25 olmalıdır")
    synchronize(
        selected_mode,
        limit=args.limit,
        medicine=args.medicine,
        min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    main()
