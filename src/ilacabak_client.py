"""Respectful HTTP client for İlacabak's public HTML pages."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from config import (
    CONNECT_TIMEOUT,
    ILACABAK_BASE_URL,
    ILACABAK_CACHE_DIR,
    ILACABAK_MATCH_THRESHOLD,
    ILACABAK_MAX_RETRIES,
    ILACABAK_REQUEST_DELAY,
    ILACABAK_ROBOTS_URL,
    ILACABAK_SEARCH_URL,
    READ_TIMEOUT,
)
from src.medicine_names import generate_medicine_aliases, normalize_medicine_name


USER_AGENT = "SifaciAI/1.0 (+local medicine index; respectful crawler)"
_GLOBAL_RATE_LOCK = threading.Lock()
_GLOBAL_LAST_REQUEST = 0.0


class IlacabakError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        attempts: int = 0,
        stage: str = "download",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts
        self.stage = stage


@dataclass(frozen=True, slots=True)
class CachedPage:
    url: str
    html: str
    cache_path: Path
    content_hash: str
    retries: int
    from_cache: bool


@dataclass(frozen=True, slots=True)
class SearchMatch:
    medicine_name: str
    product_name: str
    product_url: str
    score: float


@dataclass(frozen=True, slots=True)
class ProductMetadata:
    product_url: str
    prospectus_url: str
    hkt_url: str | None
    kub_url: str | None
    hkt_source_type: str | None
    kub_source_type: str | None
    source_date: str | None


class RobotsPolicy:
    def __init__(self, robots_text: str) -> None:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(ILACABAK_ROBOTS_URL)
        parser.parse(robots_text.splitlines())
        self._parser = parser

    def assert_allowed(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {
            "ilacabak.com",
            "www.ilacabak.com",
        }:
            raise IlacabakError(
                f"İlacabak dışı veya güvensiz URL reddedildi: {url}", retryable=False
            )
        if not self._parser.can_fetch(USER_AGENT, url):
            raise IlacabakError(
                f"robots.txt bu URL'nin alınmasına izin vermiyor: {url}",
                retryable=False,
            )


def load_robots_policy() -> RobotsPolicy:
    response = requests.get(
        ILACABAK_ROBOTS_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    if response.status_code != 200:
        raise IlacabakError(
            f"robots.txt okunamadı (HTTP {response.status_code}); tarama başlatılmadı.",
            retryable=response.status_code == 429 or response.status_code >= 500,
            attempts=1,
        )
    response.encoding = "utf-8"
    return RobotsPolicy(response.text)


class IlacabakClient:
    """Persistent, rate-limited session scoped to one download worker."""

    def __init__(self, robots_policy: RobotsPolicy) -> None:
        self.robots_policy = robots_policy
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9"}
        )

    def close(self) -> None:
        self.session.close()

    def get_html(self, url: str, *, refresh: bool = False) -> CachedPage:
        self.robots_policy.assert_allowed(url)
        cache_path = _cache_path(url)
        if cache_path.is_file() and not refresh:
            html = cache_path.read_text(encoding="utf-8")
            return CachedPage(
                url=url,
                html=html,
                cache_path=cache_path,
                content_hash=_sha256_text(html),
                retries=0,
                from_cache=True,
            )

        last_error: Exception | None = None
        for attempt in range(1, ILACABAK_MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    url,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                    allow_redirects=True,
                )
                self.robots_policy.assert_allowed(response.url)
                status = response.status_code
                if status == 429 or 500 <= status < 600:
                    raise requests.HTTPError(f"Geçici HTTP {status}", response=response)
                if status == 404:
                    raise IlacabakError(
                        f"Sayfa bulunamadı (HTTP 404): {url}",
                        retryable=False,
                        attempts=attempt,
                    )
                if 400 <= status < 500:
                    raise IlacabakError(
                        f"Kalıcı HTTP {status}: {url}",
                        retryable=False,
                        attempts=attempt,
                    )
                response.raise_for_status()
                response.encoding = "utf-8"
                html = response.text
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".part")
                temporary.write_text(html, encoding="utf-8")
                temporary.replace(cache_path)
                return CachedPage(
                    url=response.url,
                    html=html,
                    cache_path=cache_path,
                    content_hash=_sha256_text(html),
                    retries=attempt - 1,
                    from_cache=False,
                )
            except IlacabakError:
                raise
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
                last_error = error
                if attempt == ILACABAK_MAX_RETRIES:
                    break
                time.sleep(min(2**attempt, 16))
        raise IlacabakError(
            f"{url}: {last_error}",
            retryable=True,
            attempts=ILACABAK_MAX_RETRIES,
        )

    def search_medicine(
        self,
        medicine_name: str,
        *,
        refresh: bool = False,
    ) -> SearchMatch | None:
        candidates = self.search_candidates(medicine_name, refresh=refresh)
        if not candidates:
            return None
        best = candidates[0]
        return best if best.score >= ILACABAK_MATCH_THRESHOLD else None

    def search_candidates(
        self,
        medicine_name: str,
        *,
        refresh: bool = False,
    ) -> list[SearchMatch]:
        """Return ranked product candidates for transparent match diagnostics."""

        candidates: list[SearchMatch] = []
        normalized_target = normalize_medicine_name(medicine_name)
        aliases = generate_medicine_aliases(medicine_name)
        queries = [medicine_name] + [
            alias
            for alias in reversed(aliases)
            if normalize_medicine_name(alias) != normalized_target
        ]
        for search_term in dict.fromkeys(queries):
            url = f"{ILACABAK_SEARCH_URL}?{urlencode({'arama': search_term})}"
            page = self.get_html(url, refresh=refresh)
            soup = BeautifulSoup(page.html, "html.parser")
            for anchor in soup.select("a[href]"):
                product_name = anchor.get_text(" ", strip=True)
                product_url = urljoin(ILACABAK_BASE_URL, str(anchor.get("href") or ""))
                parsed = urlparse(product_url)
                if parsed.hostname not in {"ilacabak.com", "www.ilacabak.com"}:
                    continue
                if not re.fullmatch(r"/[^/]+-\d+", parsed.path.rstrip("/")):
                    continue
                normalized_product = normalize_medicine_name(product_name)
                score = max(
                    fuzz.ratio(normalized_target, normalized_product),
                    fuzz.token_set_ratio(normalized_target, normalized_product),
                ) / 100.0
                candidates.append(
                    SearchMatch(
                        medicine_name,
                        product_name,
                        product_url.rstrip("/"),
                        score,
                    )
                )
            if candidates:
                break
        if not candidates:
            return []
        best_by_url: dict[str, SearchMatch] = {}
        for candidate in candidates:
            previous = best_by_url.get(candidate.product_url)
            if previous is None or candidate.score > previous.score:
                best_by_url[candidate.product_url] = candidate
        ranked = list(best_by_url.values())
        ranked.sort(key=lambda item: (item.score, -len(item.product_name)), reverse=True)
        return ranked

    def get_product_metadata(
        self,
        product_url: str,
        *,
        refresh: bool = False,
    ) -> ProductMetadata:
        page = self.get_html(product_url, refresh=refresh)
        soup = BeautifulSoup(page.html, "html.parser")
        hkt_url = _find_link(soup, ("kullanma talimatı", "hasta kullanma talimatı"))
        kub_url = _find_link(soup, ("kısa ürün bilgisi", "kub", "küb"))
        text = soup.get_text(" ", strip=True)
        date_match = re.search(
            r"Son\s+Güncelleme\s*:\s*([^|]+?)(?=\s+(?:Paylaş|DDD|Menü|$))",
            text,
            flags=re.IGNORECASE,
        )
        return ProductMetadata(
            product_url=product_url,
            prospectus_url=f"{product_url}/prospektus",
            hkt_url=hkt_url,
            kub_url=kub_url,
            hkt_source_type=_classify_document_link(hkt_url),
            kub_source_type=_classify_document_link(kub_url),
            source_date=date_match.group(1).strip() if date_match else None,
        )

    def _wait_for_rate_limit(self) -> None:
        global _GLOBAL_LAST_REQUEST
        with _GLOBAL_RATE_LOCK:
            remaining = ILACABAK_REQUEST_DELAY - (
                time.monotonic() - _GLOBAL_LAST_REQUEST
            )
            if remaining > 0:
                time.sleep(remaining)
            _GLOBAL_LAST_REQUEST = time.monotonic()


def _find_link(soup: BeautifulSoup, terms: tuple[str, ...]) -> str | None:
    for anchor in soup.select("a[href]"):
        context = " ".join(
            [anchor.get_text(" ", strip=True), str(anchor.get("title") or "")]
        ).casefold()
        if any(term in context for term in terms):
            return urljoin(ILACABAK_BASE_URL, str(anchor.get("href"))).strip()
    return None


def _classify_document_link(url: str | None) -> str | None:
    if not url:
        return None
    hostname = (urlparse(url).hostname or "").casefold()
    if hostname.endswith("titck.gov.tr"):
        return "TITCK"
    if hostname in {"ilacabak.com", "www.ilacabak.com"}:
        return "ILACABAK_MIRROR"
    return "MANUFACTURER"


def _cache_path(url: str) -> Path:
    return ILACABAK_CACHE_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.html"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
