"""Talk to the arXiv Atom API and turn entries into flat dictionaries."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date, datetime, timezone

import requests

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_WS = re.compile(r"\s+")
_VERSION = re.compile(r"v\d+$")


def _text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    return _WS.sub(" ", node.text).strip() or None


def base_id(arxiv_id: str) -> str:
    """2410.01234v2 -> 2410.01234"""
    return _VERSION.sub("", arxiv_id)


def parse_feed(xml_text: str, ingested_at: str | None = None) -> list[dict]:
    """Parse one Atom response page into a list of paper records."""
    stamp = ingested_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    root = ET.fromstring(xml_text)
    papers: list[dict] = []

    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry.find("atom:id", NS))
        if not raw_id or "/abs/" not in raw_id:
            continue  # the API returns an error entry when a query is malformed

        pdf = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf = link.get("href")

        primary = entry.find("arxiv:primary_category", NS)
        papers.append(
            {
                "arxiv_id": raw_id.split("/abs/", 1)[1],
                "title": _text(entry.find("atom:title", NS)),
                "summary": _text(entry.find("atom:summary", NS)),
                "authors": [
                    name
                    for name in (_text(a.find("atom:name", NS)) for a in entry.findall("atom:author", NS))
                    if name
                ],
                "categories": [c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")],
                "primary_category": primary.get("term") if primary is not None else None,
                "published": _text(entry.find("atom:published", NS)),
                "updated": _text(entry.find("atom:updated", NS)),
                "pdf_url": pdf,
                "comment": _text(entry.find("arxiv:comment", NS)),
                "doi": _text(entry.find("arxiv:doi", NS)),
                "journal_ref": _text(entry.find("arxiv:journal_ref", NS)),
                "ingested_at": stamp,
            }
        )
    return papers


def day_query(category: str, day: date) -> str:
    """arXiv date ranges are GMT and formatted YYYYMMDDHHMM."""
    stamp = day.strftime("%Y%m%d")
    return f"cat:{category} AND submittedDate:[{stamp}0000 TO {stamp}2359]"


class ArxivClient:
    """Small polite client: fixed delay between calls and retries on 429/5xx."""

    def __init__(self, user_agent: str, delay_seconds: float = 3.0, timeout: int = 30, retries: int = 5):
        self.delay = delay_seconds
        self.timeout = timeout
        self.retries = retries
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent

    @staticmethod
    def _retry_after(resp: requests.Response | None) -> float | None:
        if resp is None:
            return None
        value = resp.headers.get("Retry-After", "")
        return float(value) if value.isdigit() else None

    def _get(self, params: dict) -> str:
        for attempt in range(1, self.retries + 1):
            wait = self.delay - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = None
            try:
                resp = self._session.get(API_URL, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                # 429 means we are being throttled: wait long enough for the window to reset.
                floor = 30.0 if resp is not None and resp.status_code == 429 else self.delay
                backoff = self._retry_after(resp) or floor * 2 ** (attempt - 1)
                log.warning("arXiv request failed (%s), retry %d in %.0fs", exc, attempt, backoff)
                time.sleep(backoff)
        raise RuntimeError("unreachable")

    def papers_for_day(self, category: str, day: date, limit: int) -> list[dict]:
        """Newest `limit` papers submitted in `category` during one UTC day."""
        xml_text = self._get(
            {
                "search_query": day_query(category, day),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": 0,
                "max_results": limit,
            }
        )
        return parse_feed(xml_text)

    def iter_days(self, category: str, days: list[date], limit: int) -> Iterator[tuple[date, list[dict]]]:
        for day in days:
            yield day, self.papers_for_day(category, day, limit)
