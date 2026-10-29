"""Batch ingestion: pull new papers and land them untouched in the bronze layer."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from radar.config import Settings
from radar.ingestion.arxiv_client import ArxivClient, base_id
from radar.ingestion.state import SeenIds

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    new_papers: int
    skipped_known: int
    output_file: Path | None


def _window(today: date, days_back: int) -> list[date]:
    return [today - timedelta(days=n) for n in range(days_back)]


def write_bronze(records: list[dict], bronze_dir: Path, now: datetime) -> Path:
    """One JSON-lines file per run, under an ingest_date=YYYY-MM-DD partition."""
    part = bronze_dir / f"ingest_date={now:%Y-%m-%d}"
    part.mkdir(parents=True, exist_ok=True)
    target = part / f"papers_{now:%H%M%S}.jsonl"
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, target)
    return target


def run_ingest(settings: Settings, client: ArxivClient | None = None, today: date | None = None) -> IngestResult:
    cfg = settings.arxiv
    now = datetime.now(timezone.utc)
    client = client or ArxivClient(cfg.user_agent, cfg.delay_seconds, cfg.timeout_seconds)
    seen = SeenIds(settings.state_dir / "arxiv_seen_ids.json")
    days = _window(today or now.date(), cfg.days_back)

    fresh: dict[str, dict] = {}
    skipped = 0
    try:
        for category in cfg.categories:
            for day, papers in client.iter_days(category, days, cfg.daily_sample_per_category):
                added = 0
                for paper in papers:
                    pid = base_id(paper["arxiv_id"])
                    if pid in seen or pid in fresh:
                        skipped += 1
                        continue
                    fresh[pid] = paper
                    added += 1
                log.info("%-8s %s  fetched=%d new=%d", category, day, len(papers), added)
    except requests.RequestException as exc:
        # Keep what we already have; the next run continues from the saved state.
        if not fresh:
            raise
        log.warning("stopping early after arXiv error (%s); saving %d papers fetched so far", exc, len(fresh))

    if not fresh:
        log.info("nothing new to ingest")
        return IngestResult(0, skipped, None)

    target = write_bronze(list(fresh.values()), settings.bronze_dir, now)
    seen.add_many(list(fresh))
    seen.save()  # only after the file is safely on disk
    log.info("wrote %d papers to %s", len(fresh), target)
    return IngestResult(len(fresh), skipped, target)
