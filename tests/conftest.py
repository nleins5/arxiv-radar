"""Shared fixtures: one local Spark session and a small synthetic lake."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from radar.config import AnalyticsSettings, Settings, SparkSettings
from radar.spark import get_spark

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def spark():
    session = get_spark(SparkSettings(master="local[2]", shuffle_partitions=2, driver_memory="1g"), "radar-tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(lake_root=tmp_path / "lake", analytics=AnalyticsSettings(min_term_docs=3, top_terms=10))


def make_record(idx: int, day: date, title: str, summary: str, categories=("cs.LG",), authors=("Ada Lovelace",), version=1):
    stamp = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "arxiv_id": f"2610.{idx:05d}v{version}",
        "title": title,
        "summary": summary,
        "authors": list(authors),
        "categories": list(categories),
        "primary_category": categories[0],
        "published": stamp,
        "updated": stamp,
        "pdf_url": f"http://arxiv.org/pdf/2610.{idx:05d}v{version}",
        "comment": None,
        "doi": None,
        "journal_ref": None,
        "ingested_at": "2026-10-28T00:00:00Z",
    }


def write_bronze(settings: Settings, records: list[dict], name: str = "papers_000000.jsonl") -> Path:
    part = settings.bronze_dir / "ingest_date=2026-10-28"
    part.mkdir(parents=True, exist_ok=True)
    target = part / name
    target.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return target


@pytest.fixture()
def two_week_records() -> list[dict]:
    """28 papers over 14 days. 'diffusion models' only shows up in the last week,
    'graph neural' only in the first one, 'language models' in both."""
    last_day = date(2026, 10, 27)
    records = []
    for n in range(14):
        day = last_day - timedelta(days=n)
        recent = n < 7
        records.append(
            make_record(
                2 * n,
                day,
                "Diffusion Models for Image Editing" if recent else "Graph Neural Networks for Molecules",
                "Diffusion models are trained on images." if recent else "A graph neural network predicts molecule properties.",
                categories=("cs.CV", "cs.LG"),
                authors=("Ada Lovelace", f"Author {n}"),
            )
        )
        records.append(
            make_record(
                2 * n + 1,
                day,
                "Scaling Language Models",
                "Large language models keep improving with scale.",
                categories=("cs.CL",),
                authors=("Grace Hopper",),
            )
        )
    return records
