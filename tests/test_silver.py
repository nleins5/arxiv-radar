from __future__ import annotations

from radar.ingestion.arxiv_client import parse_feed
from radar.transform.silver import run_silver
from tests.conftest import FIXTURES, write_bronze


def test_silver_deduplicates_versions_and_derives_columns(spark, settings):
    records = parse_feed((FIXTURES / "arxiv_sample.xml").read_text(), ingested_at="2026-10-28T00:00:00Z")
    write_bronze(settings, records)

    assert run_silver(spark, settings) == 3

    rows = {r["paper_id"]: r for r in spark.read.parquet(str(settings.silver_dir)).collect()}
    assert set(rows) == {"2610.00101", "2610.00202", "2610.00303"}

    attn = rows["2610.00101"]
    assert attn["version"] == 2  # newest version wins
    assert attn["title"] == "Sparse Attention for Long Sequences"
    assert attn["first_author"] == "Mira Okafor"
    assert attn["n_authors"] == 2
    assert attn["is_cross_listed"] is True
    assert attn["published_date"].isoformat() == "2026-10-20"
    assert (attn["year"], attn["month"]) == (2026, 10)

    robot = rows["2610.00202"]
    assert robot["is_cross_listed"] is False
    assert robot["doi"] == "10.1000/example.2026.202"
    assert robot["abstract_words"] == 8


def test_silver_is_idempotent(spark, settings):
    records = parse_feed((FIXTURES / "arxiv_sample.xml").read_text())
    write_bronze(settings, records)
    assert run_silver(spark, settings) == 3
    assert run_silver(spark, settings) == 3


def test_silver_rejects_records_without_title(spark, settings):
    records = parse_feed((FIXTURES / "arxiv_sample.xml").read_text())
    records[2]["title"] = None
    write_bronze(settings, records)
    assert run_silver(spark, settings) == 2


def test_silver_requires_bronze(spark, settings):
    import pytest

    with pytest.raises(FileNotFoundError):
        run_silver(spark, settings)
