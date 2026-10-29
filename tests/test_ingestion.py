from __future__ import annotations

import json
from datetime import date

import requests

from radar.ingestion.arxiv_client import base_id, day_query, parse_feed
from radar.ingestion.batch import run_ingest
from radar.ingestion.state import SeenIds
from tests.conftest import FIXTURES


def _sample_xml() -> str:
    return (FIXTURES / "arxiv_sample.xml").read_text()


def test_parse_feed_skips_error_entries():
    papers = parse_feed(_sample_xml(), ingested_at="2026-10-28T00:00:00Z")
    assert [p["arxiv_id"] for p in papers] == ["2610.00101v1", "2610.00101v2", "2610.00202v1", "2610.00303v1"]


def test_parse_feed_normalises_text_and_fields():
    first = parse_feed(_sample_xml())[0]
    assert first["title"] == "Sparse Attention for Long Sequences"
    assert first["summary"] == "We study sparse attention for long sequences. The method reduces memory use on long inputs."
    assert first["authors"] == ["Mira Okafor", "Jun Takeda"]
    assert first["categories"] == ["cs.LG", "cs.CL"]
    assert first["primary_category"] == "cs.LG"
    assert first["pdf_url"] == "http://arxiv.org/pdf/2610.00101v1"
    assert first["comment"] == "12 pages, 4 figures"

    third = parse_feed(_sample_xml())[2]
    assert third["doi"] == "10.1000/example.2026.202"
    assert third["journal_ref"] == "Example Robotics Letters 12 (2026)"


def test_base_id_strips_version_only():
    assert base_id("2610.00101v12") == "2610.00101"
    assert base_id("hep-th/9901001v1") == "hep-th/9901001"
    assert base_id("2610.00101") == "2610.00101"


def test_day_query_uses_gmt_range():
    assert day_query("cs.LG", date(2026, 10, 7)) == "cat:cs.LG AND submittedDate:[202610070000 TO 202610072359]"


def test_seen_ids_survive_a_restart(tmp_path):
    path = tmp_path / "state" / "seen.json"
    seen = SeenIds(path)
    seen.add_many(["a", "b"])
    seen.save()
    again = SeenIds(path)
    assert "a" in again and "c" not in again and len(again) == 2


class FakeClient:
    """Returns the same page for every category and day, like heavily cross-listed papers."""

    def __init__(self):
        self.papers = parse_feed(_sample_xml())

    def iter_days(self, category, days, limit):
        for day in days:
            yield day, list(self.papers)


def test_ingest_writes_bronze_once_and_is_incremental(settings):
    first = run_ingest(settings, client=FakeClient(), today=date(2026, 10, 28))
    # v1 and v2 of the same paper collapse to one id, so 3 distinct papers
    assert first.new_papers == 3
    assert first.output_file is not None and first.output_file.parent.name.startswith("ingest_date=")

    lines = first.output_file.read_text().splitlines()
    assert len(lines) == 3
    assert {json.loads(line)["primary_category"] for line in lines} == {"cs.LG", "cs.CV", "cs.CL"}

    second = run_ingest(settings, client=FakeClient(), today=date(2026, 10, 28))
    assert second.new_papers == 0
    assert second.output_file is None


class ThrottledClient(FakeClient):
    """Serves the first day, then fails the way arXiv does when it rate-limits."""

    def iter_days(self, category, days, limit):
        yield days[0], list(self.papers)
        raise requests.HTTPError("HTTP 429")


def test_ingest_keeps_partial_results_when_throttled(settings):
    result = run_ingest(settings, client=ThrottledClient(), today=date(2026, 10, 28))
    assert result.new_papers == 3
    assert len(result.output_file.read_text().splitlines()) == 3
    assert "2610.00202" in SeenIds(settings.state_dir / "arxiv_seen_ids.json")
