from __future__ import annotations

import json
import shutil
from datetime import date

from radar.streaming.micro_batch import start_query
from radar.streaming.replay import replay_bronze, reset_stream_dirs
from tests.conftest import make_record, write_bronze


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_replay_splits_bronze_into_small_files(settings, two_week_records):
    write_bronze(settings, two_week_records)
    reset_stream_dirs(settings)

    written = replay_bronze(settings, chunk_size=10, delay=0)

    files = sorted(settings.stream_input_dir.glob("part_*.jsonl"))
    assert written == len(files) == 3  # 28 records -> 10 + 10 + 8
    assert not list(settings.stream_input_dir.glob(".*tmp"))
    assert sum(len(f.read_text().splitlines()) for f in files) == 28


def test_stream_counts_papers_per_day_and_category(spark, settings):
    records = [
        make_record(1, date(2026, 10, 20), "A", "x", categories=("cs.LG",)),
        make_record(2, date(2026, 10, 20), "B", "x", categories=("cs.LG",)),
        make_record(3, date(2026, 10, 20), "C", "x", categories=("cs.CV",)),
        make_record(4, date(2026, 10, 21), "D", "x", categories=("cs.LG",)),
    ]
    reset_stream_dirs(settings)
    _write_jsonl(settings.stream_input_dir / "part_0000.jsonl", records[:2])
    _write_jsonl(settings.stream_input_dir / "part_0001.jsonl", records[2:])

    query = start_query(spark, settings, sink="memory", query_name="counts_test", trigger_seconds=1)
    try:
        query.processAllAvailable()
        rows = spark.sql("SELECT day.start AS day_start, primary_category, count FROM counts_test").collect()
    finally:
        query.stop()
        shutil.rmtree(settings.lake_root, ignore_errors=True)

    result = {(r["day_start"].date().isoformat(), r["primary_category"]): r["count"] for r in rows}
    assert result == {
        ("2026-10-20", "cs.LG"): 2,
        ("2026-10-20", "cs.CV"): 1,
        ("2026-10-21", "cs.LG"): 1,
    }
