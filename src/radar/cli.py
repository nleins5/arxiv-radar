"""Command line entry point: `radar <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from radar.config import PROJECT_ROOT, load_settings


def _spark(settings):
    from radar.spark import get_spark

    return get_spark(settings.spark)


def cmd_ingest(settings, args) -> None:
    from radar.ingestion.batch import run_ingest

    result = run_ingest(settings)
    print(f"ingested {result.new_papers} new papers ({result.skipped_known} already known)")


def cmd_silver(settings, args) -> None:
    from radar.transform.silver import run_silver

    print(f"silver table holds {run_silver(_spark(settings), settings)} papers")


def cmd_gold(settings, args) -> None:
    from radar.transform.gold import run_gold

    for name, rows in run_gold(_spark(settings), settings).items():
        print(f"  gold.{name:<17} {rows:>6} rows")


def cmd_report(settings, args) -> None:
    from radar.reporting.charts import render_reports

    for path in render_reports(_spark(settings), settings, Path(args.out)):
        print(f"  {path}")


def cmd_stream(settings, args) -> None:
    from radar.streaming.micro_batch import run_demo

    run_demo(_spark(settings), settings, chunk_size=args.chunk_size, delay=args.delay)


def cmd_run(settings, args) -> None:
    cmd_ingest(settings, args)
    cmd_silver(settings, args)
    cmd_gold(settings, args)
    cmd_report(settings, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radar", description="arXiv metadata lakehouse on PySpark")
    parser.add_argument("-c", "--config", help="path to a pipeline.yaml (default: config/pipeline.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="fetch new papers from arXiv into the bronze layer").set_defaults(func=cmd_ingest)
    sub.add_parser("silver", help="clean and de-duplicate bronze into the silver table").set_defaults(func=cmd_silver)
    sub.add_parser("gold", help="build the analytics tables from silver").set_defaults(func=cmd_gold)

    report = sub.add_parser("report", help="render charts from the gold tables")
    report.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "img"), help="output folder for PNG files")
    report.set_defaults(func=cmd_report)

    stream = sub.add_parser("stream", help="replay bronze data through a Structured Streaming query")
    stream.add_argument("--chunk-size", type=int, default=40, help="records per arriving file")
    stream.add_argument("--delay", type=float, default=3.0, help="seconds between arriving files")
    stream.set_defaults(func=cmd_stream)

    run = sub.add_parser("run", help="ingest -> silver -> gold -> report")
    run.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "img"), help="output folder for PNG files")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    settings = load_settings(args.config)
    try:
        args.func(settings, args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
