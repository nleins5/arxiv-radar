"""Structured Streaming: count new papers per category in 1-day event-time windows.

Spark processes the stream as a series of micro-batches. Here the source is a folder of
JSON-lines files; `maxFilesPerTrigger` makes every file its own micro-batch so the
behaviour is easy to watch.
"""

from __future__ import annotations

import logging
import threading

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from radar.config import Settings
from radar.schemas import BRONZE_SCHEMA
from radar.streaming.replay import replay_bronze, reset_stream_dirs

log = logging.getLogger(__name__)

_TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"


def open_stream(spark: SparkSession, input_dir: str, files_per_batch: int = 1) -> DataFrame:
    return (
        spark.readStream.schema(BRONZE_SCHEMA)
        .option("maxFilesPerTrigger", files_per_batch)
        .json(input_dir)
    )


def windowed_counts(stream: DataFrame) -> DataFrame:
    return (
        stream.withColumn("published_ts", F.to_timestamp("published", _TS_FORMAT))
        .where(F.col("published_ts").isNotNull())
        .withWatermark("published_ts", "2 days")
        .groupBy(F.window("published_ts", "1 day").alias("day"), "primary_category")
        .count()
    )


def start_query(
    spark: SparkSession,
    settings: Settings,
    sink: str = "console",
    query_name: str = "category_counts",
    trigger_seconds: int = 2,
) -> StreamingQuery:
    counts = windowed_counts(open_stream(spark, str(settings.stream_input_dir)))
    writer = counts.writeStream.outputMode("update").queryName(query_name).format(sink)
    if sink == "console":
        writer = writer.option("truncate", False).option("numRows", 15)
    writer = writer.option("checkpointLocation", str(settings.checkpoint_dir / query_name))
    return writer.trigger(processingTime=f"{trigger_seconds} seconds").start()


def run_demo(spark: SparkSession, settings: Settings, chunk_size: int = 40, delay: float = 3.0) -> None:
    """Replay the bronze layer into the stream folder while a streaming query consumes it."""
    reset_stream_dirs(settings)
    finished = threading.Event()

    def feeder() -> None:
        try:
            written = replay_bronze(settings, chunk_size=chunk_size, delay=delay)
            log.info("replay finished: %d files", written)
        finally:
            finished.set()

    threading.Thread(target=feeder, daemon=True).start()
    query = start_query(spark, settings, trigger_seconds=max(1, int(delay) - 1))
    try:
        while not finished.is_set():
            query.awaitTermination(1)
        query.processAllAvailable()
    finally:
        query.stop()
