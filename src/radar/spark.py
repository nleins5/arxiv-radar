"""SparkSession factory shared by every job."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from radar.config import SparkSettings


def get_spark(settings: SparkSettings | None = None, app_name: str = "arxiv-radar") -> SparkSession:
    # Workers must run the same Python minor version as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cfg = settings or SparkSettings()
    return (
        SparkSession.builder.appName(app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.sql.shuffle.partitions", cfg.shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
