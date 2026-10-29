"""Bronze -> silver: typed, cleaned, de-duplicated papers stored as partitioned Parquet."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from radar.config import Settings
from radar.schemas import BRONZE_SCHEMA

log = logging.getLogger(__name__)

_TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"


def read_bronze(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.schema(BRONZE_SCHEMA).option("mode", "PERMISSIVE").json(path)


def _squash(col: str):
    return F.trim(F.regexp_replace(F.col(col), r"\s+", " "))


def build_silver(bronze: DataFrame) -> DataFrame:
    """Pure transformation, kept separate from IO so it can be unit tested."""
    typed = (
        bronze.where(F.col("arxiv_id").isNotNull() & F.col("title").isNotNull())
        .withColumn("paper_id", F.regexp_replace("arxiv_id", r"v\d+$", ""))
        .withColumn("version", F.coalesce(F.regexp_extract("arxiv_id", r"v(\d+)$", 1).cast("int"), F.lit(1)))
        .withColumn("title", _squash("title"))
        .withColumn("abstract", _squash("summary"))
        .withColumn("published_at", F.to_timestamp("published", _TS_FORMAT))
        .withColumn("updated_at", F.to_timestamp("updated", _TS_FORMAT))
    )

    # The same paper can land twice (re-ingested after a new version): keep the newest.
    newest_first = Window.partitionBy("paper_id").orderBy(
        F.col("version").desc(), F.col("updated_at").desc_nulls_last(), F.col("ingested_at").desc_nulls_last()
    )
    latest = typed.withColumn("_rank", F.row_number().over(newest_first)).where("_rank = 1").drop("_rank")

    return latest.select(
        "paper_id",
        "version",
        "title",
        "abstract",
        "authors",
        F.element_at("authors", 1).alias("first_author"),
        F.size("authors").alias("n_authors"),
        "categories",
        "primary_category",
        (F.size("categories") > 1).alias("is_cross_listed"),
        "published_at",
        "updated_at",
        F.to_date("published_at").alias("published_date"),
        F.year("published_at").alias("year"),
        F.month("published_at").alias("month"),
        F.size(F.split("abstract", r"\s+")).alias("abstract_words"),
        "pdf_url",
        "doi",
        "journal_ref",
        "comment",
    )


def run_silver(spark: SparkSession, settings: Settings) -> int:
    if not settings.bronze_dir.exists() or not any(settings.bronze_dir.rglob("*.jsonl")):
        raise FileNotFoundError(f"no bronze data under {settings.bronze_dir}; run the ingest step first")

    silver = build_silver(read_bronze(spark, str(settings.bronze_dir)))
    silver.write.mode("overwrite").partitionBy("year", "month").parquet(str(settings.silver_dir))
    count = spark.read.parquet(str(settings.silver_dir)).count()
    log.info("silver: %d papers at %s", count, settings.silver_dir)
    return count
