"""Silver -> gold: analytics tables built with the DataFrame API and Spark SQL."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from radar.config import AnalyticsSettings, Settings
from radar.transform.text import extract_terms

log = logging.getLogger(__name__)

DAY = 86400


def daily_volume(silver: DataFrame) -> DataFrame:
    """Papers per day plus a trailing 7-day moving average."""
    daily = silver.groupBy("published_date").agg(F.count("*").alias("papers"))
    trailing = Window.orderBy(F.col("published_date").cast("timestamp").cast("long")).rangeBetween(-6 * DAY, 0)
    return daily.withColumn("moving_avg_7d", F.round(F.avg("papers").over(trailing), 1)).orderBy("published_date")


def category_daily(silver: DataFrame) -> DataFrame:
    return (
        silver.groupBy("published_date", "primary_category")
        .agg(F.count("*").alias("papers"))
        .orderBy("published_date", "primary_category")
    )


def category_summary(spark: SparkSession, silver: DataFrame) -> DataFrame:
    silver.createOrReplaceTempView("papers")
    return spark.sql(
        """
        SELECT primary_category                                      AS category,
               COUNT(*)                                              AS papers,
               ROUND(AVG(n_authors), 2)                              AS avg_authors,
               ROUND(AVG(abstract_words), 1)                         AS avg_abstract_words,
               ROUND(AVG(CASE WHEN is_cross_listed THEN 1.0 ELSE 0.0 END), 3) AS cross_listed_share
        FROM papers
        GROUP BY primary_category
        ORDER BY papers DESC
        """
    )


def top_authors(silver: DataFrame, limit: int) -> DataFrame:
    return (
        silver.select("paper_id", "first_author", "primary_category", F.explode("authors").alias("author"))
        .groupBy("author")
        .agg(
            F.countDistinct("paper_id").alias("papers"),
            F.sum(F.when(F.col("author") == F.col("first_author"), 1).otherwise(0)).alias("as_first_author"),
            F.array_sort(F.collect_set("primary_category")).alias("fields"),
        )
        .orderBy(F.desc("papers"), F.desc("as_first_author"), "author")
        .limit(limit)
    )


def term_trends(silver: DataFrame, cfg: AnalyticsSettings) -> DataFrame:
    """Compare how often each term appears in the latest window versus the one before it.

    Counts are document frequencies (a term counts once per paper) and growth is
    normalised by the number of papers in each window, so busy days do not look like trends.
    """
    window = cfg.trend_window_days
    latest = silver.agg(F.max("published_date")).first()[0]
    recent_start = F.date_sub(F.lit(latest), window - 1)
    previous_start = F.date_sub(F.lit(latest), 2 * window - 1)
    date_col = F.col("published_date")

    sizes = silver.agg(
        F.sum(F.when(date_col >= recent_start, 1).otherwise(0)).alias("recent_papers"),
        F.sum(F.when((date_col >= previous_start) & (date_col < recent_start), 1).otherwise(0)).alias("previous_papers"),
    ).first()
    recent_papers = max(sizes["recent_papers"] or 0, 1)
    previous_papers = max(sizes["previous_papers"] or 0, 1)

    text = silver.withColumn("text", F.concat_ws(" ", "title", "abstract"))
    stats = (
        extract_terms(text, "text")
        .groupBy("ngram", "term")
        .agg(
            F.count("*").alias("total_docs"),
            F.sum(F.when(date_col >= recent_start, 1).otherwise(0)).alias("recent_docs"),
            F.sum(F.when((date_col >= previous_start) & (date_col < recent_start), 1).otherwise(0)).alias("previous_docs"),
        )
        .where(F.col("total_docs") >= cfg.min_term_docs)
        .withColumn(
            "growth",
            F.round(
                ((F.col("recent_docs") + 1) / recent_papers) / ((F.col("previous_docs") + 1) / previous_papers), 2
            ),
        )
    )

    by_volume = Window.partitionBy("ngram").orderBy(F.desc("total_docs"), "term")
    by_growth = Window.partitionBy("ngram").orderBy(F.desc("growth"), F.desc("recent_docs"), "term")
    ranked = (
        stats.withColumn("rank_volume", F.row_number().over(by_volume))
        .withColumn("rank_growth", F.row_number().over(by_growth))
        .where((F.col("rank_volume") <= cfg.top_terms) | ((F.col("rank_growth") <= cfg.top_terms) & (F.col("recent_docs") >= cfg.min_term_docs // 2)))
    )
    return ranked.orderBy("ngram", "rank_volume")


def category_links(silver: DataFrame, limit: int = 40) -> DataFrame:
    """How often two arXiv categories are attached to the same paper."""
    pairs = silver.select("paper_id", F.explode("categories").alias("category"))
    left, right = pairs.alias("l"), pairs.alias("r")
    return (
        left.join(right, (F.col("l.paper_id") == F.col("r.paper_id")) & (F.col("l.category") < F.col("r.category")))
        .groupBy(F.col("l.category").alias("category_a"), F.col("r.category").alias("category_b"))
        .agg(F.count("*").alias("papers"))
        .orderBy(F.desc("papers"), "category_a", "category_b")
        .limit(limit)
    )


def build_gold(spark: SparkSession, silver: DataFrame, cfg: AnalyticsSettings) -> dict[str, DataFrame]:
    return {
        "daily_volume": daily_volume(silver),
        "category_daily": category_daily(silver),
        "category_summary": category_summary(spark, silver),
        "top_authors": top_authors(silver, cfg.top_authors),
        "term_trends": term_trends(silver, cfg),
        "category_links": category_links(silver),
    }


def run_gold(spark: SparkSession, settings: Settings) -> dict[str, int]:
    silver = spark.read.parquet(str(settings.silver_dir)).cache()
    counts: dict[str, int] = {}
    for name, frame in build_gold(spark, silver, settings.analytics).items():
        frame.write.mode("overwrite").parquet(str(settings.gold_dir / name))
        counts[name] = spark.read.parquet(str(settings.gold_dir / name)).count()
        log.info("gold.%s: %d rows", name, counts[name])
    silver.unpersist()
    return counts
