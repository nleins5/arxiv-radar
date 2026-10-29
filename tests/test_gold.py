from __future__ import annotations

import pytest

from radar.transform.gold import build_gold
from radar.transform.silver import build_silver, read_bronze
from radar.transform.text import tokenize
from tests.conftest import write_bronze


@pytest.fixture()
def gold(spark, settings, two_week_records):
    write_bronze(settings, two_week_records)
    silver = build_silver(read_bronze(spark, str(settings.bronze_dir))).cache()
    tables = build_gold(spark, silver, settings.analytics)
    yield {name: [r.asDict() for r in frame.collect()] for name, frame in tables.items()}
    silver.unpersist()


def test_daily_volume_has_one_row_per_day(gold):
    rows = gold["daily_volume"]
    assert len(rows) == 14
    assert all(r["papers"] == 2 for r in rows)
    assert rows[-1]["moving_avg_7d"] == 2.0


def test_category_summary_counts_primary_category_only(gold):
    by_cat = {r["category"]: r for r in gold["category_summary"]}
    assert by_cat["cs.CV"]["papers"] == 14
    assert by_cat["cs.CL"]["papers"] == 14
    assert "cs.LG" not in by_cat  # only ever a secondary category in this data
    assert by_cat["cs.CV"]["cross_listed_share"] == 1.0
    assert by_cat["cs.CL"]["cross_listed_share"] == 0.0
    assert by_cat["cs.CV"]["avg_authors"] == 2.0


def test_top_authors_counts_each_paper_once(gold):
    top = {r["author"]: r for r in gold["top_authors"]}
    assert top["Ada Lovelace"]["papers"] == 14
    assert top["Grace Hopper"]["papers"] == 14
    assert top["Ada Lovelace"]["as_first_author"] == 14
    assert top["Author 3"]["papers"] == 1
    assert top["Ada Lovelace"]["fields"] == ["cs.CV"]


def test_term_trends_flag_rising_and_fading_phrases(gold):
    terms = {(r["ngram"], r["term"]): r for r in gold["term_trends"]}

    rising = terms[(2, "diffusion models")]
    assert (rising["recent_docs"], rising["previous_docs"]) == (7, 0)
    assert rising["growth"] == 8.0

    fading = terms[(2, "graph neural")]
    assert (fading["recent_docs"], fading["previous_docs"]) == (0, 7)
    assert fading["growth"] < 0.2

    steady = terms[(2, "language models")]
    assert steady["growth"] == 1.0
    assert steady["total_docs"] == 14


def test_term_trends_drop_stopwords(gold):
    unigrams = {r["term"] for r in gold["term_trends"] if r["ngram"] == 1}
    bigrams = {r["term"] for r in gold["term_trends"] if r["ngram"] == 2}
    stop = {"for", "on", "are", "the", "of"}
    assert not stop & unigrams
    assert not any(stop & set(b.split(" ")) for b in bigrams)
    assert "models" in unigrams


def test_category_links_pair_categories_alphabetically(gold):
    links = gold["category_links"]
    assert links == [{"category_a": "cs.CV", "category_b": "cs.LG", "papers": 14}]


def test_tokenize_drops_urls_and_possessives(spark):
    df = spark.createDataFrame([("The model's code: https://github.com/x/y (self-supervised).",)], ["text"])
    tokens = tokenize(df, "text").first()["tokens"]
    assert tokens == ["the", "model", "code", "self-supervised"]
