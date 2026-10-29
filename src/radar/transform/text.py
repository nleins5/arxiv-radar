"""Text helpers used by the keyword analytics."""

from __future__ import annotations

from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_COMMON = """
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
""".split()

# Words that show up in almost every abstract and say nothing about the topic.
_ACADEMIC = """
paper propose proposed proposes present presents presented show shows shown showed demonstrate demonstrates
approach approaches method methods based using use used uses novel new work works study studies via thus however
result results experiments experimental experiment achieve achieves achieved state art existing recent
extensive significantly significant different various several many including include includes introduce
introduces introduced provide provides provided address addresses given within without across well
two three one first among towards toward can may might
alone central compared diverse independently question questions remains remain set
""".split()

STOPWORDS: list[str] = sorted(set(_COMMON) | set(_ACADEMIC))


def tokenize(df: DataFrame, text_col: str, out_col: str = "tokens") -> DataFrame:
    """Lower-case, drop URLs and possessives, strip punctuation and split on whitespace."""
    cleaned = F.lower(F.col(text_col))
    cleaned = F.regexp_replace(cleaned, r"(https?://|www\.)\S+", " ")
    cleaned = F.regexp_replace(cleaned, r"['\u2019]s\b", "")
    cleaned = F.regexp_replace(cleaned, r"[^a-z0-9\s-]", " ")
    cleaned = F.regexp_replace(cleaned, r"(^|\s)-+|-+(\s|$)", " ")
    return df.withColumn(out_col, F.split(F.trim(cleaned), r"\s+"))


def extract_terms(df: DataFrame, text_col: str, id_col: str = "paper_id", date_col: str = "published_date") -> DataFrame:
    """One row per (paper, distinct term) with ngram size 1 or 2.

    Unigrams drop stopwords. Bigrams are built on the raw token stream first, so
    they are truly adjacent words, and only then filtered so neither word is a stopword.
    """
    tokens = tokenize(df.select(id_col, date_col, text_col), text_col)

    unigrams = StopWordsRemover(inputCol="tokens", outputCol="kept", stopWords=STOPWORDS).transform(tokens)
    unigrams = unigrams.select(
        id_col, date_col, F.lit(1).alias("ngram"), F.explode(F.array_distinct("kept")).alias("term")
    ).where(F.length("term") > 2)

    bigrams = NGram(n=2, inputCol="tokens", outputCol="pairs").transform(tokens)
    stop_array = F.array(*[F.lit(w) for w in STOPWORDS])
    bigrams = (
        bigrams.select(id_col, date_col, F.explode(F.array_distinct("pairs")).alias("term"))
        .where(~F.arrays_overlap(F.split("term", " "), stop_array))
        .where(~F.exists(F.split("term", " "), lambda w: F.length(w) < 2))
        .where(F.length("term") > 5)
        .select(id_col, date_col, F.lit(2).alias("ngram"), "term")
    )
    return unigrams.unionByName(bigrams)
