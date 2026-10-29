"""Explicit schemas. Reading JSON without one means a full scan to infer types."""

from pyspark.sql.types import ArrayType, StringType, StructField, StructType

# Raw layer: exactly what the arXiv Atom feed gives us, timestamps kept as text.
BRONZE_SCHEMA = StructType(
    [
        StructField("arxiv_id", StringType(), nullable=False),
        StructField("title", StringType()),
        StructField("summary", StringType()),
        StructField("authors", ArrayType(StringType())),
        StructField("categories", ArrayType(StringType())),
        StructField("primary_category", StringType()),
        StructField("published", StringType()),
        StructField("updated", StringType()),
        StructField("pdf_url", StringType()),
        StructField("comment", StringType()),
        StructField("doi", StringType()),
        StructField("journal_ref", StringType()),
        StructField("ingested_at", StringType()),
    ]
)
