"""Render the gold tables as static charts (PNG) for the README and quick inspection."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from radar.config import Settings  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = ["#2563eb", "#f97316", "#14b8a6", "#a855f7", "#ef4444", "#64748b", "#eab308", "#0ea5e9"]
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"




def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def _hbar(ax, labels, values, color, title, value_fmt="{:,.0f}"):
    ypos = range(len(labels))
    ax.barh(ypos, values, color=color, height=0.68)
    ax.set_yticks(list(ypos), labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    top = max(values) if values else 1
    for y, v in zip(ypos, values):
        ax.text(v + top * 0.015, y, value_fmt.format(v), va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, top * 1.14)


def chart_volume(daily, per_category, out: Path, max_categories: int = 7) -> None:
    days = [r["published_date"] for r in daily]
    by_cat: dict[str, dict] = defaultdict(dict)
    for r in per_category:
        by_cat[r["primary_category"]][r["published_date"]] = r["papers"]
    totals = {c: sum(v.values()) for c, v in by_cat.items()}
    cats = sorted(totals, key=totals.get, reverse=True)
    shown, rest = cats[:max_categories], cats[max_categories:]

    series = [(cat, [by_cat[cat].get(d, 0) for d in days]) for cat in shown]
    if rest:
        series.append(
            (f"other ({len(rest)})", [sum(by_cat[c].get(d, 0) for c in rest) for d in days])
        )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = [0] * len(days)
    for i, (label, vals) in enumerate(series):
        color = PALETTE[i] if i < max_categories else "#cbd5e1"
        ax.bar(days, vals, bottom=bottom, color=color, width=0.75, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.plot(days, [r["moving_avg_7d"] for r in daily], color=INK, lw=2.2, label="7-day average")
    ax.set_title("Sampled papers per day, by primary category", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("papers in sample")
    fig.autofmt_xdate(rotation=30)
    ax.legend(ncol=5, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    _save(fig, out)


def chart_topics(terms, out: Path) -> None:
    bigrams = [r for r in terms if r["ngram"] == 2]
    steady = sorted(bigrams, key=lambda r: r["rank_volume"])[:12]
    rising = sorted(
        (r for r in bigrams if r["rank_growth"] <= 30 and r["recent_docs"] >= 6),
        key=lambda r: r["rank_growth"],
    )[:12]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))
    _hbar(left, [r["term"] for r in steady], [r["total_docs"] for r in steady], PALETTE[0], "Most common phrases (papers)")
    _hbar(
        right,
        [r["term"] for r in rising],
        [r["growth"] for r in rising],
        PALETTE[1],
        "Fastest-growing phrases (recent vs previous week)",
        value_fmt="{:.1f}x",
    )
    fig.tight_layout(w_pad=3)
    _save(fig, out)


def chart_authors(authors, out: Path) -> None:
    rows = authors[:15]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    _hbar(ax, [r["author"] for r in rows], [r["papers"] for r in rows], PALETTE[2], "Most active authors in the sample (papers)")
    _save(fig, out)


def chart_links(links, out: Path) -> None:
    rows = links[:12]
    labels = [f"{r['category_a']}  +  {r['category_b']}" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    _hbar(ax, labels, [r["papers"] for r in rows], PALETTE[3], "Categories most often listed together (papers)")
    _save(fig, out)


def render_reports(spark: SparkSession, settings: Settings, out_dir: Path) -> list[Path]:
    _style()

    def rows(name: str):
        return [r.asDict() for r in spark.read.parquet(str(settings.gold_dir / name)).collect()]

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("volume.png", lambda p: chart_volume(rows("daily_volume"), rows("category_daily"), p)),
        ("topics.png", lambda p: chart_topics(rows("term_trends"), p)),
        ("authors.png", lambda p: chart_authors(rows("top_authors"), p)),
        ("category_links.png", lambda p: chart_links(rows("category_links"), p)),
    ]
    written = []
    for filename, draw in jobs:
        target = out_dir / filename
        draw(target)
        written.append(target)
    return written
