"""Typed settings loaded from config/pipeline.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"


@dataclass(frozen=True)
class ArxivSettings:
    categories: tuple[str, ...] = ("cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML")
    days_back: int = 14
    daily_sample_per_category: int = 60
    delay_seconds: float = 3.0
    timeout_seconds: int = 30
    user_agent: str = "arxiv-radar/0.1 (metadata pipeline; github.com/nleins5)"


@dataclass(frozen=True)
class SparkSettings:
    master: str = "local[*]"
    shuffle_partitions: int = 8
    driver_memory: str = "2g"


@dataclass(frozen=True)
class AnalyticsSettings:
    top_authors: int = 25
    top_terms: int = 30
    min_term_docs: int = 6
    trend_window_days: int = 7


@dataclass(frozen=True)
class Settings:
    lake_root: Path
    arxiv: ArxivSettings = ArxivSettings()
    spark: SparkSettings = SparkSettings()
    analytics: AnalyticsSettings = AnalyticsSettings()

    @property
    def bronze_dir(self) -> Path:
        return self.lake_root / "bronze" / "arxiv"

    @property
    def silver_dir(self) -> Path:
        return self.lake_root / "silver" / "papers"

    @property
    def gold_dir(self) -> Path:
        return self.lake_root / "gold"

    @property
    def state_dir(self) -> Path:
        return self.lake_root / "state"

    @property
    def stream_input_dir(self) -> Path:
        return self.lake_root / "stream" / "incoming"

    @property
    def checkpoint_dir(self) -> Path:
        return self.lake_root / "stream" / "checkpoints"


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Read the YAML config. RADAR_LAKE overrides the lake location."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) or {}

    arxiv_raw = dict(raw.get("arxiv", {}))
    if "categories" in arxiv_raw:
        arxiv_raw["categories"] = tuple(arxiv_raw["categories"])

    lake = os.environ.get("RADAR_LAKE") or raw.get("lake_root", "data/lake")
    return Settings(
        lake_root=_resolve(lake),
        arxiv=ArxivSettings(**arxiv_raw),
        spark=SparkSettings(**raw.get("spark", {})),
        analytics=AnalyticsSettings(**raw.get("analytics", {})),
    )
