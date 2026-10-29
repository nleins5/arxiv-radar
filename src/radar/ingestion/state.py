"""Remembers which papers were already ingested so each run only adds new ones."""

from __future__ import annotations

import json
import os
from pathlib import Path


class SeenIds:
    def __init__(self, path: Path):
        self.path = path
        self._ids: set[str] = set()
        if path.exists():
            self._ids = set(json.loads(path.read_text()).get("seen_ids", []))

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def add_many(self, ids: list[str]) -> None:
        self._ids.update(ids)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seen_ids": sorted(self._ids)}))
        os.replace(tmp, self.path)  # atomic on the same filesystem
