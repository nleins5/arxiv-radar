"""Feeds stored bronze records into a folder in small files, like events arriving over time."""

from __future__ import annotations

import os
import shutil
import time
from itertools import islice
from pathlib import Path

from radar.config import Settings


def _chunks(lines, size: int):
    it = iter(lines)
    while chunk := list(islice(it, size)):
        yield chunk


def reset_stream_dirs(settings: Settings) -> None:
    """Start from an empty incoming folder and fresh checkpoints."""
    for folder in (settings.stream_input_dir, settings.checkpoint_dir):
        if folder.exists():
            shutil.rmtree(folder)
    settings.stream_input_dir.mkdir(parents=True)


def replay_bronze(settings: Settings, chunk_size: int = 40, delay: float = 3.0) -> int:
    """Copy bronze lines into stream/incoming in chunks. Returns the number of files written."""
    target = settings.stream_input_dir
    target.mkdir(parents=True, exist_ok=True)

    source_files = sorted(Path(settings.bronze_dir).rglob("*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"no bronze data under {settings.bronze_dir}; run the ingest step first")

    lines = (line for f in source_files for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    written = 0
    for chunk in _chunks(lines, chunk_size):
        tmp = target / f".part_{written:04d}.tmp"  # hidden files are ignored by Spark's file source
        tmp.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        os.replace(tmp, target / f"part_{written:04d}.jsonl")  # appear atomically
        written += 1
        time.sleep(delay)
    return written
