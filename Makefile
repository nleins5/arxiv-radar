PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup test ingest silver gold report pipeline stream clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

ingest:
	$(BIN)/radar ingest

silver:
	$(BIN)/radar silver

gold:
	$(BIN)/radar gold

report:
	$(BIN)/radar report

pipeline:
	$(BIN)/radar run

stream:
	$(BIN)/radar stream

clean:
	rm -rf data spark-warehouse metastore_db derby.log .pytest_cache
