.PHONY: help sync lint format data macro-observed macro-predict

help:
	@echo "Targets:"
	@echo "  make sync           - Install dependencies with uv"
	@echo "  make lint           - Ruff lint"
	@echo "  make format         - Ruff format"
	@echo "  make data           - Download DeXposure dataset into ./data"
	@echo "  make macro-observed - Run macroprudential tools (observed mode)"
	@echo "  make macro-predict  - Run macroprudential tools (predict mode)"

sync:
	uv sync

lint:
	uv run ruff check dexposure_fm

format:
	uv run ruff format dexposure_fm

data:
	uv run python bin/download_dataset.py

macro-observed:
	uv run python run_macroprudential_tools.py observed \
	  --date 2025-06-30 \
	  --data-path data/historical-network_week_2025-07-01.json \
	  --contagion \
	  --output-dir output/macro-tools

macro-predict:
	uv run python run_macroprudential_tools.py predict \
	  --date 2025-06-30 \
	  --horizon 4 \
	  --data-path data/historical-network_week_2025-07-01.json \
	  --device cuda \
	  --contagion \
	  --output-dir output/macro-tools
