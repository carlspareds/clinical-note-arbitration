.PHONY: help install dev-install test lint format run serve docker-up docker-down clean

PYTHON ?= python3
PIP ?= pip

help:
	@echo "Clinical Note Arbitration - Automation Commands"
	@echo "================================================"
	@echo "  make install        Install package dependencies"
	@echo "  make dev-install    Install package in editable mode with dev tools"
	@echo "  make test           Run pytest test suite with coverage"
	@echo "  make lint           Check code quality using ruff"
	@echo "  make format         Auto-format code using ruff"
	@echo "  make run            Run sample arbitration CLI execution"
	@echo "  make serve          Start interactive FastAPI web dashboard on http://localhost:8000"
	@echo "  make docker-up      Build and run docker-compose stack"
	@echo "  make docker-down    Stop docker-compose stack"
	@echo "  make clean          Remove build artifacts, caches, and temp files"

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev,providers]"

test:
	pytest tests/

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

run:
	arbiter judge --sample-index 0

serve:
	arbiter serve --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
