.PHONY: setup test lint format up down

setup:
	uv venv
	uv pip install -e .[dev]
	pre-commit install

test:
	pytest tests/

lint:
	ruff check .
	mypy src/

format:
	ruff check --fix .
	ruff format .

up:
	docker compose up -d

down:
	docker compose down
