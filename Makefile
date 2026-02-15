.PHONY: install dev test lint format check clean run web docker-build docker-run help

help:
	@echo "GAAP - General-purpose AI Architecture Platform"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install package"
	@echo "  dev           Install with dev dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linter"
	@echo "  format        Format code"
	@echo "  check         Run all checks (format, lint, test)"
	@echo "  clean         Clean build artifacts"
	@echo "  run           Run CLI"
	@echo "  web           Start web UI"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pip install streamlit pandas plotly

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-cov:
	pytest --cov=gaap --cov-report=term-missing

lint:
	ruff check gaap/ tests/

format:
	black gaap/ tests/
	isort gaap/ tests/
	ruff check gaap/ tests/ --fix

check: format lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	rm -rf gaap/__pycache__ tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m gaap.cli.main

web:
	streamlit run gaap/web/app.py

docker-build:
	docker build -t gaap:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down
