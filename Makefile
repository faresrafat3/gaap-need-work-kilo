.PHONY: install dev test test-unit test-int test-cov lint format check clean run web docker-build docker-run docker-stop build publish help

help:
	@echo "GAAP - General-purpose AI Architecture Platform"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install         Install package"
	@echo "  dev             Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-int        Run integration tests"
	@echo "  test-cov        Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linter (ruff)"
	@echo "  format          Format code (black, isort, ruff)"
	@echo "  typecheck       Run type checker (mypy)"
	@echo "  check           Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "Running:"
	@echo "  run             Run CLI"
	@echo "  web             Start Streamlit web UI"
	@echo "  api             Start FastAPI server"
	@echo "  doctor          Run diagnostics"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker containers"
	@echo "  docker-stop     Stop Docker containers"
	@echo ""
	@echo "Distribution:"
	@echo "  build           Build distribution packages"
	@echo "  publish         Publish to PyPI (requires TWINE_API_KEY)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Remove build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pip install streamlit pandas plotly
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-int:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ --cov=gaap --cov-report=term-missing --cov-report=html

lint:
	ruff check gaap/ tests/

format:
	black gaap/ tests/ --line-length=100
	isort gaap/ tests/ --profile=black --line-length=100
	ruff check gaap/ tests/ --fix

typecheck:
	mypy gaap/ --ignore-missing-imports

check: format lint typecheck test-unit

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run:
	python -m gaap.cli.main

web:
	streamlit run gaap/web/app.py

api:
	uvicorn gaap.api.fastapi_app:app --reload --port 8000

doctor:
	python -m gaap.cli.main doctor

docker-build:
	docker build -t gaap:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

build: clean
	python -m build

publish: build
	twine upload dist/*
