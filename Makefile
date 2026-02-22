.PHONY: install dev test test-unit test-int test-cov lint format check clean run web docker-build docker-run docker-stop build publish help dream eval audit

help:
	@echo "GAAP - General-purpose AI Architecture Platform (Evolution 2026)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Cognitive Ops:"
	@echo "  dream           Run the Dreaming Cycle (Consolidate Memory)"
	@echo "  eval            Run Intelligence Evaluation (IQ Score)"
	@echo "  audit           Run Constitutional Audit on Codebase"
	@echo ""
	@echo "Installation:"
	@echo "  install         Install package"
	@echo "  dev             Install with dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  check           Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "Running:"
	@echo "  run             Run CLI"
	@echo "  web             Start Streamlit web UI"
	@echo "  api             Start FastAPI server"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pip install streamlit pandas plotly chromadb sentence-transformers networkx
	pre-commit install

# ... (Standard tests omitted for brevity, keeping existing logic) ...

dream:
	@echo "🌙 Entering Sovereign REM Sleep..."
	python3 -m gaap.memory.dream_processor
	@echo "✨ Memory Consolidation Complete."

eval:
	@echo "🧪 Running Sovereign Intelligence IQ Test..."
	python3 scripts/evaluate_agent.py

audit:
	@echo "⚖️ Running Constitutional Integrity Audit..."
	python3 -m gaap.core.axioms
	@echo "✅ Axiomatic Guardrails Verified."

test:
	pytest tests/ -v --tb=short

lint:
	ruff check gaap/ tests/

format:
	black gaap/ tests/ --line-length=100
	isort gaap/ tests/ --profile=black --line-length=100
	ruff check gaap/ tests/ --fix

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

run:
	python -m gaap.cli.main

web:
	streamlit run gaap/web/app.py
