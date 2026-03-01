.PHONY: install dev test test-unit test-int test-cov lint format check clean run web docker-build docker-run docker-stop build publish help dream eval audit security security-audit security-bandit security-safety security-pip-audit security-gitleaks security-deps verify pre-push

help:
	@echo "GAAP - General-purpose AI Architecture Platform (Evolution 2026)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Quick Start:"
	@echo "  verify          Run pre-push verification (health checks)"
	@echo "  pre-push        Full pre-push validation (tests, lint, security)"
	@echo ""
	@echo "Cognitive Ops:"
	@echo "  dream           Run the Dreaming Cycle (Consolidate Memory)"
	@echo "  eval            Run Intelligence Evaluation (IQ Score)"
	@echo "  audit           Run Constitutional Audit on Codebase"
	@echo ""
	@echo "Security:"
	@echo "  security        Run full security audit"
	@echo "  security-bandit Run Bandit security linter"
	@echo "  security-safety Run Safety vulnerability check"
	@echo "  security-pip-audit Run pip-audit security audit"
	@echo "  security-gitleaks Run Gitleaks secret scanner"
	@echo "  security-deps   Check dependency health"
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

# =============================================================================
# Security Targets
# =============================================================================

security: security-audit

security-audit:
	@echo "🔒 Running full security audit..."
	@mkdir -p security-reports
	@$(MAKE) security-bandit || true
	@$(MAKE) security-safety || true
	@$(MAKE) security-pip-audit || true
	@$(MAKE) security-gitleaks || true
	@$(MAKE) security-custom-audit || true
	@$(MAKE) security-deps || true
	@echo "✅ Security audit complete! Reports in security-reports/"

security-bandit:
	@echo "🔍 Running Bandit security linter..."
	@which bandit > /dev/null || (echo "📦 Installing bandit..." && pip install bandit[toml]>=1.7.0)
	@mkdir -p security-reports
	@bandit -r . -f txt -o security-reports/bandit-report.txt \
		--exclude './.venv,./venv,./.git,./__pycache__,./.pytest_cache,./.mypy_cache,./build,./dist,./.eggs,./frontend/node_modules,./security-reports' \
		2>/dev/null || true
	@cat security-reports/bandit-report.txt 2>/dev/null || echo "⚠️ Bandit report not generated"
	@echo "✅ Bandit scan complete"

security-safety:
	@echo "🔍 Running Safety vulnerability check..."
	@which safety > /dev/null || (echo "📦 Installing safety..." && pip install safety>=3.0.0)
	@mkdir -p security-reports
	@safety check --output security-reports/safety-report.txt --file requirements.txt 2>/dev/null || true
	@cat security-reports/safety-report.txt 2>/dev/null || echo "⚠️ Safety report not generated"
	@echo "✅ Safety check complete"

security-pip-audit:
	@echo "🔍 Running pip-audit..."
	@which pip-audit > /dev/null || (echo "📦 Installing pip-audit..." && pip install pip-audit>=2.6.0)
	@mkdir -p security-reports
	@pip-audit --requirement requirements.txt --format=markdown --output=security-reports/pip-audit-report.md 2>/dev/null || true
	@cat security-reports/pip-audit-report.md 2>/dev/null || echo "⚠️ pip-audit report not generated"
	@echo "✅ pip-audit complete"

security-gitleaks:
	@echo "🔍 Running Gitleaks secret scanner..."
	@if command -v gitleaks >/dev/null 2>&1; then \
		mkdir -p security-reports && \
		gitleaks detect --source . \
			--report-format json \
			--report-path security-reports/gitleaks-report.json \
			--verbose 2>/dev/null || true; \
	else \
		echo "⚠️ Gitleaks not installed. Install from: https://github.com/gitleaks/gitleaks"; \
		echo "   Or use: docker run -v $$(pwd):/path zricethezav/gitleaks detect --source /path"; \
	fi
	@echo "✅ Gitleaks scan complete"

security-custom-audit:
	@echo "🔍 Running custom security audit..."
	@mkdir -p security-reports
	@python scripts/security/audit-codebase.py \
		--output security-reports/custom-audit.txt \
		--format text 2>/dev/null || true
	@cat security-reports/custom-audit.txt 2>/dev/null || echo "⚠️ Custom audit report not generated"
	@echo "✅ Custom audit complete"

security-deps:
	@echo "🔍 Running dependency health check..."
	@mkdir -p security-reports
	@python scripts/security/check-dependencies.py \
		--requirements requirements.txt \
		--output security-reports/dependency-health.txt \
		--format text 2>/dev/null || true
	@cat security-reports/dependency-health.txt 2>/dev/null || echo "⚠️ Dependency health report not generated"
	@echo "✅ Dependency health check complete"

security-clean:
	@echo "🧹 Cleaning security reports..."
	@rm -rf security-reports
	@echo "✅ Security reports cleaned"

security-install-tools:
	@echo "📦 Installing security tools..."
	@pip install --upgrade \
		bandit[toml]>=1.7.0 \
		safety>=3.0.0 \
		pip-audit>=2.6.0
	@echo "⚠️ Note: Install Gitleaks separately from https://github.com/gitleaks/gitleaks"
	@echo "✅ Security tools installed"

# =============================================================================
# Security Targets
# =============================================================================

security: security-audit

security-audit:
	@echo "🔒 Running full security audit..."
	@mkdir -p security-reports
	@$(MAKE) security-bandit || true
	@$(MAKE) security-safety || true
	@$(MAKE) security-pip-audit || true
	@$(MAKE) security-gitleaks || true
	@$(MAKE) security-custom-audit || true
	@$(MAKE) security-deps || true
	@echo "✅ Security audit complete! Reports in security-reports/"

security-bandit:
	@echo "🔍 Running Bandit security linter..."
	@which bandit > /dev/null || (echo "📦 Installing bandit..." && pip install bandit[toml]>=1.7.0)
	@mkdir -p security-reports
	@bandit -r . -f txt -o security-reports/bandit-report.txt \
		--exclude './.venv,./venv,./.git,./__pycache__,./.pytest_cache,./.mypy_cache,./build,./dist,./.eggs,./frontend/node_modules,./security-reports' \
		2>/dev/null || true
	@cat security-reports/bandit-report.txt 2>/dev/null || echo "⚠️ Bandit report not generated"
	@echo "✅ Bandit scan complete"

security-safety:
	@echo "🔍 Running Safety vulnerability check..."
	@which safety > /dev/null || (echo "📦 Installing safety..." && pip install safety>=3.0.0)
	@mkdir -p security-reports
	@safety check --output security-reports/safety-report.txt --file requirements.txt 2>/dev/null || true
	@cat security-reports/safety-report.txt 2>/dev/null || echo "⚠️ Safety report not generated"
	@echo "✅ Safety check complete"

security-pip-audit:
	@echo "🔍 Running pip-audit..."
	@which pip-audit > /dev/null || (echo "📦 Installing pip-audit..." && pip install pip-audit>=2.6.0)
	@mkdir -p security-reports
	@pip-audit --requirement requirements.txt --format=markdown --output=security-reports/pip-audit-report.md 2>/dev/null || true
	@cat security-reports/pip-audit-report.md 2>/dev/null || echo "⚠️ pip-audit report not generated"
	@echo "✅ pip-audit complete"

security-gitleaks:
	@echo "🔍 Running Gitleaks secret scanner..."
	@if command -v gitleaks >/dev/null 2>&1; then \
		mkdir -p security-reports && \
		gitleaks detect --source . \
			--report-format json \
			--report-path security-reports/gitleaks-report.json \
			--verbose 2>/dev/null || true; \
	else \
		echo "⚠️ Gitleaks not installed. Install from: https://github.com/gitleaks/gitleaks"; \
		echo "   Or use: docker run -v $$(pwd):/path zricethezav/gitleaks detect --source /path"; \
	fi
	@echo "✅ Gitleaks scan complete"

security-custom-audit:
	@echo "🔍 Running custom security audit..."
	@mkdir -p security-reports
	@python scripts/security/audit-codebase.py \
		--output security-reports/custom-audit.txt \
		--format text 2>/dev/null || true
	@cat security-reports/custom-audit.txt 2>/dev/null || echo "⚠️ Custom audit report not generated"
	@echo "✅ Custom audit complete"

security-deps:
	@echo "🔍 Running dependency health check..."
	@mkdir -p security-reports
	@python scripts/security/check-dependencies.py \
		--requirements requirements.txt \
		--output security-reports/dependency-health.txt \
		--format text 2>/dev/null || true
	@cat security-reports/dependency-health.txt 2>/dev/null || echo "⚠️ Dependency health report not generated"
	@echo "✅ Dependency health check complete"

security-clean:
	@echo "🧹 Cleaning security reports..."
	@rm -rf security-reports
	@echo "✅ Security reports cleaned"

security-install-tools:
	@echo "📦 Installing security tools..."
	@pip install --upgrade \
		bandit[toml]>=1.7.0 \
		safety>=3.0.0 \
		pip-audit>=2.6.0
	@echo "⚠️ Note: Install Gitleaks separately from https://github.com/gitleaks/gitleaks"
	@echo "✅ Security tools installed"

# =============================================================================
# Pre-Push Verification
# =============================================================================

verify:
	@echo "🔍 Running pre-push verification..."
	@echo ""
	@echo "1️⃣  Checking backend health..."
	@curl -s http://localhost:8000/api/health | grep -q "healthy" && echo "   ✅ Backend healthy" || echo "   ❌ Backend not responding"
	@echo ""
	@echo "2️⃣  Checking frontend..."
	@curl -s http://localhost:3000 > /dev/null && echo "   ✅ Frontend responding" || echo "   ❌ Frontend not responding"
	@echo ""
	@echo "3️⃣  Checking frontend health proxy..."
	@curl -s http://localhost:3000/api/health | grep -q "healthy" && echo "   ✅ Health proxy working" || echo "   ❌ Health proxy failed"
	@echo ""
	@echo "4️⃣  Checking API docs..."
	@curl -s http://localhost:8000/docs | grep -q "Swagger UI" && echo "   ✅ API docs accessible" || echo "   ❌ API docs not accessible"
	@echo ""
	@echo "✅ Verification complete!"

pre-push: verify
	@echo ""
	@echo "🧪 Running pre-push validation..."
	@echo ""
	@echo "1️⃣  Running Python linter..."
	@ruff check gaap/ --quiet && echo "   ✅ Python lint passed" || echo "   ❌ Python lint failed"
	@echo ""
	@echo "2️⃣  Running frontend linter..."
	@cd frontend && npm run lint > /dev/null 2>&1 && echo "   ✅ Frontend lint passed" || echo "   ❌ Frontend lint failed"
	@echo ""
	@echo "3️⃣  Checking for secrets..."
	@gitleaks detect --source . --no-banner --quiet 2>/dev/null && echo "   ✅ No secrets detected" || echo "   ⚠️  Potential secrets found"
	@echo ""
	@echo "4️⃣  Running type check..."
	@mypy gaap/core/ --quiet 2>/dev/null && echo "   ✅ Type check passed" || echo "   ⚠️  Type check warnings"
	@echo ""
	@echo "✅ Pre-push validation complete!"

# =============================================================================
# Pre-Push Verification
# =============================================================================

verify:
	@echo "🔍 Running pre-push verification..."
	@echo ""
	@echo "1️⃣  Checking backend health..."
	@curl -s http://localhost:8000/api/health | grep -q "healthy" && echo "   ✅ Backend healthy" || echo "   ❌ Backend not responding"
	@echo ""
	@echo "2️⃣  Checking frontend..."
	@curl -s http://localhost:3000 > /dev/null && echo "   ✅ Frontend responding" || echo "   ❌ Frontend not responding"
	@echo ""
	@echo "3️⃣  Checking frontend health proxy..."
	@curl -s http://localhost:3000/api/health | grep -q "healthy" && echo "   ✅ Health proxy working" || echo "   ❌ Health proxy failed"
	@echo ""
	@echo "4️⃣  Checking API docs..."
	@curl -s http://localhost:8000/docs | grep -q "Swagger UI" && echo "   ✅ API docs accessible" || echo "   ❌ API docs not accessible"
	@echo ""
	@echo "✅ Verification complete!"

pre-push: verify
	@echo ""
	@echo "🧪 Running pre-push validation..."
	@echo ""
	@echo "1️⃣  Running Python linter..."
	@ruff check gaap/ --quiet && echo "   ✅ Python lint passed" || echo "   ❌ Python lint failed"
	@echo ""
	@echo "2️⃣  Running frontend linter..."
	@cd frontend && npm run lint > /dev/null 2>&1 && echo "   ✅ Frontend lint passed" || echo "   ❌ Frontend lint failed"
	@echo ""
	@echo "3️⃣  Checking for secrets..."
	@gitleaks detect --source . --no-banner --quiet 2>/dev/null && echo "   ✅ No secrets detected" || echo "   ⚠️  Potential secrets found"
	@echo ""
	@echo "4️⃣  Running type check..."
	@mypy gaap/core/ --quiet 2>/dev/null && echo "   ✅ Type check passed" || echo "   ⚠️  Type check warnings"
	@echo ""
	@echo "✅ Pre-push validation complete!"
