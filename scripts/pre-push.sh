#!/bin/bash
# Pre-Push Verification Script for GAAP
# Run this before pushing to GitHub to ensure everything is working

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"

# Track failures
FAILED=0

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    FAILED=1
}

echo ""
echo "🔍 GAAP Pre-Push Verification"
echo "=============================="
echo ""

# 1. Check backend health
log_info "Checking backend health..."
if curl -s "${BACKEND_URL}/api/health" | grep -q "healthy"; then
    log_success "Backend is healthy"
else
    log_error "Backend is not responding or unhealthy"
fi
echo ""

# 2. Check frontend
log_info "Checking frontend..."
if curl -s "${FRONTEND_URL}" > /dev/null; then
    log_success "Frontend is responding"
else
    log_error "Frontend is not responding"
fi
echo ""

# 3. Check frontend health proxy
log_info "Checking frontend health proxy..."
if curl -s "${FRONTEND_URL}/api/health" | grep -q "healthy"; then
    log_success "Health proxy is working"
else
    log_warning "Health proxy may have issues (check manually)"
fi
echo ""

# 4. Check API docs
log_info "Checking API documentation..."
if curl -s "${BACKEND_URL}/docs" | grep -q "Swagger UI"; then
    log_success "API docs are accessible"
else
    log_warning "API docs may not be accessible"
fi
echo ""

# 5. Check for secrets (if gitleaks available)
log_info "Checking for secrets..."
if command -v gitleaks &> /dev/null; then
    if gitleaks detect --source . --no-banner --quiet 2>/dev/null; then
        log_success "No secrets detected"
    else
        log_warning "Potential secrets found - review with: gitleaks detect --source ."
    fi
else
    log_warning "Gitleaks not installed - install from https://github.com/gitleaks/gitleaks"
fi
echo ""

# 6. Python lint check
log_info "Running Python linter..."
if ruff check gaap/ --quiet 2>/dev/null; then
    log_success "Python lint passed"
else
    log_warning "Python lint has warnings (run: ruff check gaap/)"
fi
echo ""

# 7. Frontend lint check (if in frontend directory)
if [ -d "frontend" ]; then
    log_info "Running frontend linter..."
    if cd frontend && npm run lint > /dev/null 2>&1; then
        log_success "Frontend lint passed"
    else
        log_warning "Frontend lint has issues"
    fi
    cd ..
fi
echo ""

# 8. Check git status
log_info "Checking git status..."
if [ -z "$(git status --porcelain)" ]; then
    log_success "Working directory is clean"
else
    log_warning "You have uncommitted changes"
    git status --short
fi
echo ""

# Summary
echo "=============================="
if [ $FAILED -eq 0 ]; then
    log_success "All critical checks passed!"
    echo ""
    echo "You can now push to GitHub:"
    echo "  git push origin main"
    exit 0
else
    log_error "Some checks failed. Please fix before pushing."
    exit 1
fi
