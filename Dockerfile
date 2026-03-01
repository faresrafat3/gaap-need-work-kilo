# GAAP Backend Dockerfile - Production Ready
# ===========================================

FROM python:3.12-slim-bookworm AS builder

# Set build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.12-slim-bookworm AS production

# Set production environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    GAAP_ENVIRONMENT=production \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r gaap && useradd -r -g gaap gaap

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=gaap:gaap gaap/ ./gaap/
COPY --chown=gaap:gaap alembic/ ./alembic/
COPY --chown=gaap:gaap alembic.ini .
COPY --chown=gaap:gaap run_backend.py .

# Create data directory
RUN mkdir -p /app/data && chown -R gaap:gaap /app

# Switch to non-root user
USER gaap

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "gaap.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
