# GAAP Docker Image
# Multi-stage build for optimized image size

# ==============================
# Builder Stage
# ==============================
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --upgrade pip wheel build

# Build wheel
COPY gaap/ ./gaap/
RUN python -m build --wheel

# ==============================
# Runtime Stage
# ==============================
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Install web dependencies
RUN pip install streamlit pandas plotly fastapi uvicorn

# Create non-root user
RUN useradd --create-home --shell /bin/bash gaap
USER gaap

# Copy application code
COPY --chown=gaap:gaap gaap/ ./gaap/
COPY --chown=gaap:gaap tests/ ./tests/

# Create data directory
RUN mkdir -p /home/gaap/.gaap

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command (Streamlit web UI)
CMD ["streamlit", "run", "gaap/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
