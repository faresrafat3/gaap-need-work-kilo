# GAAP - General-purpose AI Architecture Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A sophisticated multi-layer LLM orchestration system designed for robust, self-healing AI task processing.

## Features

- **4-Layer Architecture**: Interface → Strategic → Tactical → Execution
- **Self-Healing**: 5-level recovery (Retry → Refine → Pivot → Strategy → Human)
- **MAD Panels**: Multi-Agent Debate for quality assurance
- **Smart Routing**: Automatic provider selection based on priority/cost
- **Hierarchical Memory**: L1-L4 memory tiers
- **Observability**: OpenTelemetry tracing + Prometheus metrics
- **Rate Limiting**: Token Bucket, Sliding Window, Adaptive strategies
- **CLI**: Full-featured command-line interface
- **Web UI**: Streamlit-based web interface

## Installation

```bash
# Clone the repository
git clone https://github.com/gaap-system/gaap.git
cd gaap

# Quick setup
make dev

# Or manual setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install streamlit pandas plotly
```

## Quick Start

### CLI Usage

```bash
# Quick chat
gaap chat "Write a Python function for binary search"

# Interactive mode
gaap interactive

# Manage providers
gaap providers list
gaap providers test groq

# View models
gaap models list
gaap models tiers

# Configuration
gaap config show
gaap config set default_budget 20.0

# History
gaap history list
gaap history search "binary"

# System
gaap status
gaap version
gaap doctor

# Web UI
gaap web
```

### Web UI

```bash
# Start web interface
gaap web

# Or directly
streamlit run gaap/web/app.py
```

Open http://localhost:8501 in your browser.

### Python API

```python
import asyncio
from gaap import GAAPEngine, GAAPRequest

async def main():
    engine = GAAPEngine(budget=10.0)
    
    request = GAAPRequest(
        text="Write a Python function for binary search",
        priority="NORMAL"
    )
    response = await engine.process(request)
    
    print(response.output)
    print(f"Quality Score: {response.quality_score}")
    print(f"Cost: ${response.total_cost_usd:.4f}")

asyncio.run(main())
```

## Project Structure

```
gaap/
├── core/           # Types, base classes, config, exceptions
│   ├── observability.py    # OpenTelemetry & Prometheus
│   └── rate_limiter.py     # Token Bucket & Adaptive
├── layers/         # 4-layer architecture (L0-L3)
├── providers/      # LLM provider implementations
├── routing/        # Smart routing & fallback
├── security/       # Firewall/prompt injection
├── healing/        # Self-healing system
├── memory/         # Hierarchical memory
├── context/        # Context management
├── storage/        # JSON storage for history/config
├── cli/            # Command-line interface
│   └── commands/   # CLI commands
└── web/            # Streamlit web UI
    └── pages/      # Web pages
```

## Architecture

| Layer | Name | Responsibility |
|-------|------|---------------|
| L0 | Interface | Security scan, intent classification, routing |
| L1 | Strategic | Tree of Thoughts, MAD Architecture Panel |
| L2 | Tactical | Task decomposition, DAG construction |
| L3 | Execution | Task execution, quality evaluation |

## Development

```bash
# Run tests
make test

# Format and lint
make format
make lint

# Run all checks
make check

# Clean build artifacts
make clean
```

## Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or with docker-compose
docker-compose up -d
```

## Environment Variables

Create a `.gaap_env` file:

```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=...
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Documentation

- [AGENTS.md](AGENTS.md) - Development guidelines for AI agents
