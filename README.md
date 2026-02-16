# GAAP - General-purpose AI Architecture Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**GAAP** is a sophisticated multi-layer LLM orchestration system designed for robust, self-healing AI task processing with intelligent routing, quality assurance, and comprehensive security.

## Key Features

- **4-Layer Architecture**: Interface -> Strategic -> Tactical -> Execution pipeline
- **Self-Healing System**: 5-level automatic recovery (Retry -> Refine -> Pivot -> Strategy -> Human)
- **MAD Panels**: Multi-Agent Debate for quality assurance with 6 critic types
- **Smart Routing**: Automatic provider selection based on priority, cost, and performance
- **Hierarchical Memory**: 4-tier memory system (Working, Episodic, Semantic, Procedural)
- **Security First**: 7-layer prompt firewall with audit trail
- **Multiple Providers**: Groq, Cerebras, Gemini, Mistral, G4F, WebChat (Kimi, DeepSeek, GLM)
- **CLI & Web UI**: Full-featured command-line interface and Streamlit dashboard

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/gaap-system/gaap.git
cd gaap
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install streamlit pandas plotly
```

### Basic Usage

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
    print(f"Quality Score: {response.quality_score:.2f}")
    print(f"Cost: ${response.total_cost_usd:.4f}")
    print(f"Time: {response.total_time_ms:.0f}ms")

asyncio.run(main())
```

### CLI Usage

```bash
# Quick chat
gaap chat "Write a binary search function"

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

# System diagnostics
gaap doctor

# Web UI
gaap web
```

## Architecture

GAAP implements a 4-layer cognitive architecture:

```
+-------------------+     +-------------------+     +-------------------+     +-------------------+
|     Layer 0       |     |     Layer 1       |     |     Layer 2       |     |     Layer 3       |
|     Interface     | --> |    Strategic      | --> |    Tactical       | --> |    Execution      |
+-------------------+     +-------------------+     +-------------------+     +-------------------+
| - Security Scan   |     | - Tree of Thoughts |     | - Task Decompose  |     | - Parallel Exec   |
| - Intent Classify |     | - MAD Panel       |     | - DAG Construction|     | - Quality Eval    |
| - Complexity Est. |     | - Architecture Gen|     | - Dependency Res. |     | - Genetic Twin    |
| - Routing Decision|     | - Consensus Build |     | - Critical Path   |     | - MAD Quality     |
+-------------------+     +-------------------+     +-------------------+     +-------------------+
```

| Layer | Name | Responsibility |
|-------|------|----------------|
| **L0** | Interface | Security scan, intent classification, complexity estimation, routing |
| **L1** | Strategic | Tree of Thoughts exploration, MAD Architecture Panel |
| **L2** | Tactical | Task decomposition, DAG construction, dependency resolution |
| **L3** | Execution | Parallel execution, Genetic Twin verification, MAD Quality Panel |

## Supported Providers

| Provider | Type | Latency | Rate Limit | Cost |
|----------|------|---------|------------|------|
| **Groq** | Free Tier | ~227ms | 30 RPM/key | Free |
| **Cerebras** | Free Tier | ~511ms | 30 RPM/key | Free |
| **Gemini** | Free Tier | ~384ms | 5 RPM/key | Free |
| **Mistral** | Free Tier | ~603ms | 60 RPM/key | Free |
| **GitHub Models** | Free Tier | ~1500ms | 15 RPM/key | Free |
| **G4F** | Free Multi | Varies | ~5 RPM | Free |
| **WebChat (Kimi)** | Web-based | ~3s | Varies | Free |
| **WebChat (DeepSeek)** | Web-based | ~2s | Varies | Free |
| **WebChat (GLM)** | Web-based | ~2s | Varies | Free |

## Self-Healing System

GAAP implements a 5-level self-healing hierarchy:

```
L1: RETRY          - Simple retry for transient errors
L2: REFINE         - Prompt refinement for syntax/logic errors
L3: PIVOT          - Model change for capability limits
L4: STRATEGY SHIFT - Task simplification for complex failures
L5: HUMAN ESCALATE - Manual intervention required
```

## Memory System

4-tier hierarchical memory:

```
L1: Working Memory    - Fast, limited (100 items) - Current context
L2: Episodic Memory   - Event history - Learning from experience
L3: Semantic Memory   - Patterns & rules - Extracted knowledge
L4: Procedural Memory - Acquired skills - Templates & procedures
```

## Project Structure

```
gaap/
+-- core/           # Types, config, exceptions, base classes
+-- layers/         # 4-layer architecture (L0-L3)
+-- providers/      # LLM provider implementations
+-- routing/        # Smart routing & fallback
+-- security/       # Firewall & audit trail
+-- healing/        # Self-healing system
+-- memory/         # Hierarchical memory
+-- context/        # Context management
+-- storage/        # JSON storage
+-- cli/            # Command-line interface
+-- web/            # Streamlit web UI
+-- api/            # FastAPI REST API
+-- cache/          # Caching system
+-- mad/            # Multi-Agent Debate
+-- validators/     # Code validators
+-- meta_learning/  # Meta-learning system
```

## Development

```bash
# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Format and lint
black gaap/ tests/
isort gaap/ tests/
ruff check gaap/ tests/ --fix
mypy gaap/

# Run all checks
make check
```

## Docker

```bash
# Build
docker build -t gaap .

# Run
docker run -p 8501:8501 -p 8080:8080 gaap

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

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system architecture
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Providers Guide](docs/PROVIDERS.md) - Provider setup and comparison
- [CLI Guide](docs/CLI_GUIDE.md) - Command-line interface reference
- [Development Guide](docs/DEVELOPMENT.md) - Development setup and guidelines
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Security Guide](docs/SECURITY.md) - Security features and best practices

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
