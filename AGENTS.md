# AGENTS.md - GAAP System Development Guide

## Build/Lint/Test Commands

### Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run single test file
pytest tests/unit/test_observability_rate_limiter.py -v

# Run with coverage
pytest --cov=gaap --cov-report=term-missing

# Run tests matching pattern
pytest -k "provider" -v
```

### Linting & Type Checking
```bash
# Format code with black
black gaap/ tests/

# Sort imports with isort
isort gaap/ tests/

# Lint with ruff
ruff check gaap/ tests/

# Type check with mypy
mypy gaap/

# Run all checks
black --check gaap/ tests/ && isort --check gaap/ tests/ && ruff check gaap/ tests/
```

### Benchmarks
```bash
# Run benchmarks
python tests/benchmarks/test_providers.py

# Multi-provider benchmark
python tests/benchmarks/multi_provider_bench.py --dataset mmlu --samples 10
```

## Project Structure

```
gaap/
├── __init__.py              # Main exports
├── gaap_engine.py            # Core engine
├── core/
│   ├── types.py             # Data types, enums, dataclasses
│   ├── base.py              # Base classes (Agent, Provider, Layer)
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── observability.py     # OpenTelemetry & Prometheus
│   └── rate_limiter.py      # Token Bucket & Adaptive rate limiting
├── layers/                   # 4-layer architecture
│   ├── layer0_interface.py  # Security + Classification + Routing
│   ├── layer1_strategic.py  # ToT + MAD Architecture Panel
│   ├── layer2_tactical.py   # Task Decomposition + DAG
│   └── layer3_execution.py  # Executor Pool + Quality Panel
├── providers/                # LLM providers
│   ├── base_provider.py     # Abstract provider interface
│   ├── free_tier/           # Free tier providers (Groq, Gemini)
│   └── chat_based/          # Chat-based providers (G4F)
├── routing/                  # Smart routing
│   ├── router.py            # Intelligent model selection
│   └── fallback.py          # Fallback management
├── security/                 # Security systems
│   └── firewall.py          # Prompt injection detection
├── healing/                  # Self-healing system
│   └── healer.py            # 5-level recovery
├── memory/                   # Hierarchical memory
│   └── hierarchical.py      # L1-L4 memory tiers
├── context/                  # Context management
├── api/                      # API routes
└── cli/                      # CLI interface
    └── main.py

tests/
├── unit/                     # Unit tests
├── integration/              # Integration tests
└── benchmarks/               # Performance benchmarks
```

## Code Style Guidelines

### Imports
```python
# Standard library first
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

# Third-party
from aiohttp import ClientSession

# Local imports (use gaap prefix)
from gaap.core.types import Task, TaskResult, TaskPriority
from gaap.core.exceptions import GAAPException, ProviderError
from gaap.layers.layer0_interface import Layer0Interface
```

### Naming Conventions
- **Classes**: PascalCase (`BaseProvider`, `TaskResult`, `SmartRouter`)
- **Methods/Functions**: snake_case (`chat_completion`, `get_stats`, `execute_task`)
- **Constants**: SCREAMING_SNAKE_CASE (`CONTEXT_LIMITS`, `MODEL_COSTS`, `MAX_RETRIES`)
- **Private attributes**: underscore prefix (`_logger`, `_memory`, `_is_initialized`)
- **Type variables**: single letter (`T`, `R`, `T_co`)

### Types & Dataclasses
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, auto

@dataclass
class TaskResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
```

### Async Patterns
```python
import asyncio

async def chat_completion(
    self,
    messages: List[Message],
    model: str,
    **kwargs
) -> ChatCompletionResponse:
    async with asyncio.timeout(self.timeout):
        result = await self._make_request(messages, model)
    return result

# Use asyncio.gather for parallel execution
results = await asyncio.gather(*tasks)
```

### Error Handling
```python
from gaap.core.exceptions import (
    ProviderError, ProviderRateLimitError, ProviderTimeoutError
)

try:
    result = await provider.chat_completion(messages, model)
except ProviderRateLimitError as e:
    if e.recoverable:
        result = await self._heal_and_retry(e)
    else:
        raise
```

### Logging
```python
import logging

logger = logging.getLogger("gaap.provider.groq")

logger.debug(f"Processing request with model {model}")
logger.info(f"Provider {name} initialized")
logger.warning(f"Rate limit approaching: {current}/{limit}")
logger.error(f"Request failed: {error}")
```

### Testing Patterns
```python
import pytest

@pytest.mark.asyncio
async def test_provider_connection():
    from gaap.providers.free_tier import GroqProvider
    from gaap.core.types import Message, MessageRole
    
    provider = GroqProvider(api_key="test-key")
    result = await provider.chat_completion(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="llama-3.3-70b"
    )
    assert result.success
```

## Environment Variables

Required for provider operations:
```bash
# .gaap_env file format
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GITHUB_TOKEN=github_pat_...
```

## Key Architecture Notes

1. **4-Layer System**: Interface → Strategic → Tactical → Execution
2. **Self-Healing**: 5 levels (Retry → Refine → Pivot → Strategy → Human)
3. **MAD Panels**: Multi-Agent Debate for quality assurance
4. **Smart Routing**: Automatic provider selection based on priority/cost
5. **Hierarchical Memory**: L1-L4 memory tiers (Working → Episodic → Semantic → Procedural)

## Before Committing

1. Run `black gaap/ tests/` to format code
2. Run `isort gaap/ tests/` to sort imports
3. Run `ruff check gaap/ tests/` to lint
4. Run `mypy gaap/` for type checking
5. Run `pytest` to ensure tests pass
6. Never commit API keys or secrets