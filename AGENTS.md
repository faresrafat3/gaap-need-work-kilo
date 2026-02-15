# AGENTS.md - GAAP Development Guide

## Build/Lint/Test Commands

### Setup
```bash
source .venv/bin/activate
pip install -e ".[dev]"
pip install streamlit pandas plotly
```

### Testing
```bash
pytest                          # All tests
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests
pytest -k "provider" -v         # Tests matching pattern
pytest --cov=gaap --cov-report=term-missing

# Single test file
pytest tests/unit/test_router.py -v

# Single test function
pytest tests/unit/test_router.py::test_route_selection -v

# Single test class
pytest tests/unit/test_healing.py::TestSelfHealing -v
```

### Linting & Formatting
```bash
black gaap/ tests/              # Format code
isort gaap/ tests/              # Sort imports
ruff check gaap/ tests/ --fix   # Lint with auto-fix
mypy gaap/                      # Type check
make check                      # Run all checks
```

### CLI & Web
```bash
gaap --help                     # CLI help
gaap chat "Hello"               # Quick chat
gaap interactive                # Interactive mode
gaap web                        # Start web UI
gaap doctor                     # Run diagnostics
```

## Project Structure
```
gaap/
├── core/           # Types, base classes, config, exceptions
├── layers/         # 4-layer architecture (L0-L3)
├── providers/      # LLM providers (Groq, Gemini, G4F)
├── routing/        # Smart routing & fallback
├── security/       # Prompt firewall
├── healing/        # Self-healing (5 levels)
├── memory/         # Hierarchical memory (L1-L4)
├── storage/        # JSON storage
├── cli/            # CLI commands
└── web/            # Streamlit web UI

tests/
├── unit/           # Unit tests (172 tests)
├── integration/    # Integration tests
└── benchmarks/     # Performance benchmarks
```

## Code Style

### Imports (isort profile: black)
```python
# Standard library
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Third-party
from aiohttp import ClientSession

# Local (use gaap prefix)
from gaap.core.types import Task, TaskResult
from gaap.core.exceptions import GAAPException
```

### Naming
- **Classes**: PascalCase (`BaseProvider`, `TaskResult`)
- **Functions/Methods**: snake_case (`chat_completion`, `get_stats`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_RETRIES`, `MODEL_COSTS`)
- **Private**: underscore prefix (`_logger`, `_memory`)
- **Type variables**: single letter (`T`, `R`)

### Types & Dataclasses
```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum, auto

@dataclass
class TaskResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
```

### Async Patterns
```python
async def process(self, request: Request) -> Response:
    async with asyncio.timeout(self.timeout):
        result = await self._execute(request)
    return result

results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Error Handling
```python
from gaap.core.exceptions import ProviderError, ProviderRateLimitError

try:
    result = await provider.chat_completion(messages, model)
except ProviderRateLimitError as e:
    if e.recoverable:
        result = await self._heal_and_retry(e)
    else:
        raise ProviderError(f"Failed: {e}") from e
```

### Logging
```python
import logging
logger = logging.getLogger("gaap.provider.groq")
logger.debug(f"Processing with model {model}")
logger.info(f"Provider initialized")
logger.warning(f"Rate limit: {current}/{limit}")
logger.error(f"Request failed: {error}")
```

### Testing
```python
import pytest

@pytest.mark.asyncio
async def test_provider(mock_provider):
    result = await mock_provider.chat_completion(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="llama-3.3-70b"
    )
    assert result.success
```

## Environment

Create `.gaap_env` file:
```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=...
```

## Architecture

| Layer | Name | Purpose |
|-------|------|---------|
| L0 | Interface | Security, classification, routing |
| L1 | Strategic | Tree of Thoughts, MAD Panel |
| L2 | Tactical | Task decomposition, DAG |
| L3 | Execution | Task execution, quality check |

## Before Committing

1. `black gaap/ tests/` - Format
2. `isort gaap/ tests/` - Sort imports
3. `ruff check gaap/ tests/` - Lint
4. `pytest` - All tests pass
5. Never commit secrets/API keys
