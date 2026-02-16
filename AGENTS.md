# AGENTS.md - GAAP Development Guide

> **Docs:** `docs/ARCHITECTURE.md` | `docs/API_REFERENCE.md` | `docs/PROVIDERS.md` | `docs/DEVELOPMENT.md`

## Build/Lint/Test Commands

### Setup
```bash
source .venv/bin/activate
pip install -e ".[dev]"
pip install streamlit pandas plotly
```

### Testing
```bash
pytest                              # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/integration/ -v        # Integration tests
pytest -k "provider" -v             # Tests matching pattern
pytest -x                           # Stop on first failure
pytest --lf                         # Last failed tests only
pytest -m "not slow"                # Skip slow tests
pytest --cov=gaap --cov-report=term-missing

# Single test
pytest tests/unit/test_router.py::test_route_selection -v
```

### Linting & Formatting
```bash
black gaap/ tests/                  # Format code
isort gaap/ tests/                  # Sort imports
ruff check gaap/ tests/ --fix       # Lint with auto-fix
mypy gaap/ --ignore-missing-imports # Type check
make check                          # Run all checks (format + lint + typecheck + test)
```

### CLI & Web
```bash
gaap --help                         # CLI help
gaap chat "Hello"                   # Quick chat
gaap interactive                    # Interactive mode
gaap web                            # Start web UI
gaap doctor                         # Run diagnostics
```

## Project Structure
```
gaap/
├── core/           # Types, base classes, config, exceptions
├── layers/         # 4-layer architecture (L0-L3)
├── providers/      # LLM providers (Groq, Gemini, G4F, WebChat)
├── routing/        # Smart routing & fallback
├── security/       # 7-layer prompt firewall
├── healing/        # Self-healing (5 levels)
├── memory/         # 4-tier hierarchical memory
├── storage/        # JSON storage
├── cli/            # CLI commands
├── web/            # Streamlit web UI
├── api/            # FastAPI REST API
└── tools/          # Built-in tools

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── benchmarks/     # Performance benchmarks
```

## Code Style

### Imports (isort profile: black, line-length: 100)
```python
# Standard library
import asyncio
from dataclasses import dataclass, field
from typing import Any

# Third-party
from aiohttp import ClientSession

# Local (use gaap prefix)
from gaap.core.types import Task, TaskResult
from gaap.core.exceptions import GAAPException
```

### Naming Conventions
- **Classes**: PascalCase (`BaseProvider`, `TaskResult`)
- **Functions/Methods**: snake_case (`chat_completion`, `get_stats`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_RETRIES`, `MODEL_COSTS`)
- **Private**: underscore prefix (`_logger`, `_memory`)
- **Type variables**: single letter (`T`, `R`)

### Types (mypy strict mode)
```python
@dataclass
class TaskResult:
    success: bool
    output: Any
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

def process(self, request: Request) -> Response:
    ...

async def fetch(self) -> AsyncIterator[str]:
    ...
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
logger = logging.getLogger("gaap.provider.groq")
logger.info(f"Processing with model {model}")
```

### Testing
```python
@pytest.mark.asyncio
async def test_provider(mock_provider):
    result = await mock_provider.chat_completion(
        messages=[Message(role=MessageRole.USER, content="test")],
        model="llama-3.3-70b"
    )
    assert result.success
```

## Architecture

| Layer | Name | Purpose |
|-------|------|---------|
| L0 | Interface | Security, classification, routing |
| L1 | Strategic | Tree of Thoughts, MAD Panel |
| L2 | Tactical | Task decomposition, DAG |
| L3 | Execution | Task execution, quality check |

## Environment

Create `.gaap_env` file:
```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
CEREBRAS_API_KEY=...
```

## Before Committing

1. `make check` - Run all checks (recommended)
2. Or manually: `black gaap/ tests/` → `isort gaap/ tests/` → `ruff check gaap/ tests/` → `pytest`
3. Never commit secrets/API keys

---

## Kilo Knowledge System

> نظام التوثيق المعرفي - يتعلم Kilo من كل جلسة

### Commands
| Command | Description |
|---------|-------------|
| `/compact` | حفظ المعرفة وتنظيف السياق |
| `/knowledge add` | إضافة درس/حل جديد |
| `/knowledge search <query>` | بحث في المعرفة |
| `/wisdom` | أهم الدروس المستفادة |

### Key Lessons
1. **لا تستخدم `# mypy: ignore-errors`** - حل سطحي يخفي المشكلة
2. **تحويل صريح**: `int(len(x) * 1.5)` وليس `len(x) * 1.5`
3. **إضافة return type**: `def foo() -> None:` لكل دالة

### Storage
- `~/.kilo/knowledge/` - معرفة عامة
- `.kilo/` - معرفة المشروع