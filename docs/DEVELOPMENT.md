# GAAP Development Guide

This guide covers setting up a development environment and contributing to GAAP.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Frontend Development](#frontend-development)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Running the Full Stack](#running-the-full-stack)
8. [Common Tasks](#common-tasks)
9. [Debugging](#debugging)
10. [Documentation](#documentation)
11. [Troubleshooting](#troubleshooting)
12. [Code Quality Standards](#code-quality-standards)
13. [Resources](#resources)

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **pip** or **uv** package manager
- **Git**
- **Make** (optional, for shortcuts)

### Frontend Prerequisites

- **Node.js 18+**
- **npm** or **yarn**
- **Modern browser**

---

## Setup

### Quick Setup

```bash
# Clone repository
git clone https://github.com/gaap-system/gaap.git
cd gaap

# Run setup script
./setup_dev.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows

# Install package with dev dependencies
pip install -e ".[dev]"

# Install additional dependencies
pip install streamlit pandas plotly

# Setup pre-commit hooks
pre-commit install
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create a `.gaap_env` file:

```bash
# API Keys (required for providers)
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk_...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GITHUB_TOKEN=ghp_...

# Optional configuration
GAAP_ENVIRONMENT=development
GAAP_LOG_LEVEL=DEBUG
```

---

## Project Structure

```
gaap/
+-- core/                   # Core types and utilities
|   +-- types.py           # Data types, enums, constants
|   +-- exceptions.py      # Exception hierarchy
|   +-- config.py          # Configuration management
|   +-- base.py            # Base classes
|   +-- rate_limiter.py    # Rate limiting strategies
|   +-- observability.py   # Tracing and metrics
|   +-- memory_guard.py    # Memory protection
|
+-- layers/                 # 4-layer architecture
|   +-- layer0_interface.py # Security, classification, routing
|   +-- layer1_strategic.py # ToT, MAD Panel
|   +-- layer2_tactical.py # Decomposition, DAG
|   +-- layer3_execution.py # Execution, quality
|
+-- providers/              # LLM providers
|   +-- base_provider.py   # Abstract base class
|   +-- free_tier/         # Free API providers
|   +-- chat_based/        # G4F provider
|   +-- webchat_providers.py # WebChat providers
|   +-- smart_router.py    # Multi-provider routing
|
+-- routing/                # Routing and fallback
|   +-- router.py          # Smart router
|   +-- fallback.py        # Fallback manager
|
+-- security/               # Security features
|   +-- firewall.py        # Prompt firewall
|   +-- audit.py           # Audit trail
|
+-- healing/                # Self-healing system
|   +-- healer.py          # Healing implementation
|
+-- memory/                 # Memory system
|   +-- hierarchical.py    # 4-tier memory
|
+-- cli/                    # Command-line interface
|   +-- main.py            # Entry point
|   +-- commands/          # CLI commands
|
+-- web/                    # Web UI
|   +-- app.py             # Streamlit app
|   +-- pages/             # UI pages
|
+-- api/                    # REST API
|   +-- fastapi_app.py     # FastAPI app
|   +-- routes.py          # API routes
|
+-- storage/                # Persistence
|   +-- json_store.py      # JSON storage
|
+-- context/                # Context management
+-- cache/                  # Caching system
+-- validators/             # Code validators
+-- mad/                    # Multi-Agent Debate
+-- meta_learning/          # Meta-learning

frontend/
+-- src/
    +-- app/              # 10 Next.js pages
    +-- components/
        +-- common/       # 17 reusable components
        +-- dashboard/    # Dashboard widgets
        +-- layout/       # Header, Sidebar
        +-- ooda/         # OODA visualization
        +-- steering/     # Steering controls
    +-- stores/           # 5 Zustand stores
    +-- hooks/            # 3 custom hooks
    +-- lib/              # API client, types
```

---

## Code Style

### Formatting Tools

```bash
# Format code with black
black gaap/ tests/

# Sort imports with isort
isort gaap/ tests/

# Lint with ruff
ruff check gaap/ tests/ --fix

# Type check with mypy
mypy gaap/
```

### Style Guidelines

#### Imports

```python
# Standard library
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

# Third-party
from aiohttp import ClientSession

# Local (use gaap prefix)
from gaap.core.types import Task, TaskResult
from gaap.core.exceptions import GAAPException
```

#### Classes

```python
@dataclass
class MyResult:
    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Functions

```python
async def process_task(
    task: Task,
    provider: BaseProvider,
    timeout: float = 30.0,
) -> TaskResult:
    if not task.description:
        raise ValueError("Task description required")
    
    result = await provider.chat_completion(
        messages=[Message(role=MessageRole.USER, content=task.description)],
        model="llama-3.3-70b"
    )
    
    return TaskResult(
        success=True,
        output=result.choices[0].message.content
    )
```

#### Enums

```python
class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
```

#### Constants

```python
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 30.0
MODEL_COSTS: Final[dict[str, float]] = {
    "gpt-4o": 2.50,
}
```

---

## Frontend Development

### Code Style

- **ESLint** for linting
- **Prettier** for formatting
- **TypeScript** strict mode enabled

```bash
cd frontend
npm run lint
npm run format
npm run typecheck
```

### Component Development

Components are organized by domain:
- `common/` - Reusable UI components (buttons, inputs, modals)
- `dashboard/` - Dashboard-specific widgets
- `layout/` - Header, Sidebar, and navigation
- `ooda/` - OODA loop visualization components
- `steering/` - Steering control components

### State Management

State is managed using **Zustand** stores located in `src/stores/`:
- `useTaskStore` - Task management state
- `useUIStore` - UI state (sidebar, modals)
- `useSettingsStore` - User preferences
- `useWebSocketStore` - Real-time connection state
- `useAuthStore` - Authentication state

### API Integration

- **React Query** for server state and caching
- **WebSocket** for real-time updates

```typescript
const { data, isLoading } = useQuery({
  queryKey: ['tasks'],
  queryFn: fetchTasks,
});

const socket = useWebSocket('/ws/tasks');
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_router.py -v

# Specific test function
pytest tests/unit/test_router.py::test_route_selection -v

# With coverage
pytest --cov=gaap --cov-report=term-missing

# Fast tests only (skip slow)
pytest -m "not slow"
```

### Test Organization

```
tests/
+-- conftest.py             # Fixtures
+-- unit/                   # Unit tests
|   +-- test_router.py
|   +-- test_layer0_interface.py
|   +-- test_layer1_strategic.py
|   +-- test_layer2_tactical.py
|   +-- test_layer3_execution.py
|   +-- test_healing.py
|   +-- test_memory.py
+-- integration/            # Integration tests
|   +-- test_full_architecture.py
+-- benchmarks/             # Performance benchmarks
```

### Writing Tests

```python
import pytest
from gaap.core.types import Task, TaskPriority, TaskComplexity


@pytest.fixture
def sample_task():
    return Task(
        id="test-123",
        description="Test task",
        priority=TaskPriority.NORMAL,
        complexity=TaskComplexity.SIMPLE,
    )


class TestMyFeature:
    @pytest.mark.asyncio
    async def test_process_task(self, sample_task, mock_provider):
        result = await process_task(sample_task, mock_provider)
        
        assert result.success
        assert result.output is not None
        assert result.error is None
    
    def test_task_creation(self):
        task = Task(description="Test")
        
        assert task.priority == TaskPriority.NORMAL
        assert task.status == ExecutionStatus.PENDING
```

### Test Fixtures

Available in `conftest.py`:

- `sample_task` - Basic task
- `sample_tasks` - List of tasks
- `mock_provider` - Mock LLM provider
- `mock_router` - Mock router
- `mock_memory` - Mock memory
- `healing_context` - Healing test context
- `safe_input` / `malicious_input` - Security test inputs

### Frontend Testing

```bash
cd frontend

# Unit tests
npm test

# E2E tests
npm run test:e2e

# Coverage report
npm run test:coverage
```

**Testing Stack:**
- **Jest** for unit tests
- **React Testing Library** for component tests
- **Playwright** for E2E tests

---

## Debugging

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger
logger = logging.getLogger("gaap.engine")
logger.debug("Processing request")
logger.info("Request completed")
logger.warning("Rate limit approaching")
logger.error("Request failed: %s", error)
```

### Debug Mode

```bash
# Enable debug logging
export GAAP_LOG_LEVEL=DEBUG

# Run with debug
gaap chat "test" --debug
```

### Interactive Debugging

```python
import asyncio
from gaap import GAAPEngine, GAAPRequest

async def debug_main():
    engine = GAAPEngine(budget=100.0)
    
    # Enable all debug output
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    request = GAAPRequest(text="Debug test")
    response = await engine.process(request)
    
    # Inspect response
    print(f"Intent: {response.intent}")
    print(f"Task Graph: {response.task_graph}")
    print(f"Results: {response.execution_results}")

asyncio.run(debug_main())
```

### Memory Profiling

```python
from gaap.core.memory_guard import get_rss_mb

print(f"Memory usage: {get_rss_mb():.0f} MB")
```

---

## Documentation

### Building Docs

```bash
# Generate API docs (if using Sphinx)
cd docs && make html

# Or view markdown docs directly
cat docs/ARCHITECTURE.md
```

### Writing Documentation

```python
async def process_task(
    task: Task,
    provider: BaseProvider,
    timeout: float = 30.0,
) -> TaskResult:
    """
    Process a single task using the specified provider.
    
    Args:
        task: The task to process.
        provider: The LLM provider to use.
        timeout: Maximum execution time in seconds.
    
    Returns:
        TaskResult containing the output and metrics.
    
    Raises:
        TaskTimeoutError: If execution exceeds timeout.
        ProviderError: If provider fails.
    
    Example:
        >>> task = Task(description="Write hello world")
        >>> result = await process_task(task, provider)
        >>> print(result.output)
        print("Hello, World!")
    """
```

---

## Running the Full Stack

### Backend

```bash
uvicorn gaap.api.main:app --reload
```

### Frontend

```bash
cd frontend && npm run dev
```

### Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Common Tasks

### Adding a New Provider

1. Create provider file in `gaap/providers/`
2. Inherit from `BaseProvider`
3. Implement required methods
4. Add tests
5. Update documentation

### Adding a New CLI Command

1. Create command file in `gaap/cli/commands/`
2. Implement `cmd_<name>` function
3. Register in `gaap/cli/main.py`
4. Add tests
5. Update CLI guide

### Adding a New Layer

1. Create layer file in `gaap/layers/`
2. Inherit from `BaseLayer`
3. Implement `process()` method
4. Integrate in `GAAPEngine`
5. Add tests
6. Update architecture docs

### Adding a New Page (Frontend)

1. Create page file in `frontend/src/app/`
2. Add route configuration if needed
3. Create associated components
4. Add tests

### Adding a New Component (Frontend)

1. Create component in appropriate directory under `src/components/`
2. Export from barrel file (`index.ts`)
3. Add TypeScript types for props
4. Add tests

### Adding a New Store (Frontend)

1. Create store file in `frontend/src/stores/`
2. Define state interface
3. Implement actions and selectors
4. Export from `stores/index.ts`

### Adding a New API Endpoint (Frontend)

1. Add API function in `frontend/src/lib/api.ts`
2. Add types in `frontend/src/lib/types.ts`
3. Create React Query hook if needed
4. Add error handling

---

## Release Process

1. Update version in `pyproject.toml` and `gaap/__init__.py`
2. Update `CHANGELOG.md`
3. Run all tests: `make check`
4. Create release PR
5. After merge, tag release:
   ```bash
   git tag -a v1.x.x -m "Release v1.x.x"
   git push origin v1.x.x
   ```
6. Build and publish:
   ```bash
   python -m build
   twine upload dist/*
   ```

---

## Troubleshooting

### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e ".[dev]"
```

### Type Errors

```bash
# Run mypy to see all type issues
mypy gaap/ --ignore-missing-imports
```

### Test Failures

```bash
# Run with verbose output
pytest tests/unit/test_file.py -v --tb=long

# Run specific test
pytest tests/unit/test_file.py::TestClass::test_method -v
```

### Pre-commit Failures

```bash
# Run pre-commit on all files
pre-commit run --all-files

# Skip pre-commit (not recommended)
git commit --no-verify
```

### Frontend Troubleshooting

#### Node Module Issues

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### Type Errors

```bash
# Check TypeScript errors
npm run typecheck

# Regenerate types if API changed
npm run generate:types
```

#### Build Errors

```bash
# Clear Next.js cache
rm -rf .next
npm run build
```

#### WebSocket Connection Issues

```bash
# Check backend is running
curl http://localhost:8000/health

# Check WebSocket endpoint
wscat -c ws://localhost:8000/ws
```

#### Hot Reload Not Working

```bash
# Restart dev server
npm run dev

# Check for file watcher limits (Linux)
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## Code Quality Standards

### Type Hints

All new code must include type hints. Target: 80%+ coverage.

```python
# ✅ Good
async def process(
    request: GAAPRequest,
    timeout: float = 30.0,
) -> GAAPResponse | None:
    ...

# ❌ Bad
async def process(request, timeout=30.0):
    ...
```

### Error Handling

Never use silent exception handling. Always log or propagate.

```python
# ✅ Good: Specific exceptions with logging
try:
    result = await api.call()
except RateLimitError as e:
    logger.warning(f"Rate limited: {e}")
    return await retry_with_backoff()
except APIError as e:
    logger.error(f"API error: {e}")
    raise

# ❌ Bad: Silent catch-all
try:
    result = await api.call()
except Exception:
    pass
```

### Nesting

Maximum nesting level: 5. Extract helper functions for complex logic.

```python
# ✅ Good: Extracted helper
async def process_request(request: Request) -> Result:
    if not request.valid:
        return Result.error("Invalid request")
    
    validated = self._validate_request(request)
    return self._execute(validated)

def _validate_request(self, request: Request) -> ValidatedRequest:
    ...

def _execute(self, request: ValidatedRequest) -> Result:
    ...

# ❌ Bad: Deep nesting
async def process_request(request):
    if request:
        if request.valid:
            if request.data:
                for item in request.data:
                    if item:
                        # 5+ levels deep!
                        ...
```

### Constants

Define constants at module level, not inline magic numbers.

```python
# ✅ Good
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 30.0
MEMORY_LIMIT_MB: Final[int] = int(os.getenv("GAAP_MEMORY_LIMIT_MB", "4000"))

# ❌ Bad
if memory > 4000:  # Magic number
    ...
```

### Docstrings

Use Google-style docstrings with Args, Returns, Raises, and Example.

```python
async def process_task(
    task: Task,
    provider: BaseProvider,
    timeout: float = 30.0,
) -> TaskResult:
    """
    Process a single task using the specified provider.
    
    Args:
        task: The task to process.
        provider: The LLM provider to use.
        timeout: Maximum execution time in seconds.
    
    Returns:
        TaskResult containing the output and metrics.
    
    Raises:
        TaskTimeoutError: If execution exceeds timeout.
        ProviderError: If provider fails.
    
    Example:
        >>> task = Task(description="Write hello world")
        >>> result = await process_task(task, provider)
        >>> print(result.output)
        print("Hello, World!")
    """
```

### Quality Checklist

Before submitting a PR, ensure:

- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No `except Exception: pass` patterns
- [ ] Nesting level ≤ 5
- [ ] No magic numbers (use constants)
- [ ] Tests pass: `pytest tests/`
- [ ] Linting passes: `ruff check gaap/`
- [ ] Type check passes: `mypy gaap/`

### Frontend Quality Standards

#### TypeScript Strict Mode

All TypeScript code must pass strict mode checks:

```typescript
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true
  }
}
```

#### Component Props Typing

All component props must be explicitly typed:

```typescript
interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  variant = 'primary',
  disabled = false,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`btn btn-${variant}`}
    >
      {label}
    </button>
  );
};
```

#### Accessibility Requirements

All components must meet WCAG 2.1 AA standards:
- All interactive elements must be keyboard accessible
- Images must have alt text
- Forms must have proper labels
- Color contrast must meet minimum ratios
- Focus states must be visible

### Frontend Quality Checklist

Before submitting a PR, ensure:

- [ ] TypeScript compiles without errors
- [ ] ESLint passes: `npm run lint`
- [ ] Components have proper prop types
- [ ] Accessibility requirements met
- [ ] Unit tests pass: `npm test`
- [ ] No console errors in browser

---

## Resources

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Examples](../examples/)
- [Evolution Plan 2026](evolution_plan_2026/)