# GAAP Development Guide

This guide covers setting up a development environment and contributing to GAAP.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Documentation](#documentation)

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **pip** or **uv** package manager
- **Git**
- **Make** (optional, for shortcuts)

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

---

## Resources

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Examples](examples/)