# Contributing to GAAP

Thank you for your interest in contributing to GAAP! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be considerate, welcoming, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/gaap-system/gaap/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details (Python version, OS)
   - Relevant logs or error messages

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue with:
   - Clear use case description
   - Why it would benefit the project
   - Possible implementation approach

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear message
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gaap.git
cd gaap

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
pip install streamlit pandas plotly

# Setup pre-commit hooks
pre-commit install
```

## Code Style

We use the following tools for code quality:

### Formatting

```bash
# Format code
black gaap/ tests/

# Sort imports
isort gaap/ tests/
```

### Linting

```bash
# Run ruff
ruff check gaap/ tests/ --fix

# Type checking
mypy gaap/
```

### Code Style Guidelines

- **Line length**: 100 characters maximum
- **Imports**: Use `isort` with black profile
- **Types**: All functions must have type hints
- **Docstrings**: Use triple quotes for public functions/classes
- **Naming**:
  - Classes: PascalCase (`BaseProvider`, `TaskResult`)
  - Functions/Methods: snake_case (`chat_completion`, `get_stats`)
  - Constants: SCREAMING_SNAKE_CASE (`MAX_RETRIES`, `MODEL_COSTS`)
  - Private: underscore prefix (`_logger`, `_memory`)

### Example Code Style

```python
from dataclasses import dataclass, field
from typing import Any

from gaap.core.types import TaskPriority


@dataclass
class MyResult:
    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


async def process_task(task_id: str, priority: TaskPriority) -> MyResult:
    if not task_id:
        raise ValueError("task_id is required")
    
    return MyResult(
        success=True,
        output={"processed": task_id},
    )
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest --cov=gaap --cov-report=term-missing

# Specific test file
pytest tests/unit/test_router.py -v

# Specific test function
pytest tests/unit/test_router.py::test_route_selection -v
```

### Writing Tests

```python
import pytest
from gaap.core.types import Task, TaskPriority


@pytest.fixture
def sample_task():
    return Task(
        id="test-123",
        description="Test task",
        priority=TaskPriority.NORMAL,
    )


class TestMyFeature:
    @pytest.mark.asyncio
    async def test_process_task(self, sample_task):
        result = await process_task(sample_task.id, sample_task.priority)
        assert result.success
```

## Project Structure

```
gaap/
+-- core/           # Types, config, exceptions
+-- layers/         # 4-layer architecture
+-- providers/      # LLM providers
+-- routing/        # Smart routing
+-- security/       # Firewall, audit
+-- healing/        # Self-healing
+-- memory/         # Memory system
+-- cli/            # CLI commands
+-- web/            # Web UI
```

## Pull Request Process

1. **Update documentation** if you change functionality
2. **Add tests** for new features
3. **Run all checks** before submitting:
   ```bash
   make check  # or: black . && isort . && ruff check . && pytest
   ```
4. **Link issues** in your PR description
5. **Request review** from maintainers

## Release Process

1. Update `CHANGELOG.md`
2. Update version in `pyproject.toml` and `gaap/__init__.py`
3. Create a release PR
4. After merge, tag the release:
   ```bash
   git tag -a v1.x.x -m "Release v1.x.x"
   git push origin v1.x.x
   ```

## Questions?

- Open a [Discussion](https://github.com/gaap-system/gaap/discussions)
- Check [Documentation](docs/)

Thank you for contributing!
