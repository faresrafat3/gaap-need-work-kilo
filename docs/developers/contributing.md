# Contributing to GAAP

Guidelines for contributing to the GAAP project.

---

## Code Style

We use automated tools to enforce code style.

### Python

#### Formatter: Ruff

```bash
# Format code
ruff format gaap/ tests/

# Check formatting
ruff format --check gaap/ tests/
```

#### Linter: Ruff

```bash
# Run linter with auto-fix
ruff check gaap/ tests/ --fix

# Check only
ruff check gaap/ tests/
```

#### Type Checker: mypy

```bash
# Type check
mypy gaap/

# Strict mode
mypy gaap/ --strict
```

### TypeScript/JavaScript

```bash
cd frontend

# Lint
npm run lint

# Format
npm run format

# Type check
npm run typecheck
```

### Style Guidelines

#### Python

```python
# Naming conventions
class MyClass:           # PascalCase
    CONSTANT = 1         # SCREAMING_SNAKE_CASE
    
    def method_name(self):   # snake_case
        local_var = 1        # snake_case
        self._private = 2    # _prefix for private

# Type hints - REQUIRED
def process_data(data: dict[str, Any]) -> Result:
    pass

# Docstrings
def complex_function(param1: str, param2: int) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
    pass
```

#### TypeScript

```typescript
// Naming conventions
interface MyInterface { }    // PascalCase
type MyType = string;        // PascalCase
class MyClass { }            // PascalCase
const CONSTANT = 1;          // SCREAMING_SNAKE_CASE
function myFunction() { }    // camelCase
const myVariable = 1;        // camelCase

// Types - REQUIRED
function processData(data: Record<string, unknown>): Result {
  return { success: true };
}
```

---

## Testing Requirements

### Test Coverage Target

| Component | Target |
|-----------|--------|
| Core Engine | 80% |
| API Layer | 75% |
| Utilities | 70% |
| Frontend | 60% |

### Running Tests

#### Python

```bash
# All tests
pytest

# Specific test
pytest tests/unit/test_router.py::test_route_selection -v

# With coverage
pytest --cov=gaap --cov-report=term-missing

# Coverage HTML report
pytest --cov=gaap --cov-report=html
open htmlcov/index.html
```

#### TypeScript

```bash
cd frontend

# All tests
npm test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage

# E2E tests
npx playwright test
```

### Writing Tests

#### Python Test Example

```python
import pytest
from unittest.mock import Mock, patch

from gaap.core.types import Task, TaskPriority
from gaap.layers.layer0_interface import IntentClassifier


class TestIntentClassifier:
    """Test intent classification."""
    
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()
    
    def test_classify_code_generation(self, classifier):
        # Arrange
        prompt = "Write a function to calculate fibonacci"
        
        # Act
        result = classifier.classify(prompt)
        
        # Assert
        assert result.intent == "CODE_GENERATION"
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_classify_async(self, classifier):
        result = await classifier.classify_async("Debug this error")
        assert result.intent == "DEBUGGING"


def test_complex_scenario():
    """Integration test example."""
    # Given
    with patch('gaap.providers.get_provider') as mock:
        mock.return_value = Mock(complete=lambda x: "result")
        
        # When
        result = process_with_provider("input")
        
        # Then
        assert result == "result"
        mock.assert_called_once()
```

#### TypeScript Test Example

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatComponent } from './ChatComponent';

describe('ChatComponent', () => {
  it('should send message on submit', async () => {
    // Arrange
    const onSend = jest.fn();
    render(<ChatComponent onSend={onSend} />);
    
    // Act
    const input = screen.getByPlaceholderText('Type a message...');
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(screen.getByText('Send'));
    
    // Assert
    expect(onSend).toHaveBeenCalledWith('Hello');
  });
  
  it('should show loading state', () => {
    render(<ChatComponent isLoading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
});
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **E2E Tests**: Test full user flows

---

## PR Process

### Before Creating PR

1. **Run all checks:**

```bash
# Python
ruff format gaap/ tests/
ruff check gaap/ tests/ --fix
mypy gaap/
pytest

# TypeScript
cd frontend
npm run lint
npm run typecheck
npm test
```

2. **Update documentation** if needed
3. **Add tests** for new features
4. **Update CHANGELOG.md**

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally

## Related Issues
Fixes #123
```

### Review Process

1. **Automated Checks** must pass:
   - CI/CD pipeline
   - Code coverage thresholds
   - Security scans

2. **Code Review** by maintainers:
   - At least 1 approval required
   - All comments resolved
   - No merge conflicts

3. **Merge Requirements**:
   - Squash and merge preferred
   - Delete branch after merge

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting) |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `chore` | Build/process changes |
| `perf` | Performance improvements |
| `security` | Security fixes |

### Scopes

| Scope | Description |
|-------|-------------|
| `api` | API endpoints |
| `engine` | OODA engine |
| `memory` | Memory system |
| `swarm` | Swarm intelligence |
| `security` | Security features |
| `ui` | User interface |
| `docs` | Documentation |

### Examples

```bash
# Feature
git commit -m "feat(api): add session export endpoint"

# Bug fix
git commit -m "fix(engine): resolve race condition in OODA loop"

# Breaking change
git commit -m "feat(api)!: change response format for /chat

BREAKING CHANGE: response now includes metadata field"

# With scope and body
git commit -m "feat(memory): implement semantic search

- Add vector similarity search
- Integrate with OpenAI embeddings
- Add caching layer for embeddings

Closes #456"
```

### Commit Message Tips

- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Reference issues in footer
- Explain WHY, not just WHAT

---

## Development Workflow

### Setting Up

```bash
# 1. Fork on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/gaap.git
cd gaap

# 3. Add upstream remote
git remote add upstream https://github.com/gaap-system/gaap.git

# 4. Create branch
git checkout -b feature/my-feature
```

### Making Changes

```bash
# Make changes
git add .
git commit -m "feat(scope): description"

# Keep up to date
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin feature/my-feature
```

### Creating PR

1. Go to GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in PR template
5. Request review

---

## Issue Reporting

### Bug Reports

Include:
- Clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (OS, Python version, etc.)
- Logs/error messages

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered
- Additional context

---

## Questions?

- 💬 [GitHub Discussions](https://github.com/gaap-system/gaap/discussions)
- 📧 Email: dev@gaap.io

Thank you for contributing!
