# GAAP Examples

This directory contains example scripts demonstrating various GAAP features.

## Examples

| File | Description |
|------|-------------|
| `01_quick_start.py` | Basic usage and simple chat |
| `02_observability.py` | Tracing and metrics collection |
| `03_rate_limiting.py` | Rate limiting strategies |
| `04_custom_provider.py` | Creating custom providers |
| `05_error_handling.py` | Error handling and self-healing |
| `06_complete_workflow.py` | **Complete real-world workflow examples** |
| `07_testing_guide.py` | **Testing infrastructure and advanced patterns** |

## Running Examples

```bash
# Activate virtual environment
source .venv/bin/activate

# Run an example
python examples/01_quick_start.py
python examples/02_observability.py
python examples/03_rate_limiting.py
python examples/04_custom_provider.py
python examples/05_error_handling.py
python examples/06_complete_workflow.py
python examples/07_testing_guide.py
```

## Prerequisites

1. Install GAAP in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

2. Set up API keys in `.gaap_env`:
   ```bash
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=...
   ```

## Example Descriptions

### 01_quick_start.py
- Creating a GAAP engine
- Processing simple requests
- Basic chat functionality

### 02_observability.py
- OpenTelemetry tracing
- Prometheus metrics
- Recording LLM calls and healing attempts

### 03_rate_limiting.py
- Token Bucket limiter
- Sliding Window limiter
- Adaptive rate limiter
- Waiting for available tokens

### 04_custom_provider.py
- Creating a custom provider class
- Implementing required methods
- Registering the provider

### 05_error_handling.py
- GAAP exception types
- Healing levels
- Retry logic patterns
- Error serialization

### 06_complete_workflow.py (NEW 🎉)
- **Example 1**: Simple code generation (Binary Search)
- **Example 2**: Multi-step project creation (REST API)
- **Example 3**: Code review and refactoring
- **Example 4**: Debugging with self-healing
- **Example 5**: Memory and learning from experience
- **Example 6**: Provider comparison (Groq vs Gemini)

### 07_testing_guide.py (NEW 🎉)
- **Test 1**: Basic engine with mock provider
- **Test 2**: Security firewall testing
- **Test 3**: Error handling and self-healing
- **Test 4**: Hierarchical memory system
- **Test 5**: Multi-step workflow
- **Pattern 1**: Retry with exponential backoff
- **Pattern 2**: Circuit breaker