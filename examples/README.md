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