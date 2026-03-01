# GAAP Provider Configuration

Configure AI providers to power GAAP's intelligence.

## Overview

GAAP supports multiple LLM providers with automatic failover:

| Provider | Model | Status | Use Case |
|----------|-------|--------|----------|
| **Kimi** | kimi-k2.5-thinking | Default | General coding |
| **DeepSeek** | deepseek-chat | Available | General coding |
| **GLM** | GLM-5 | Available | General coding |

## Configuration

### Environment Variables

Set provider API keys in `.env`:

```bash
# Required - at least one provider
GAAP_KILO_API_KEY=your_kimi_key
GAAP_DEEPSEEK_API_KEY=your_deepseek_key
GAAP_GLM_API_KEY=your_glm_key
```

### Provider Priority

Default priority order:
1. Kimi (primary)
2. DeepSeek (fallback)
3. GLM (final fallback)

To change priority:

```bash
# In .env
GAAP_DEFAULT_PROVIDER=deepseek
GAAP_FALLBACK_PROVIDERS=glm,kimi
```

## Provider Status

### Check Status

```bash
curl http://localhost:8000/api/providers
```

### Response

```json
[
  {
    "name": "kimi",
    "type": "WEBCHAT",
    "enabled": true,
    "priority": 1,
    "models": ["kimi-k2.5-thinking"],
    "health": "healthy",
    "stats": {
      "total_requests": 150,
      "success_rate": 0.97
    }
  }
]
```

### Test Connection

```bash
curl -X POST http://localhost:8000/api/providers/kimi/test
```

## Per-Request Provider Selection

### API

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "provider": "deepseek"
  }'
```

### Web Interface

Select provider from dropdown before sending message.

## Adding New Providers

### Via API

```bash
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "openai",
    "provider_type": "api",
    "api_key": "sk-...",
    "default_model": "gpt-4",
    "priority": 2
  }'
```

### Via Config File

```yaml
# config/providers.yml
providers:
  - name: openai
    type: api
    enabled: true
    priority: 2
    models:
      - gpt-4
      - gpt-3.5-turbo
    config:
      api_key: ${OPENAI_API_KEY}
      timeout: 60
```

## Provider Types

| Type | Description | Example |
|------|-------------|---------|
| `webchat` | Browser-based providers | Kimi, DeepSeek, GLM |
| `api` | Direct API providers | OpenAI, Anthropic |
| `local` | Local models | Ollama, llama.cpp |
| `bridge` | Adapter providers | Custom wrappers |

## Fallback Behavior

When primary provider fails:

1. Log the failure
2. Try next provider in priority order
3. Return success from first working provider
4. Report error if all fail

```python
# Example fallback sequence
preferred = "kimi"  # Fails
try "deepseek"      # Succeeds
return deepseek_response
```

## Health Monitoring

### Automatic Health Checks

GAAP monitors provider health:
- Success rate tracking
- Latency measurement
- Consecutive failure counting

### Manual Health Check

```bash
curl http://localhost:8000/api/health
```

## Cost Tracking

Track usage per provider:

```bash
curl http://localhost:8000/api/system/metrics
```

Response includes:
```json
{
  "providers": {
    "kimi": {
      "requests": 100,
      "tokens": 50000,
      "estimated_cost": 2.50
    }
  }
}
```

## Troubleshooting

### Provider Not Available

```bash
# Check status
curl http://localhost:8000/api/providers

# Test specific provider
curl -X POST http://localhost:8000/api/providers/kimi/test
```

### All Providers Failed

Common causes:
- Missing API keys
- Network connectivity issues
- Provider rate limiting

Check:
```bash
# Verify API keys
echo $GAAP_KILO_API_KEY

# Test network
curl -I https://api.provider.com
```

### Switching Default Provider

```bash
# In .env or environment
export GAAP_DEFAULT_PROVIDER=deepseek

# Restart GAAP
```

## Security

- Store API keys in environment variables
- Never commit keys to version control
- Use separate keys for dev/production
- Rotate keys regularly

## See Also

- [API Documentation](../api/README.md)
- [Configuration Guide](../developers/setup.md)
