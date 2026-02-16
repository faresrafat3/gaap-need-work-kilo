# GAAP Providers Guide

This document provides comprehensive information about all supported LLM providers in GAAP.

## Table of Contents

1. [Provider Overview](#provider-overview)
2. [Provider Comparison](#provider-comparison)
3. [Free Tier Providers](#free-tier-providers)
4. [Multi-Provider Setup](#multi-provider-setup)
5. [WebChat Providers](#webchat-providers)
6. [G4F Provider](#g4f-provider)
7. [Custom Providers](#custom-providers)

---

## Provider Overview

GAAP supports multiple LLM providers through a unified interface:

```
+-------------------+
|    GAAPEngine     |
+-------------------+
         |
         v
+-------------------+
|   SmartRouter     |
+-------------------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
+-----+ +--+ +--+ +--+ +-----+
|Groq | |Gem| |Mis| |G4F| |WebChat|
+-----+ +--+ +--+ +--+ +-----+
```

---

## Provider Comparison

### Performance Summary

| Provider | Type | Avg Latency | Rate Limit | Cost | Quality |
|----------|------|-------------|------------|------|---------|
| **Groq** | Free Tier | 227ms | 30 RPM/key | Free | 87% MMLU |
| **Cerebras** | Free Tier | 511ms | 30 RPM/key | Free | 87% MMLU |
| **Mistral** | Free Tier | 603ms | 60 RPM | Free | High |
| **Gemini** | Free Tier | 384ms | 5 RPM/key | Free | High |
| **GitHub Models** | Free Tier | 1500ms | 15 RPM | Free | High |
| **G4F** | Free Multi | Variable | ~5 RPM | Free | 94% MMLU |
| **Kimi (WebChat)** | Web-based | ~3s | Variable | Free | High |
| **DeepSeek (WebChat)** | Web-based | ~2s | Variable | Free | High |

### Best Use Cases

| Use Case | Recommended Provider | Reason |
|----------|---------------------|--------|
| **Speed Critical** | Groq | Fastest (227ms) |
| **Reliability** | Cerebras | Consistent performance |
| **High Volume** | Cerebras + Groq | Combined 420 RPM |
| **Complex Reasoning** | G4F (Gemini 2.5) | Best quality |
| **Cost Sensitive** | All Free | All are free tier |
| **No API Key** | G4F or WebChat | No authentication |

---

## Free Tier Providers

### Groq

**Fastest provider with excellent performance.**

#### Setup

```bash
# Get API key from console.groq.com
export GROQ_API_KEY=gsk_...
```

```python
from gaap.providers import GroqProvider

provider = GroqProvider(
    api_key="gsk_...",
    rate_limit=30  # Requests per minute
)
```

#### Available Models

| Model | Context | Speed | Best For |
|-------|---------|-------|----------|
| `llama-3.3-70b-versatile` | 128K | Fastest | General purpose |
| `llama-3.1-8b-instant` | 128K | Ultra-fast | Simple tasks |
| `qwen3-32b` | 32K | Fast | Reasoning |

#### Rate Limits

- **30 RPM** per API key
- **Unlimited** daily requests
- Can use multiple keys for higher throughput

#### Usage Example

```python
import asyncio
from gaap.providers import GroqProvider
from gaap.core.types import Message, MessageRole

async def main():
    provider = GroqProvider(api_key="gsk_...")
    
    response = await provider.chat_completion(
        messages=[
            Message(role=MessageRole.USER, content="Hello!")
        ],
        model="llama-3.3-70b-versatile"
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

### Cerebras

**Most reliable free provider.**

#### Setup

```bash
# Get API key from cerebras.ai
export CEREBRAS_API_KEY=csk_...
```

```python
from gaap.providers.free_tier import CerebrasProvider

provider = CerebrasProvider(
    api_key="csk_...",
    rate_limit=30
)
```

#### Available Models

| Model | Context | Description |
|-------|---------|-------------|
| `llama3.3-70b` | 128K | Main production model |
| `qwen3-235b-a22b-instruct` | Large | Large context model |

#### Advantages

- Consistent performance (~511ms)
- No daily limits
- Excellent for production

---

### Gemini

**High quality with large context window.**

#### Setup

```bash
export GEMINI_API_KEY=...
```

```python
from gaap.providers import GeminiProvider

# Single key
provider = GeminiProvider(api_key="...")

# Key pool for higher throughput
provider = GeminiProvider(
    api_key="primary_key",
    api_keys=["key1", "key2", "key3", "key4", "key5", "key6", "key7"]
)
```

#### Available Models

| Model | Context | Notes |
|-------|---------|-------|
| `gemini-1.5-pro` | 1M tokens | Largest context |
| `gemini-1.5-flash` | 1M tokens | Faster variant |

#### Rate Limits

- **5 RPM** per key (very restrictive)
- Use key pool for higher throughput
- Daily quota: 20 requests per key

---

### Mistral

**Balanced performance with good rate limits.**

#### Setup

```bash
export MISTRAL_API_KEY=...
```

```python
from gaap.providers.free_tier import MistralProvider

provider = MistralProvider(api_key="...")
```

#### Available Models

| Model | Context | Rate Limit |
|-------|---------|------------|
| `mistral-small-latest` | 32K | 60 RPM |
| `mistral-medium-latest` | 32K | 60 RPM |
| `mistral-large-latest` | 32K | 60 RPM |

---

### GitHub Models

**Access to OpenAI models for free.**

#### Setup

```bash
export GITHUB_TOKEN=ghp_...
```

```python
from gaap.providers.free_tier import GitHubModelsProvider

provider = GitHubModelsProvider(api_key="ghp_...")
```

#### Available Models

| Model | Notes |
|-------|-------|
| `gpt-4o` | Full GPT-4o |
| `gpt-4o-mini` | Faster variant |
| `llama-3.3-70b-instruct` | Meta model |

#### Rate Limits

- **15 RPM**
- ~1,000 requests per day

---

## Multi-Provider Setup

### UnifiedGAAPProvider

Automatic failover across providers:

```python
from gaap.providers import UnifiedGAAPProvider

provider = UnifiedGAAPProvider(
    default_provider="kimi",
    fallback_order=["kimi", "deepseek", "glm", "groq"]
)

# Automatically tries next provider on failure
response = await provider.chat_completion(messages, model)
```

### Smart Router Configuration

```python
from gaap.routing import SmartRouter, RoutingStrategy
from gaap.providers import GroqProvider, GeminiProvider

providers = [
    GroqProvider(api_key="gsk_...", priority=85),
    GeminiProvider(api_key="...", priority=40),
]

router = SmartRouter(
    providers=providers,
    strategy=RoutingStrategy.SPEED_FIRST,
    budget_limit=100.0
)

decision = await router.route(messages, task)
print(f"Selected: {decision.selected_provider}/{decision.selected_model}")
```

### Multi-Key Setup

For maximum throughput:

```python
# 7 Groq keys = 210 RPM
groq_keys = ["gsk_1", "gsk_2", "gsk_3", "gsk_4", "gsk_5", "gsk_6", "gsk_7"]

# 7 Cerebras keys = 210 RPM  
cerebras_keys = ["csk_1", "csk_2", "csk_3", "csk_4", "csk_5", "csk_6", "csk_7"]

# Combined: 420 RPM capacity
```

---

## WebChat Providers

WebChat providers use browser-based interfaces for free access.

### Kimi (Moonshot)

```python
from gaap.providers.webchat_providers import KimiWebChatProvider

provider = KimiWebChatProvider(
    session_path="./sessions/kimi",  # Session persistence
    headless=True
)

# Models
# - moonshot-v1-8k
# - moonshot-v1-32k
# - moonshot-v1-128k
```

### DeepSeek

```python
from gaap.providers.webchat_providers import DeepSeekWebChatProvider

provider = DeepSeekWebChatProvider(
    session_path="./sessions/deepseek"
)

# Models
# - deepseek-chat
# - deepseek-reasoner
```

### GLM (Zhipu)

```python
from gaap.providers.webchat_providers import GLMWebChatProvider

provider = GLMWebChatProvider(
    session_path="./sessions/glm"
)

# Models
# - glm-4
# - glm-4-plus
```

### Session Persistence

WebChat providers require session persistence:

```python
# First run - logs in and saves session
provider = KimiWebChatProvider(session_path="./sessions")
await provider.login()  # Opens browser for manual login

# Subsequent runs - uses saved session
provider = KimiWebChatProvider(session_path="./sessions")
# Automatically authenticated
```

---

## G4F Provider

G4F (GPT4Free) provides free access to multiple AI services.

### Setup

```bash
pip install g4f
```

```python
from gaap.providers.chat_based import G4FProvider

provider = G4FProvider(default_model="gemini-2.5-flash")

# No API key required!
response = await provider.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gemini-2.5-flash"
)
```

### Available Models

| Model | Provider | Quality |
|-------|----------|---------|
| `gemini-2.5-pro` | Pollinations | 100% MMLU |
| `gemini-2.5-flash` | Pollinations | 94% MMLU |
| `gpt-4o-mini` | Multiple | High |
| `deepseek-v3` | Multiple | High |
| `llama-4-scout` | Multiple | Fast |

### Advantages

- **No API keys** required
- **Multiple providers** in one interface
- **High quality** models (Gemini 2.5)
- **Completely free**

### Limitations

- **~5 RPM** rate limit
- **No SLA** - service may vary
- **Best for development** and research

---

## Custom Providers

### Creating a Custom Provider

```python
from gaap.providers.base_provider import BaseProvider
from gaap.core.types import Message, ChatCompletionResponse, ProviderType

class MyCustomProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(
            name="my_custom",
            provider_type=ProviderType.PAID,
            api_key=api_key,
            base_url=base_url,
            models=["model-a", "model-b"],
            rate_limit=60
        )
    
    async def chat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ChatCompletionResponse:
        # Implement your provider logic
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [m.to_dict() for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            data = await response.json()
        
        return self._parse_response(data)
    
    def get_available_models(self) -> list[str]:
        return self.models
    
    def get_model_info(self, model: str) -> dict:
        return {"name": model, "provider": self.name}
```

### Registering Custom Providers

```python
from gaap import GAAPEngine

engine = GAAPEngine(
    providers=[
        MyCustomProvider(api_key="...", base_url="https://api.example.com"),
        GroqProvider(api_key="gsk_..."),  # Fallback
    ]
)
```

---

## Provider Selection Strategy

### Routing Strategies

```python
from gaap.routing import RoutingStrategy

# Quality First - Best results regardless of cost
router = SmartRouter(providers, strategy=RoutingStrategy.QUALITY_FIRST)

# Cost Optimized - Cheapest acceptable option
router = SmartRouter(providers, strategy=RoutingStrategy.COST_OPTIMIZED)

# Speed First - Fastest response
router = SmartRouter(providers, strategy=RoutingStrategy.SPEED_FIRST)

# Balanced - Weigh all factors
router = SmartRouter(providers, strategy=RoutingStrategy.BALANCED)

# Smart - Context-aware decision
router = SmartRouter(providers, strategy=RoutingStrategy.SMART)
```

### Model Tiers

```python
from gaap.core.types import ModelTier

# Tier 1 - Strategic (smartest)
TIER_1_STRATEGIC: ["claude-3-5-opus", "gpt-4o", "gemini-1.5-pro"]

# Tier 2 - Tactical (balanced)
TIER_2_TACTICAL: ["gpt-4o-mini", "llama-3.3-70b", "mistral-large"]

# Tier 3 - Efficient (fast)
TIER_3_EFFICIENT: ["llama-3.1-8b", "mistral-small"]

# Tier 4 - Private (local)
TIER_4_PRIVATE: ["ollama-models", "local-models"]
```

---

## Best Practices

### 1. Use Multiple Providers

```python
providers = [
    GroqProvider(api_key="gsk_...", priority=85),    # Primary - fast
    CerebrasProvider(api_key="csk_...", priority=95), # Backup - reliable
    G4FProvider(priority=50),                         # Fallback - free
]
```

### 2. Implement Rate Limit Handling

```python
from gaap.core.exceptions import ProviderRateLimitError

try:
    response = await provider.chat_completion(messages, model)
except ProviderRateLimitError as e:
    # Wait and retry, or switch provider
    await asyncio.sleep(e.details.get("retry_after_seconds", 60))
    response = await fallback_provider.chat_completion(messages, model)
```

### 3. Use Key Pools for High Volume

```python
# Distribute load across multiple keys
gemini = GeminiProvider(
    api_key=keys[0],
    api_keys=keys  # Automatically rotates
)
```

### 4. Monitor Usage

```python
stats = engine.get_stats()
print(f"Router stats: {stats['router_stats']}")
print(f"Provider usage: {stats['provider_usage']}")
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Rate limit errors | Too many requests | Add key pool or reduce frequency |
| Timeout errors | Slow model/provider | Increase timeout or switch provider |
| Auth errors | Invalid API key | Check key format and permissions |
| No provider available | All providers exhausted | Add more providers or wait |

### Diagnostics

```bash
# Test individual provider
gaap providers test groq

# Full system diagnostics
gaap doctor
```

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Provider API details
- [Architecture Guide](ARCHITECTURE.md) - How providers integrate
- [Examples](examples/) - Provider usage examples