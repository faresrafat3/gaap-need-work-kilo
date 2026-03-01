# ADR-003: Provider Abstraction

## Status
Accepted

## Context

GAAP needed to support multiple LLM providers:
- Kimi (primary)
- DeepSeek
- GLM
- Future providers (OpenAI, Anthropic, etc.)

Requirements:
- Easy to add new providers
- Automatic failover between providers
- Unified interface regardless of backend
- Cost tracking per provider
- Health monitoring

## Decision

We implemented a provider abstraction layer with:
1. **Base Provider Interface** - Common contract
2. **Smart Router** - Automatic routing and failover
3. **Provider Factory** - Easy instantiation
4. **Health Tracking** - Built-in monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Smart Router                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  Kimi   │ │DeepSeek │ │   GLM   │ │ Custom  │  ...      │
│  │Provider │ │Provider │ │Provider │ │Provider │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
│       │           │           │           │                │
│       └───────────┴─────┬─────┴───────────┘                │
│                         │                                    │
│                   ┌─────▼─────┐                             │
│                   │  Unified  │                             │
│                   │ Interface │                             │
│                   └───────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Base Provider Interface

```python
class BaseProvider(ABC):
    """Abstract base for all LLM providers."""
    
    name: str
    provider_type: ProviderType
    
    @abstractmethod
    async def complete(
        self, 
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate completion from messages."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> list[str]:
        """List available models."""
        pass
    
    def get_stats(self) -> ProviderStats:
        """Get usage statistics."""
        return ProviderStats(
            total_requests=self._requests,
            successful_requests=self._successes,
            failed_requests=self._failures,
            average_latency_ms=self._avg_latency,
        )
```

## Provider Types

```python
class ProviderType(Enum):
    """Types of LLM providers."""
    WEBCHAT = "webchat"      # Browser-based providers
    API = "api"              # Direct API providers
    LOCAL = "local"          # Local models
    BRIDGE = "bridge"        # Adapter providers
```

## Smart Router

The router handles provider selection:

```python
class SmartRouter:
    """Routes requests to best available provider."""
    
    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._health: dict[str, ProviderHealth] = {}
    
    def register_provider(self, provider: BaseProvider) -> None:
        """Add a provider to the pool."""
        self._providers[provider.name] = provider
    
    async def route(
        self, 
        request: Request,
        preferred: str | None = None,
    ) -> CompletionResult:
        """Route request to best provider."""
        
        # Try preferred provider first
        if preferred and preferred in self._providers:
            result = await self._try_provider(preferred, request)
            if result.success:
                return result
        
        # Fall back to healthiest provider
        for name in self._get_healthy_providers():
            result = await self._try_provider(name, request)
            if result.success:
                return result
        
        raise AllProvidersFailed()
    
    def _get_healthy_providers(self) -> list[str]:
        """Get providers sorted by health score."""
        return sorted(
            self._providers.keys(),
            key=lambda p: self._health[p].score,
            reverse=True,
        )
```

## Provider Factory

Easy provider creation:

```python
class ProviderFactory:
    """Factory for creating provider instances."""
    
    @staticmethod
    def create(
        name: str,
        provider_type: ProviderType,
        **kwargs,
    ) -> BaseProvider:
        """Create provider by type."""
        
        creators = {
            ProviderType.WEBCHAT: WebChatBridgeProvider,
            ProviderType.API: APIProvider,
            ProviderType.LOCAL: LocalProvider,
        }
        
        creator = creators.get(provider_type)
        if not creator:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        return creator(name=name, **kwargs)
    
    @staticmethod
    def create_kimi(**kwargs) -> WebChatBridgeProvider:
        """Create Kimi provider."""
        return WebChatBridgeProvider(
            name="kimi",
            model="kimi-k2.5-thinking",
            **kwargs,
        )
    
    @staticmethod
    def create_deepseek(**kwargs) -> WebChatBridgeProvider:
        """Create DeepSeek provider."""
        return WebChatBridgeProvider(
            name="deepseek",
            model="deepseek-chat",
            **kwargs,
        )
```

## Health Monitoring

```python
@dataclass
class ProviderHealth:
    """Health status for a provider."""
    
    status: HealthStatus  # healthy, degraded, unhealthy
    score: float  # 0.0 to 1.0
    last_success: datetime | None
    last_failure: datetime | None
    consecutive_failures: int
    
    def update(self, success: bool, latency_ms: float) -> None:
        """Update health based on request result."""
        if success:
            self.last_success = datetime.now()
            self.consecutive_failures = 0
            self.status = HealthStatus.HEALTHY
        else:
            self.last_failure = datetime.now()
            self.consecutive_failures += 1
            if self.consecutive_failures > 5:
                self.status = HealthStatus.UNHEALTHY
```

## Caching

Providers are cached with TTL:

```python
@dataclass
class CachedProvider:
    """Provider with cache metadata."""
    
    provider: BaseProvider
    created_at: float
    last_accessed: float
    access_count: int
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.last_accessed > CACHE_TTL
```

## Usage Example

```python
# Initialize router
router = SmartRouter()

# Register providers
router.register_provider(ProviderFactory.create_kimi())
router.register_provider(ProviderFactory.create_deepseek())
router.register_provider(ProviderFactory.create_glm())

# Use router
result = await router.route(
    request=chat_request,
    preferred="kimi",
)

# Automatic failover if Kimi fails
```

## Provider Configuration

```yaml
# config/providers.yml
providers:
  - name: kimi
    type: webchat
    enabled: true
    priority: 1
    models:
      - kimi-k2.5-thinking
    config:
      timeout: 120
      
  - name: deepseek
    type: webchat
    enabled: true
    priority: 2
    models:
      - deepseek-chat
    config:
      timeout: 120
      
  - name: glm
    type: webchat
    enabled: false
    priority: 3
    models:
      - GLM-5
```

## Alternatives Considered

### Single Provider
- **Pros:** Simple, no abstraction overhead
- **Cons:** No failover, vendor lock-in
- **Verdict:** Unacceptable for production

### LangChain Adapters
- **Pros:** Rich ecosystem
- **Cons:** Heavy dependency, less control
- **Verdict:** Good for prototyping, too heavy for core

### Direct API Calls
- **Pros:** Zero overhead
- **Cons:** No abstraction, hard to test
- **Verdict:** Too coupled, hard to maintain

## Consequences

### Positive
- Easy to add new providers
- Automatic failover improves reliability
- Unified cost tracking
- Simple testing with mocks
- Clear separation of concerns

### Negative
- Additional abstraction layer
- Need to maintain provider-specific code
- Caching complexity

## Future Considerations

1. **Provider-Specific Optimization**: Route by task type
2. **Cost-Aware Routing**: Prefer cheaper providers
3. **Quality Scoring**: Route by provider quality for task type
4. **Geographic Routing**: Route to nearest provider

## References

- `gaap/providers/base_provider.py`
- `gaap/routing/router.py`
- `gaap/api/providers.py`
