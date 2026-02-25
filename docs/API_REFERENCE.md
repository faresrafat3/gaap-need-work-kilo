# GAAP API Reference

This document provides complete API documentation for GAAP.

## Table of Contents

1. [Engine API](#engine-api)
2. [Types](#types)
3. [Providers](#providers)
4. [Routing](#routing)
5. [Healing](#healing)
6. [Memory](#memory)
7. [Security](#security)
8. [Configuration](#configuration)
9. [Research API](#research-api)
10. [REST API Reference](#rest-api-reference)
11. [WebSocket API](#websocket-api)
12. [Pydantic Models Reference](#pydantic-models-reference)
13. [CLI Commands](#cli-commands)

---

## Engine API

### GAAPEngine

The main entry point for GAAP.

```python
from gaap import GAAPEngine, GAAPRequest

engine = GAAPEngine(
    providers: list | None = None,
    budget: float = 100.0,
    enable_context: bool = True,
    enable_healing: bool = True,
    enable_memory: bool = True,
    enable_security: bool = True,
    project_path: str | None = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `providers` | `list[BaseProvider]` | `[UnifiedGAAPProvider()]` | List of LLM providers |
| `budget` | `float` | `100.0` | Maximum budget in USD |
| `enable_context` | `bool` | `True` | Enable context orchestration |
| `enable_healing` | `bool` | `True` | Enable self-healing system |
| `enable_memory` | `bool` | `True` | Enable hierarchical memory |
| `enable_security` | `bool` | `True` | Enable security firewall |
| `project_path` | `str | None` | `None` | Path for context loading |

#### Methods

##### process()

Process a complete request through all layers.

```python
async def process(request: GAAPRequest) -> GAAPResponse
```

##### chat()

Simple chat interface.

```python
async def chat(message: str, context: dict | None = None) -> str
```

##### get_stats()

Get system statistics.

```python
def get_stats() -> dict[str, Any]
```

Returns:
```python
{
    "requests_processed": int,
    "successful": int,
    "failed": int,
    "success_rate": float,
    "layer0_stats": dict,
    "layer1_stats": dict,
    "layer2_stats": dict,
    "layer3_stats": dict,
    "router_stats": dict,
}
```

##### shutdown()

Shutdown engine and close providers.

```python
def shutdown() -> None
```

---

### GAAPRequest

Input request structure.

```python
from gaap import GAAPRequest

request = GAAPRequest(
    text: str,                          # User input text
    context: dict[str, Any] | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    budget_limit: float | None = None,
    metadata: dict[str, Any] = {}
)
```

---

### GAAPResponse

Output response structure.

```python
@dataclass
class GAAPResponse:
    request_id: str
    success: bool
    output: Any | None
    error: str | None
    
    # Journey through layers
    intent: StructuredIntent | None
    architecture_spec: ArchitectureSpec | None
    task_graph: TaskGraph | None
    execution_results: list[ExecutionResult]
    
    # Metrics
    total_time_ms: float
    total_cost_usd: float
    total_tokens: int
    quality_score: float
    
    metadata: dict[str, Any]
```

---

### Convenience Functions

#### create_engine()

Create an engine with common providers.

```python
from gaap import create_engine

engine = create_engine(
    groq_api_key: str | None = None,
    gemini_api_key: str | None = None,
    gemini_api_keys: list[str] | None = None,
    budget: float = 100.0,
    project_path: str | None = None,
    enable_all: bool = True
) -> GAAPEngine
```

#### quick_chat()

One-shot chat function.

```python
from gaap import quick_chat

response = await quick_chat(
    message: str,
    groq_api_key: str | None = None,
    budget: float = 10.0
) -> str
```

---

## Types

### Enums

#### TaskPriority

```python
class TaskPriority(Enum):
    CRITICAL = auto()    # Maximum resources
    HIGH = auto()        # High priority
    NORMAL = auto()      # Default
    LOW = auto()         # Low priority
    BACKGROUND = auto()  # Non-blocking
```

#### TaskComplexity

```python
class TaskComplexity(Enum):
    TRIVIAL = auto()        # Single line
    SIMPLE = auto()         # Single function
    MODERATE = auto()       # Single component
    COMPLEX = auto()        # Multiple components
    ARCHITECTURAL = auto()  # Full system
```

#### TaskType

```python
class TaskType(Enum):
    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    RESEARCH = auto()
    ANALYSIS = auto()
    PLANNING = auto()
    ORCHESTRATION = auto()
```

#### LayerType

```python
class LayerType(Enum):
    INTERFACE = 0       # Layer 0
    STRATEGIC = 1       # Layer 1
    TACTICAL = 2        # Layer 2
    EXECUTION = 3       # Layer 3
    META_LEARNING = 4   # Layer 4
    EXTERNAL = 5        # Layer 5
```

#### ModelTier

```python
class ModelTier(Enum):
    TIER_1_STRATEGIC = auto()  # Smartest models
    TIER_2_TACTICAL = auto()   # Balanced
    TIER_3_EFFICIENT = auto()  # Fast/cheap
    TIER_4_PRIVATE = auto()    # Local
```

#### HealingLevel

```python
class HealingLevel(Enum):
    L1_RETRY = auto()
    L2_REFINE = auto()
    L3_PIVOT = auto()
    L4_STRATEGY_SHIFT = auto()
    L5_HUMAN_ESCALATION = auto()
```

#### ExecutionStatus

```python
class ExecutionStatus(Enum):
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    ESCALATED = auto()
    CANCELLED = auto()
```

### Data Classes

#### Message

```python
@dataclass
class Message:
    role: MessageRole
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict() -> dict[str, Any]
```

#### Task

```python
@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    type: TaskType = TaskType.CODE_GENERATION
    priority: TaskPriority = TaskPriority.NORMAL
    complexity: TaskComplexity = TaskComplexity.MODERATE
    context: TaskContext = field(default_factory=TaskContext)
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 2000
    max_retries: int = 3
    retry_count: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: TaskResult | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### TaskResult

```python
@dataclass
class TaskResult:
    success: bool
    output: Any
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
```

#### RoutingDecision

```python
@dataclass
class RoutingDecision:
    selected_provider: str
    selected_model: str
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## Providers

### BaseProvider

Abstract base class for all providers.

```python
class BaseProvider(ABC):
    def __init__(
        self,
        name: str,
        provider_type: ProviderType,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
        rate_limit: int = 60,
        timeout: int = 120,
        max_retries: int = 3
    )
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    def get_available_models() -> list[str]:
        """Return list of available models."""
        pass
    
    @abstractmethod
    def get_model_info(model: str) -> ModelInfo:
        """Return model information."""
        pass
```

### GroqProvider

```python
from gaap.providers import GroqProvider

provider = GroqProvider(
    api_key: str,
    models: list[str] | None = None,
    rate_limit: int = 30
)

# Available models
models = provider.get_available_models()
# ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'qwen3-32b']
```

### GeminiProvider

```python
from gaap.providers import GeminiProvider

provider = GeminiProvider(
    api_key: str,
    api_keys: list[str] | None = None,  # Key pool rotation
    models: list[str] | None = None,
    rate_limit: int = 5
)

# Key pool rotation
provider = GeminiProvider(
    api_key="primary_key",
    api_keys=["key1", "key2", "key3"]
)
```

### UnifiedGAAPProvider

Multi-provider with automatic failover.

```python
from gaap.providers import UnifiedGAAPProvider

provider = UnifiedGAAPProvider(
    default_provider: str = "kimi",  # Primary provider
    fallback_order: list[str] = ["kimi", "deepseek", "glm"]
)

# Automatically tries next provider on failure
response = await provider.chat_completion(messages, model)
```

---

## Routing

### SmartRouter

```python
from gaap.routing import SmartRouter, RoutingStrategy

router = SmartRouter(
    providers: list[BaseProvider],
    strategy: RoutingStrategy = RoutingStrategy.SMART,
    budget_limit: float = 100.0
)

# Strategies
class RoutingStrategy(Enum):
    QUALITY_FIRST = "quality_first"
    COST_OPTIMIZED = "cost_optimized"
    SPEED_FIRST = "speed_first"
    BALANCED = "balanced"
    SMART = "smart"

# Route request
decision = await router.route(
    messages: list[Message],
    task: Task
) -> RoutingDecision
```

### FallbackManager

```python
from gaap.routing import FallbackManager

fallback = FallbackManager(router: SmartRouter)

# Execute with automatic fallback
response = await fallback.execute_with_fallback(
    messages: list[Message],
    primary_provider: str,
    primary_model: str,
    fallback_providers: list[str] | None = None
) -> ChatCompletionResponse

# Reset provider health
fallback.reset_health()
```

---

## Healing

### SelfHealingSystem

```python
from gaap.healing import SelfHealingSystem

healing = SelfHealingSystem(
    max_level: HealingLevel = HealingLevel.L4_STRATEGY_SHIFT,
    on_escalate: Callable | None = None
)

# Heal from error
result = await healing.heal(
    error: Exception,
    task: Task,
    execute_func: Callable[[Task], Awaitable[Any]],
    context: dict | None = None
) -> RecoveryResult

# Get statistics
stats = healing.get_stats()
# {
#     "total_attempts": int,
#     "successful_recoveries": int,
#     "escalations": int,
#     "recovery_rate": float,
#     "errors_by_category": dict,
#     "healing_by_level": dict
# }
```

### ErrorClassifier

```python
from gaap.healing import ErrorClassifier

category = ErrorClassifier.classify(error: Exception) -> ErrorCategory

# Categories
class ErrorCategory(Enum):
    TRANSIENT = auto()     # Network, rate limits
    SYNTAX = auto()        # Parse errors
    LOGIC = auto()         # Validation failures
    MODEL_LIMIT = auto()   # Context, timeout
    RESOURCE = auto()      # Budget, quota
    CRITICAL = auto()      # Security violations
    UNKNOWN = auto()
```

---

## Memory

### HierarchicalMemory

```python
from gaap.memory import HierarchicalMemory, MemoryTier

memory = HierarchicalMemory(
    working_size: int = 100,
    storage_path: str | None = None
)

# Store in specific tier
memory.store(key: str, content: Any, tier: MemoryTier = MemoryTier.WORKING)

# Retrieve from any tier
content = memory.retrieve(key: str) -> Any | None

# Record episode
memory.record_episode(episode: EpisodicMemory)

# Get relevant context
context = memory.get_relevant_context(query: str) -> dict

# Get statistics
stats = memory.get_stats()

# Save/load to disk
memory.save() -> dict[str, bool]
memory.load() -> dict[str, bool]
```

### EpisodicMemory

```python
from gaap.memory import EpisodicMemory

episode = EpisodicMemory(
    task_id: str,
    action: str,
    result: str,
    success: bool,
    duration_ms: float,
    tokens_used: int,
    cost_usd: float,
    model: str,
    provider: str,
    lessons: list[str] = []
)

memory.record_episode(episode)
```

---

## Research API

### DeepDiscoveryEngine

Main orchestrator for deep research and knowledge building.

```python
from gaap.research import DeepDiscoveryEngine, DDEConfig

engine = DeepDiscoveryEngine(
    config=DDEConfig(research_depth=3),
    llm_provider=provider,
)

result = await engine.research("FastAPI async best practices")
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `DDEConfig` | Configuration options |
| `llm_provider` | `BaseProvider` | LLM provider for synthesis |
| `knowledge_graph` | `KnowledgeGraphBuilder` | Knowledge graph storage |
| `sqlite_store` | `SQLiteStore` | Persistent storage |

#### Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `research(query: str)` | `ResearchResult` | Main research method |
| `quick_search(query: str, max_results: int)` | `list[Source]` | Quick search without deep analysis |
| `get_stats()` | `dict` | Get engine statistics |
| `update_config(new_config: DDEConfig)` | `None` | Update configuration |
| `close()` | `None` | Close all resources |

---

### DDEConfig

Master configuration for Deep Discovery Engine.

```python
from gaap.research import DDEConfig

# Default configuration
config = DDEConfig()

# Presets
config = DDEConfig.quick()      # Quick research (depth=1)
config = DDEConfig.standard()   # Standard research (depth=3)
config = DDEConfig.deep()       # Deep research (depth=5)
config = DDEConfig.academic()   # Academic research (high ETS)

# Custom configuration
config = DDEConfig(
    research_depth=4,
    max_total_sources=100,
    web_fetcher=WebFetcherConfig(provider="serper", api_key="..."),
)
```

#### Global Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_depth` | `int` | `3` | Exploration depth (1-5) |
| `max_total_sources` | `int` | `50` | Maximum sources to process |
| `max_total_hypotheses` | `int` | `20` | Maximum hypotheses to build |
| `max_execution_time_seconds` | `int` | `300` | Timeout for research |
| `parallel_processing` | `bool` | `True` | Enable parallel processing |
| `max_parallel_tasks` | `int` | `5` | Maximum parallel tasks |
| `check_existing_research` | `bool` | `True` | Check cache before research |

#### WebFetcherConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"duckduckgo"` | Search provider |
| `api_key` | `str \| None` | `None` | API key (if required) |
| `max_results` | `int` | `10` | Maximum search results |
| `timeout_seconds` | `int` | `30` | Request timeout |
| `rate_limit_per_second` | `float` | `2.0` | Rate limiting |

#### SourceAuditConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_ets_threshold` | `float` | `0.3` | Minimum ETS score |
| `domain_overrides` | `dict` | `{}` | Custom domain scores |
| `blacklist_domains` | `list` | `[]` | Blacklisted domains |
| `check_author` | `bool` | `True` | Check author credibility |
| `check_date` | `bool` | `True` | Check content freshness |

#### SynthesizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_hypotheses` | `int` | `10` | Maximum hypotheses per research |
| `min_confidence_threshold` | `float` | `0.5` | Minimum hypothesis confidence |
| `cross_validate_enabled` | `bool` | `True` | Enable cross-validation |
| `detect_contradictions` | `bool` | `True` | Detect contradictions |

#### DeepDiveConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_depth` | `int` | `3` | Default exploration depth |
| `max_depth` | `int` | `5` | Maximum depth allowed |
| `citation_follow_depth` | `int` | `2` | Citation following depth |
| `max_sources_per_depth` | `int` | `20` | Sources per depth level |

---

### Types

#### Source

```python
from gaap.research import Source, SourceStatus

source = Source(
    url="https://fastapi.tiangolo.com/async/",
    title="Async in FastAPI",
    domain="fastapi.tiangolo.com",
    ets_score=1.0,  # Official docs
    content="...",
    author="Sebastián Ramírez",
    publish_date=date(2024, 1, 15),
    status=SourceStatus.AUDITED,
)
```

#### Hypothesis

```python
from gaap.research import Hypothesis, HypothesisStatus

hypothesis = Hypothesis(
    id="abc123",
    statement="FastAPI supports async/await for concurrent operations",
    status=HypothesisStatus.VERIFIED,
    confidence=0.95,
    supporting_sources=[source1, source2],
    reasoning="Verified across official docs and examples",
)
```

#### ETSLevel

```python
from gaap.research import ETSLevel

# Epistemic Trust Score levels
ETSLevel.VERIFIED      # 1.0 - Official docs, verified repos
ETSLevel.RELIABLE      # 0.7 - Peer-reviewed papers, high-reputation SO
ETSLevel.QUESTIONABLE  # 0.5 - Medium articles, tutorials
ETSLevel.UNRELIABLE    # 0.3 - Random blogs, AI summaries
ETSLevel.BLACKLISTED   # 0.0 - Contradictory or banned domains
```

#### ResearchResult

```python
@dataclass
class ResearchResult:
    success: bool
    query: str
    finding: ResearchFinding | None
    metrics: ResearchMetrics
    execution_trace: list[ExecutionStep]
    total_time_ms: float
    error: str | None
```

#### ResearchMetrics

```python
@dataclass
class ResearchMetrics:
    sources_found: int
    sources_fetched: int
    sources_passed_ets: int
    hypotheses_generated: int
    hypotheses_verified: int
    triples_extracted: int
    avg_ets_score: float
    llm_calls: int
    web_requests: int
```

---

### WebFetcher

Web search and content fetching with multiple providers.

```python
from gaap.research import WebFetcher, WebFetcherConfig

fetcher = WebFetcher(WebFetcherConfig(
    provider="duckduckgo",  # Free, no API key
    max_results=10,
))

# Search the web
results = await fetcher.search("Python async best practices")

# Fetch content from URL
content = await fetcher.fetch_content("https://example.com/article")

# Batch fetch
contents = await fetcher.fetch_batch(["url1", "url2", "url3"])
```

#### Supported Providers

| Provider | API Key | Description |
|----------|---------|-------------|
| `duckduckgo` | No | Free HTML scraping (default) |
| `serper` | Yes | Google search via Serper API |
| `perplexity` | Yes | Perplexity AI search |
| `brave` | Yes | Brave Search API |

---

### SourceAuditor

Epistemic Trust Score (ETS) evaluation for sources.

```python
from gaap.research import SourceAuditor, SourceAuditConfig

auditor = SourceAuditor(SourceAuditConfig(
    min_ets_threshold=0.3,
))

# Audit a single source
result = auditor.audit(source)
print(f"ETS Score: {result.ets_score}")
print(f"ETS Level: {result.ets_level}")

# Audit batch with filtering
passed, filtered = auditor.audit_batch(sources)

# Customize domain scores
auditor.set_domain_score("mydomain.com", 0.9)
auditor.add_blacklist_domain("spam.com")
```

#### Domain Scores (Default)

| Category | Score | Examples |
|----------|-------|----------|
| Official Docs | 1.0 | docs.python.org, fastapi.tiangolo.com |
| Official Repos | 0.85-0.9 | github.com, pypi.org |
| Community Q&A | 0.7-0.75 | stackoverflow.com, wikipedia.org |
| Tech Blogs | 0.5-0.55 | medium.com, dev.to |
| Random Blogs | 0.3-0.4 | blogspot.com, wordpress.com |
| Blacklisted | 0.0 | example.com, test.com |

---

### ContentExtractor

Clean text extraction from web pages.

```python
from gaap.research import ContentExtractor, ContentExtractorConfig

extractor = ContentExtractor(ContentExtractorConfig(
    max_content_length=50000,
    extract_code_blocks=True,
    extract_links=True,
))

# Extract from URL
content = await extractor.extract("https://example.com/article")
print(content.title)
print(content.content)
print(content.author)
print(content.code_blocks)
print(content.links)
```

---

### Synthesizer

LLM-powered hypothesis synthesis and verification.

```python
from gaap.research import Synthesizer, SynthesizerConfig

synthesizer = Synthesizer(
    llm_provider=provider,
    config=SynthesizerConfig(
        max_hypotheses=10,
        cross_validate_enabled=True,
    ),
)

# Extract claims from content
claims = await synthesizer.extract_claims(content, source)

# Build hypothesis from claim
hypothesis = await synthesizer.build_hypothesis(claims[0], sources)

# Verify hypothesis against sources
verified = await synthesizer.verify_hypothesis(hypothesis, all_sources)

# Extract knowledge triples
triples = await synthesizer.extract_triples(content, source)

# Find contradictions between hypotheses
contradictions = await synthesizer.find_contradictions(hypotheses)
```

---

### DeepDive

Deep exploration protocol with citation mapping.

```python
from gaap.research import DeepDive, DeepDiveConfig

deep_dive = DeepDive(DeepDiveConfig(
    default_depth=3,
    max_depth=5,
    citation_follow_depth=2,
))

# Execute deep dive
result = await deep_dive.explore(
    query="FastAPI async best practices",
    depth=4,
)

print(f"Sources found: {len(result.sources)}")
print(f"Primary sources: {len(result.primary_sources)}")
print(f"Citations followed: {result.citations_followed}")
```

#### Depth Levels

| Depth | Actions |
|-------|---------|
| 1 | Basic search + content extraction |
| 2 | + Citation following |
| 3 | + Cross-validation + hypothesis building |
| 4 | + Related topic exploration |
| 5 | Full deep dive with all features |

---

### KnowledgeIntegrator

Permanent storage for research results (no TTL).

```python
from gaap.research import KnowledgeIntegrator, StorageConfig

integrator = KnowledgeIntegrator(
    knowledge_graph=kg_builder,
    sqlite_store=sqlite,
    config=StorageConfig(
        knowledge_graph_enabled=True,
        sqlite_cache_enabled=True,
    ),
)

# Store research finding
finding_id = await integrator.store_research(finding)

# Find similar existing research
existing = await integrator.find_similar("FastAPI async")

# Get research by topic
findings = await integrator.get_by_topic("async")

# Add triple to knowledge graph
triple_id = await integrator.add_triple(
    subject="FastAPI",
    predicate="supports",
    object="async/await",
    source=source,
)
```

---

## REST API Reference

GAAP provides a comprehensive REST API with 47 endpoints across 7 modules for system management, session handling, budget tracking, memory operations, healing configuration, and provider management.

### System API (`/api/system`)

System-level operations for health monitoring, metrics, and administration.

#### GET /api/system/health

Check system health status.

```bash
curl http://localhost:8000/api/system/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "components": {
    "database": "healthy",
    "providers": "healthy",
    "memory": "healthy"
  }
}
```

#### GET /api/system/metrics

Get system performance metrics.

```bash
curl http://localhost:8000/api/system/metrics
```

**Response:**
```json
{
  "requests_total": 1523,
  "requests_per_minute": 12.5,
  "avg_latency_ms": 245,
  "error_rate": 0.02,
  "memory_usage_mb": 512,
  "cpu_usage_percent": 45
}
```

#### GET /api/system/logs

Retrieve system logs with filtering.

```bash
curl "http://localhost:8000/api/system/logs?level=ERROR&limit=100"
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `string` | `INFO` | Log level filter |
| `limit` | `int` | `100` | Maximum entries |
| `since` | `datetime` | `null` | Start timestamp |

#### POST /api/system/restart

Restart the system (requires admin privileges).

```bash
curl -X POST http://localhost:8000/api/system/restart
```

#### GET /api/system/info

Get system information and version.

```bash
curl http://localhost:8000/api/system/info
```

**Response:**
```json
{
  "name": "GAAP",
  "version": "1.0.0",
  "python_version": "3.11.0",
  "environment": "production",
  "features": ["healing", "memory", "security"]
}
```

#### GET /api/system/events

Stream system events (Server-Sent Events).

```bash
curl http://localhost:8000/api/system/events
```

#### POST /api/system/clear-cache

Clear system caches.

```bash
curl -X POST http://localhost:8000/api/system/clear-cache
```

---

### Sessions API (`/api/sessions`)

Session lifecycle management for request tracking.

#### GET /api/sessions

List all sessions with pagination.

```bash
curl "http://localhost:8000/api/sessions?status=active&limit=50"
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `string` | `null` | Filter by status |
| `limit` | `int` | `50` | Max results |
| `offset` | `int` | `0` | Pagination offset |

#### POST /api/sessions

Create a new session.

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "My Session", "budget_limit": 10.0}'
```

**Request Body:**
```json
{
  "name": "My Session",
  "budget_limit": 10.0,
  "metadata": {}
}
```

#### GET /api/sessions/{session_id}

Get session details.

```bash
curl http://localhost:8000/api/sessions/abc123
```

#### PUT /api/sessions/{session_id}

Update session properties.

```bash
curl -X PUT http://localhost:8000/api/sessions/abc123 \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Name"}'
```

#### DELETE /api/sessions/{session_id}

Delete a session.

```bash
curl -X DELETE http://localhost:8000/api/sessions/abc123
```

#### POST /api/sessions/{session_id}/pause

Pause an active session.

```bash
curl -X POST http://localhost:8000/api/sessions/abc123/pause
```

#### POST /api/sessions/{session_id}/resume

Resume a paused session.

```bash
curl -X POST http://localhost:8000/api/sessions/abc123/resume
```

#### POST /api/sessions/{session_id}/export

Export session data.

```bash
curl -X POST http://localhost:8000/api/sessions/abc123/export \
  -H "Content-Type: application/json" \
  -d '{"format": "json"}'
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `string` | `json` | Export format (`json`, `csv`, `markdown`) |

#### POST /api/sessions/{session_id}/cancel

Cancel an active session.

```bash
curl -X POST http://localhost:8000/api/sessions/abc123/cancel
```

---

### Budget API (`/api/budget`)

Budget tracking and alerting.

#### GET /api/budget

Get current budget status.

```bash
curl http://localhost:8000/api/budget
```

**Response:**
```json
{
  "total_budget": 100.0,
  "used": 23.45,
  "remaining": 76.55,
  "percent_used": 23.45
}
```

#### GET /api/budget/usage

Get detailed usage breakdown.

```bash
curl "http://localhost:8000/api/budget/usage?period=daily"
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | `string` | `daily` | Period (`hourly`, `daily`, `weekly`, `monthly`) |

#### GET /api/budget/alerts

List budget alerts.

```bash
curl http://localhost:8000/api/budget/alerts
```

#### PUT /api/budget/limits

Update budget limits.

```bash
curl -X PUT http://localhost:8000/api/budget/limits \
  -H "Content-Type: application/json" \
  -d '{"daily_limit": 10.0, "monthly_limit": 200.0}'
```

#### POST /api/budget/alerts/{alert_id}/acknowledge

Acknowledge a budget alert.

```bash
curl -X POST http://localhost:8000/api/budget/alerts/alert123/acknowledge
```

---

### Memory API (`/api/memory`)

Hierarchical memory operations.

#### GET /api/memory/stats

Get memory statistics.

```bash
curl http://localhost:8000/api/memory/stats
```

**Response:**
```json
{
  "working_memory_size": 45,
  "episodic_count": 234,
  "semantic_triples": 1500,
  "total_size_mb": 12.5
}
```

#### GET /api/memory/tiers

List memory tiers and their status.

```bash
curl http://localhost:8000/api/memory/tiers
```

#### POST /api/memory/consolidate

Trigger memory consolidation.

```bash
curl -X POST http://localhost:8000/api/memory/consolidate
```

#### POST /api/memory/clear/{tier}

Clear a specific memory tier.

```bash
curl -X POST http://localhost:8000/api/memory/clear/working
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `tier` | `string` | Tier name (`working`, `episodic`, `semantic`) |

#### GET /api/memory/search

Search memory contents.

```bash
curl "http://localhost:8000/api/memory/search?q=fastapi&limit=20"
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | `string` | required | Search query |
| `tier` | `string` | `null` | Filter by tier |
| `limit` | `int` | `20` | Max results |

#### POST /api/memory/save

Save memory state to disk.

```bash
curl -X POST http://localhost:8000/api/memory/save
```

#### POST /api/memory/load

Load memory state from disk.

```bash
curl -X POST http://localhost:8000/api/memory/load
```

---

### Healing API (`/api/healing`)

Self-healing system configuration and monitoring.

#### GET /api/healing/config

Get healing configuration.

```bash
curl http://localhost:8000/api/healing/config
```

#### PUT /api/healing/config

Update healing configuration.

```bash
curl -X PUT http://localhost:8000/api/healing/config \
  -H "Content-Type: application/json" \
  -d '{"max_level": "L3_PIVOT", "auto_escalate": true}'
```

#### GET /api/healing/history

Get healing attempt history.

```bash
curl "http://localhost:8000/api/healing/history?limit=50"
```

#### GET /api/healing/patterns

Get learned healing patterns.

```bash
curl http://localhost:8000/api/healing/patterns
```

#### POST /api/healing/reset

Reset healing statistics.

```bash
curl -X POST http://localhost:8000/api/healing/reset
```

#### GET /api/healing/stats

Get healing statistics.

```bash
curl http://localhost:8000/api/healing/stats
```

**Response:**
```json
{
  "total_attempts": 45,
  "successful_recoveries": 42,
  "escalations": 3,
  "recovery_rate": 0.933,
  "by_level": {
    "L1_RETRY": 20,
    "L2_REFINE": 15,
    "L3_PIVOT": 7,
    "L4_STRATEGY_SHIFT": 3
  }
}
```

---

### Config API (`/api/config`)

Configuration management.

#### GET /api/config

Get complete configuration.

```bash
curl http://localhost:8000/api/config
```

#### PUT /api/config

Update configuration.

```bash
curl -X PUT http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"system": {"log_level": "DEBUG"}}'
```

#### GET /api/config/{module}

Get module-specific configuration.

```bash
curl http://localhost:8000/api/config/healing
```

#### PUT /api/config/{module}

Update module-specific configuration.

```bash
curl -X PUT http://localhost:8000/api/config/healing \
  -H "Content-Type: application/json" \
  -d '{"max_level": "L4_STRATEGY_SHIFT"}'
```

#### POST /api/config/validate

Validate configuration changes.

```bash
curl -X POST http://localhost:8000/api/config/validate \
  -H "Content-Type: application/json" \
  -d '{"system": {"log_level": "INVALID"}}'
```

**Response:**
```json
{
  "valid": false,
  "errors": ["Invalid log level: INVALID"]
}
```

#### POST /api/config/reload

Reload configuration from file.

```bash
curl -X POST http://localhost:8000/api/config/reload
```

#### GET /api/config/presets/list

List available configuration presets.

```bash
curl http://localhost:8000/api/config/presets/list
```

#### GET /api/config/schema/all

Get complete configuration schema.

```bash
curl http://localhost:8000/api/config/schema/all
```

---

### Providers API (`/api/providers`)

Provider management and testing.

#### GET /api/providers

List all configured providers.

```bash
curl http://localhost:8000/api/providers
```

**Response:**
```json
{
  "providers": [
    {
      "name": "groq",
      "type": "groq",
      "status": "healthy",
      "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    }
  ]
}
```

#### POST /api/providers

Add a new provider.

```bash
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{"name": "openai", "type": "openai", "api_key": "sk-..."}'
```

#### GET /api/providers/{name}

Get provider details.

```bash
curl http://localhost:8000/api/providers/groq
```

#### PUT /api/providers/{name}

Update provider configuration.

```bash
curl -X PUT http://localhost:8000/api/providers/groq \
  -H "Content-Type: application/json" \
  -d '{"rate_limit": 60}'
```

#### DELETE /api/providers/{name}

Remove a provider.

```bash
curl -X DELETE http://localhost:8000/api/providers/groq
```

#### POST /api/providers/{name}/test

Test provider connectivity.

```bash
curl -X POST http://localhost:8000/api/providers/groq/test
```

**Response:**
```json
{
  "success": true,
  "latency_ms": 245,
  "model_tested": "llama-3.3-70b-versatile"
}
```

#### POST /api/providers/{name}/enable

Enable a disabled provider.

```bash
curl -X POST http://localhost:8000/api/providers/groq/enable
```

#### POST /api/providers/{name}/disable

Temporarily disable a provider.

```bash
curl -X POST http://localhost:8000/api/providers/groq/disable
```

---

## WebSocket API

GAAP provides real-time communication through WebSocket endpoints for live events, OODA loop visualization, and steering commands.

### /ws/events

Real-time events stream for all system events.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/events');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};
```

**Event Types:**
| Type | Description |
|------|-------------|
| `request.started` | New request started |
| `request.completed` | Request finished |
| `request.failed` | Request failed |
| `healing.triggered` | Healing initiated |
| `healing.recovered` | Recovery successful |
| `budget.warning` | Budget threshold reached |
| `provider.status` | Provider status change |

**Sample Event:**
```json
{
  "type": "request.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "request_id": "abc123",
    "duration_ms": 1250,
    "cost_usd": 0.0023
  }
}
```

---

### /ws/ooda

OODA (Observe-Orient-Decide-Act) loop visualization for real-time agent monitoring.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/ooda');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('OODA State:', data);
};
```

**OODA States:**
| State | Description |
|-------|-------------|
| `OBSERVE` | Gathering information |
| `ORIENT` | Analyzing context |
| `DECIDE` | Making routing decisions |
| `ACT` | Executing actions |

**Sample Message:**
```json
{
  "phase": "ORIENT",
  "session_id": "sess123",
  "context": {
    "intent": "code_generation",
    "complexity": "moderate",
    "estimated_tokens": 2000
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### /ws/steering

Steering commands for manual control (pause, resume, veto).

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/steering');

// Send steering command
ws.send(JSON.stringify({
  "command": "pause",
  "session_id": "sess123",
  "reason": "User requested pause"
}));
```

**Commands:**
| Command | Description |
|---------|-------------|
| `pause` | Pause current execution |
| `resume` | Resume paused execution |
| `veto` | Cancel current action |
| `redirect` | Change execution path |
| `escalate` | Force human review |

**Sample Command:**
```json
{
  "command": "veto",
  "session_id": "sess123",
  "action_id": "act456",
  "reason": "Potential security risk detected"
}
```

**Response:**
```json
{
  "status": "acknowledged",
  "command": "veto",
  "executed_at": "2024-01-15T10:30:00Z"
}
```

---

## Pydantic Models Reference

### System Models

```python
from gaap.api.models import (
    HealthResponse,
    MetricsResponse,
    SystemInfo,
    LogEntry,
    SystemEvent,
)

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    components: dict[str, str]

class MetricsResponse(BaseModel):
    requests_total: int
    requests_per_minute: float
    avg_latency_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
```

### Session Models

```python
from gaap.api.models import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionList,
    SessionExport,
)

class SessionCreate(BaseModel):
    name: str
    budget_limit: float | None = None
    metadata: dict[str, Any] = {}

class SessionResponse(BaseModel):
    id: str
    name: str
    status: Literal["active", "paused", "completed", "cancelled"]
    created_at: datetime
    budget_limit: float | None
    budget_used: float
    request_count: int
    metadata: dict[str, Any]
```

### Budget Models

```python
from gaap.api.models import (
    BudgetStatus,
    BudgetUsage,
    BudgetAlert,
    BudgetLimits,
)

class BudgetStatus(BaseModel):
    total_budget: float
    used: float
    remaining: float
    percent_used: float
    period_start: datetime
    period_end: datetime

class BudgetAlert(BaseModel):
    id: str
    type: Literal["warning", "critical", "exceeded"]
    threshold: float
    current_value: float
    message: str
    acknowledged: bool
    created_at: datetime
```

### Memory Models

```python
from gaap.api.models import (
    MemoryStats,
    MemoryTier,
    MemorySearchRequest,
    MemorySearchResult,
)

class MemoryStats(BaseModel):
    working_memory_size: int
    episodic_count: int
    semantic_triples: int
    total_size_mb: float
    last_consolidation: datetime | None

class MemorySearchRequest(BaseModel):
    query: str
    tier: MemoryTier | None = None
    limit: int = 20
    include_metadata: bool = True
```

### Healing Models

```python
from gaap.api.models import (
    HealingConfig,
    HealingHistory,
    HealingPattern,
    HealingStats,
)

class HealingConfig(BaseModel):
    max_level: HealingLevel
    auto_escalate: bool
    escalation_timeout_seconds: int
    notify_on_escalation: bool

class HealingStats(BaseModel):
    total_attempts: int
    successful_recoveries: int
    escalations: int
    recovery_rate: float
    by_level: dict[str, int]
    errors_by_category: dict[str, int]
```

### Config Models

```python
from gaap.api.models import (
    ConfigUpdate,
    ConfigValidation,
    ConfigPreset,
    ConfigSchema,
)

class ConfigUpdate(BaseModel):
    system: SystemConfig | None = None
    healing: HealingConfig | None = None
    budget: BudgetConfig | None = None
    providers: list[ProviderConfig] | None = None

class ConfigValidation(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
```

### Provider Models

```python
from gaap.api.models import (
    ProviderCreate,
    ProviderUpdate,
    ProviderResponse,
    ProviderTestResult,
)

class ProviderCreate(BaseModel):
    name: str
    type: str
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] | None = None
    rate_limit: int = 60
    timeout: int = 120

class ProviderResponse(BaseModel):
    name: str
    type: str
    status: Literal["healthy", "degraded", "disabled"]
    models: list[str]
    rate_limit: int
    request_count: int
    last_request: datetime | None

class ProviderTestResult(BaseModel):
    success: bool
    latency_ms: float
    model_tested: str
    error: str | None = None
```

### WebSocket Models

```python
from gaap.api.models import (
    WSEventMessage,
    WSOODAMessage,
    WSSteeringCommand,
    WSSteeringResponse,
)

class WSEventMessage(BaseModel):
    type: str
    timestamp: datetime
    data: dict[str, Any]

class WSOODAMessage(BaseModel):
    phase: Literal["OBSERVE", "ORIENT", "DECIDE", "ACT"]
    session_id: str
    context: dict[str, Any]
    timestamp: datetime

class WSSteeringCommand(BaseModel):
    command: Literal["pause", "resume", "veto", "redirect", "escalate"]
    session_id: str
    action_id: str | None = None
    reason: str | None = None
    redirect_target: str | None = None
```

---

## Security

### PromptFirewall

```python
from gaap.security import PromptFirewall, RiskLevel

firewall = PromptFirewall(strictness: str = "high")

# Scan input
result = firewall.scan(
    input_text: str,
    context: dict | None = None
) -> FirewallResult

# Result structure
@dataclass
class FirewallResult:
    is_safe: bool
    risk_level: RiskLevel
    detected_patterns: list[str]
    sanitized_input: str
    recommendations: list[str]
    scan_time_ms: float
    layer_scores: dict[str, float]

# Risk levels
class RiskLevel(Enum):
    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    BLOCKED = auto()
```

### AuditTrail

```python
from gaap.security import AuditTrail

audit = AuditTrail(storage_path: str | None = None)

# Record action
entry = audit.record(
    action: str,
    agent_id: str,
    resource: str,
    result: str,
    details: dict | None = None
) -> AuditEntry

# Verify integrity
is_valid = audit.verify_integrity() -> bool

# Get history
history = audit.get_agent_history(agent_id: str) -> list[AuditEntry]
recent = audit.get_recent(limit: int = 100) -> list[AuditEntry]

# Export
audit.export(path: str)
```

### CapabilityManager

```python
from gaap.security import CapabilityManager

cap_manager = CapabilityManager(secret_key: str = "gaap-secret")

# Issue token
token = cap_manager.issue_token(
    agent_id: str,
    resource: str,
    action: str,
    ttl_seconds: int = 300,
    constraints: dict | None = None
) -> CapabilityToken

# Verify token
is_valid = cap_manager.verify_token(
    token: CapabilityToken,
    requested_resource: str,
    requested_action: str
) -> bool

# Revoke token
cap_manager.revoke_token(agent_id: str, resource: str, action: str)
```

---

## Configuration

### GAAPConfig

```python
from gaap.core import GAAPConfig

config = GAAPConfig(
    system: SystemConfig,
    firewall: FirewallConfig,
    parser: ParserConfig,
    strategic_planner: StrategicPlannerConfig,
    tactical_decomposer: TacticalDecomposerConfig,
    execution: ExecutionConfig,
    quality_panel: QualityPanelConfig,
    security: SecurityConfig,
    budget: BudgetConfig,
    providers: list[ProviderSettings]
)
```

### ConfigBuilder

Fluent interface for building configuration.

```python
from gaap.core import ConfigBuilder

config = (ConfigBuilder()
    .with_system(name="MyApp", environment="production")
    .with_budget(monthly=1000, daily=50)
    .with_provider("groq", api_key="gsk_...")
    .with_provider("gemini", api_key="...")
    .with_security(sandbox_type="docker")
    .with_execution(max_parallel=5)
    .build())
```

### ConfigManager

Singleton configuration manager.

```python
from gaap.core import ConfigManager, get_config, load_config

# Load from file
config = load_config(config_path="config.yaml")

# Or use global manager
manager = ConfigManager(
    config_path="config.yaml",
    env_prefix="GAAP_",
    auto_reload=False
)
config = manager.config

# Get nested value
value = manager.get("system.log_level", default="INFO")

# Get API key
api_key = manager.get_api_key("groq")
```

---

## CLI Commands

### chat

Quick one-shot chat.

```bash
gaap chat "Write a binary search function"
gaap chat "Explain async/await in Python" --model llama-3.3-70b
```

### interactive

Interactive chat session.

```bash
gaap interactive
```

### providers

Manage providers.

```bash
gaap providers list
gaap providers test groq
gaap providers test --all
```

### models

View available models.

```bash
gaap models list
gaap models tiers
gaap models info llama-3.3-70b
```

### config

Configuration management.

```bash
gaap config show
gaap config set default_budget 20.0
gaap config get system.log_level
```

### history

View request history.

```bash
gaap history list
gaap history search "binary"
gaap history show <request_id>
```

### doctor

System diagnostics.

```bash
gaap doctor
```

### web

Start web UI.

```bash
gaap web
gaap web --port 8502
```

---

## Constants

### CONTEXT_LIMITS

```python
from gaap.core import CONTEXT_LIMITS

CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-3-5-sonnet": 200000,
    "gemini-1.5-pro": 1000000,
    "llama-3-70b": 128000,
    # ...
}
```

### MODEL_COSTS

```python
from gaap.core import MODEL_COSTS

MODEL_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},  # Per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    # ...
}
```

### CRITIC_WEIGHTS

```python
from gaap.core import CRITIC_WEIGHTS

CRITIC_WEIGHTS = {
    CriticType.LOGIC: 0.35,
    CriticType.SECURITY: 0.25,
    CriticType.PERFORMANCE: 0.20,
    CriticType.STYLE: 0.10,
    CriticType.COMPLIANCE: 0.05,
    CriticType.ETHICS: 0.05,
}
```

---

## Exceptions

All exceptions inherit from `GAAPException`:

```python
from gaap.core.exceptions import (
    GAAPException,
    ConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RoutingError,
    NoAvailableProviderError,
    BudgetExceededError,
    TaskError,
    SecurityError,
    PromptInjectionError,
)

try:
    response = await engine.process(request)
except ProviderRateLimitError as e:
    print(f"Rate limited: {e.details}")
    print(f"Retry after: {e.details.get('retry_after_seconds')}s")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e.details}")
```

### Exception Properties

```python
class GAAPException(Exception):
    error_code: str           # "GAAP_XXX"
    error_category: str       # "provider", "routing", etc.
    severity: str             # "debug", "info", "warning", "error", "critical"
    
    message: str
    details: dict[str, Any]
    suggestions: list[str]
    recoverable: bool
    context: dict[str, Any]
    timestamp: datetime
    traceback: str | None
    
    def to_dict() -> dict[str, Any]
```

---

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) - System architecture details
- [Providers Guide](PROVIDERS.md) - Provider setup
- [Examples](examples/) - Code examples