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
9. [CLI Commands](#cli-commands)

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