"""
GAAP Core Module
================

Foundation components for the GAAP system:

Base Classes:
    - BaseComponent: Base for all components
    - BaseAgent: Base for all agents
    - BaseLayer: Base for OODA layers
    - BaseProvider: Base for LLM providers
    - BaseMemory: Base for memory systems
    - BaseHealer: Base for self-healing
    - BaseCritic: Base for MAD critics

Types:
    - Task, TaskPriority, TaskType
    - Message, MessageRole
    - OODAPhase, OODAState
    - ProviderType, ModelTier

Configuration:
    - GAAPConfig: Main configuration
    - ConfigManager: Configuration management
    - ConfigBuilder: Fluent config builder

Exceptions:
    - GAAPException: Base exception
    - ProviderError, TaskError, SecurityError
    - HealingError, MADError

Utilities:
    - Logging, observability, rate limiting
    - Axiom validation, governance
"""

from .base import (  # Base Classes; Context; Result; Decorators; Utilities
    BaseAgent,
    BaseComponent,
    BaseCritic,
    BaseHealer,
    BaseLayer,
    BaseMemory,
    BasePlugin,
    BaseProvider,
    ContextManager,
    ExecutionContext,
    ExecutionResult,
    create_iacp_message,
    gather_with_concurrency,
    measure_time,
    run_with_timeout,
    setup_logger,
    validate_input,
    with_retry,
)
from .config import (  # Manager; Dataclasses; Functions
    BudgetConfig,
    ConfigBuilder,
    ConfigManager,
    ContextManagementConfig,
    CriticConfig,
    ExecutionConfig,
    ExternalConnectorsConfig,
    FirewallConfig,
    GAAPConfig,
    ParserConfig,
    ProviderSettings,
    QualityPanelConfig,
    ResourceAllocatorConfig,
    SecurityConfig,
    StrategicPlannerConfig,
    SystemConfig,
    TacticalDecomposerConfig,
    get_config,
    get_config_manager,
    get_default_config,
    init_config,
    load_config,
)
from .exceptions import (  # Configuration; Context; Base; Healing; MAD; Plugin; Provider; Routing; Security; Task; Utilities
    AxiomError,
    AxiomViolationError,
    BudgetExceededError,
    CapabilityError,
    CircularDependencyError,
    ConfigLoadError,
    ConfigurationError,
    ConsensusNotReachedError,
    ContextError,
    ContextLoadError,
    ContextOverflowError,
    CriticError,
    DependencyAxiomError,
    GAAPException,
    HealingError,
    HealingFailedError,
    HumanEscalationError,
    InterfaceAxiomError,
    InvalidConfigValueError,
    MADError,
    MaxRetriesExceededError,
    MemoryAccessError,
    MissingConfigError,
    ModelNotFoundError,
    NoAvailableProviderError,
    PluginError,
    PluginExecutionError,
    PluginLoadError,
    PromptInjectionError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderNotAvailableError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    RoutingConflictError,
    RoutingError,
    SandboxEscapeError,
    SecurityError,
    SecurityScanError,
    SyntaxAxiomError,
    TaskDependencyError,
    TaskError,
    TaskExecutionError,
    TaskTimeoutError,
    TaskValidationError,
    TokenExpiredError,
    get_error_severity,
    is_recoverable,
    wrap_exception,
)
from .types import (  # Constants; Data Classes; TypedDicts; Enums
    CONTEXT_LIMITS,
    CRITIC_WEIGHTS,
    DEFAULT_CONTEXT_BUDGETS,
    MAX_RETRIES_PER_LEVEL,
    MODEL_COSTS,
    AgentCapabilities,
    AgentIdentity,
    ArchitectureSpecDict,
    AtomicTaskDict,
    CapabilityToken,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ContextBudget,
    ContextLevel,
    ContextWindow,
    CriticEvaluation,
    CriticType,
    ExecutionMetrics,
    ExecutionStatus,
    HealingLevel,
    IACPHeader,
    IACPMessage,
    IACPPayload,
    IACPSecurity,
    LayerType,
    MADDecision,
    MemoryType,
    Message,
    MessageRole,
    ModelInfo,
    ModelTier,
    OODAPhase,
    OODAState,
    ProviderConfig,
    ProviderType,
    ReplanTrigger,
    RoutingContext,
    RoutingDecision,
    SecurityRiskLevel,
    SecurityScanResult,
    StructuredIntentDict,
    SystemMetrics,
    Task,
    TaskComplexity,
    TaskContext,
    TaskPriority,
    TaskResult,
    TaskType,
    Usage,
)

__all__ = [
    # Enums
    "TaskPriority",
    "TaskComplexity",
    "TaskType",
    "LayerType",
    "ModelTier",
    "ProviderType",
    "MessageRole",
    "CriticType",
    "HealingLevel",
    "ExecutionStatus",
    "SecurityRiskLevel",
    "ContextLevel",
    "MemoryType",
    "OODAPhase",
    "ReplanTrigger",
    # Data Classes
    "Message",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "Usage",
    "ChatCompletionResponse",
    "TaskContext",
    "TaskResult",
    "Task",
    "AgentCapabilities",
    "AgentIdentity",
    "ProviderConfig",
    "ModelInfo",
    "RoutingDecision",
    "RoutingContext",
    "SecurityScanResult",
    "CapabilityToken",
    "ContextBudget",
    "ContextWindow",
    "CriticEvaluation",
    "MADDecision",
    "IACPHeader",
    "IACPPayload",
    "IACPSecurity",
    "IACPMessage",
    "ExecutionMetrics",
    "SystemMetrics",
    "OODAState",
    # Config Classes
    "FirewallConfig",
    "ParserConfig",
    "StrategicPlannerConfig",
    "ResourceAllocatorConfig",
    "TacticalDecomposerConfig",
    "ExecutionConfig",
    "CriticConfig",
    "QualityPanelConfig",
    "ExternalConnectorsConfig",
    "SecurityConfig",
    "BudgetConfig",
    "ContextManagementConfig",
    "ProviderSettings",
    "SystemConfig",
    "GAAPConfig",
    # Manager & Builder
    "ConfigManager",
    "ConfigBuilder",
    # Base Classes
    "BaseComponent",
    "BaseAgent",
    "BaseProvider",
    "BaseLayer",
    "BaseCritic",
    "BaseHealer",
    "BasePlugin",
    "BaseMemory",
    # Context
    "ContextManager",
    "ExecutionContext",
    "ExecutionResult",
    # Exceptions
    "GAAPException",
    "ConfigurationError",
    "InvalidConfigValueError",
    "MissingConfigError",
    "ConfigLoadError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderNotAvailableError",
    "ProviderRateLimitError",
    "ProviderAuthenticationError",
    "ProviderTimeoutError",
    "ModelNotFoundError",
    "ProviderResponseError",
    "RoutingError",
    "NoAvailableProviderError",
    "BudgetExceededError",
    "RoutingConflictError",
    "TaskError",
    "TaskValidationError",
    "TaskDependencyError",
    "CircularDependencyError",
    "TaskTimeoutError",
    "TaskExecutionError",
    "MaxRetriesExceededError",
    "SecurityError",
    "PromptInjectionError",
    "CapabilityError",
    "TokenExpiredError",
    "SecurityScanError",
    "SandboxEscapeError",
    "ContextError",
    "ContextOverflowError",
    "ContextLoadError",
    "MemoryAccessError",
    "MADError",
    "ConsensusNotReachedError",
    "CriticError",
    "HealingError",
    "HealingFailedError",
    "HumanEscalationError",
    "PluginError",
    "PluginLoadError",
    "PluginExecutionError",
    # Axiom Exceptions
    "AxiomError",
    "AxiomViolationError",
    "SyntaxAxiomError",
    "DependencyAxiomError",
    "InterfaceAxiomError",
    # Functions
    "get_default_config",
    "load_config",
    "init_config",
    "get_config",
    "get_config_manager",
    "wrap_exception",
    "is_recoverable",
    "get_error_severity",
    "setup_logger",
    "run_with_timeout",
    "gather_with_concurrency",
    "create_iacp_message",
    # Decorators
    "measure_time",
    "with_retry",
    "validate_input",
    # Constants
    "CONTEXT_LIMITS",
    "MODEL_COSTS",
    "DEFAULT_CONTEXT_BUDGETS",
    "CRITIC_WEIGHTS",
    "MAX_RETRIES_PER_LEVEL",
]

# Logging
from .logging import (
    get_standard_logger,
    GAAPLogger,
    configure_logging,
    get_logger,
    set_log_level,
)

__all__.extend(
    [
        # Logging
        "get_logger",
        "set_log_level",
        "configure_logging",
        "GAAPLogger",
        "get_standard_logger",
    ]
)

# External Connectors Config
from .config import ExternalConnectorsConfig

__all__.append("ExternalConnectorsConfig")

# Observability
from .observability import (
    Metrics,
    MetricsConfig,
    Observability,
    Tracer,
    TracingConfig,
    get_metrics,
    get_tracer,
    observability,
    traced,
)

# Rate Limiting
from .rate_limiter import (
    AdaptiveRateLimiter,
    BaseRateLimiter,
    CompositeRateLimiter,
    LeakyBucketRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    create_rate_limiter,
)

__all__.extend(
    [
        # Observability
        "TracingConfig",
        "MetricsConfig",
        "Tracer",
        "Metrics",
        "Observability",
        "observability",
        "get_tracer",
        "get_metrics",
        "traced",
        # Rate Limiting
        "RateLimitStrategy",
        "RateLimitConfig",
        "RateLimitResult",
        "BaseRateLimiter",
        "TokenBucketRateLimiter",
        "SlidingWindowRateLimiter",
        "LeakyBucketRateLimiter",
        "AdaptiveRateLimiter",
        "CompositeRateLimiter",
        "create_rate_limiter",
    ]
)

# World Model
from gaap.core.world_model import (
    Action,
    Prediction,
    RiskLevel,
    WorldModel,
)

__all__.extend(
    [
        "WorldModel",
        "Action",
        "Prediction",
        "RiskLevel",
    ]
)

# Dynamic Persona Engine
from gaap.core.persona import (
    Persona,
    PersonaRegistry,
    PersonaSwitcher,
    PersonaTier,
)

__all__.extend(
    [
        "Persona",
        "PersonaRegistry",
        "PersonaSwitcher",
        "PersonaTier",
    ]
)

# Semantic Distiller
from gaap.core.semantic_distiller import (
    SemanticDistiller,
    SemanticMatrix,
)

__all__.extend(
    [
        "SemanticDistiller",
        "SemanticMatrix",
    ]
)

# Contrastive Reasoning
from gaap.core.contrastive import (
    ContrastivePath,
    ContrastiveReasoner,
    ContrastiveResult,
)

__all__.extend(
    [
        "ContrastivePath",
        "ContrastiveReasoner",
        "ContrastiveResult",
    ]
)

# Semantic Pressure
from gaap.core.semantic_pressure import (
    Constraint,
    ConstraintSeverity,
    ConstraintViolation,
    SemanticConstraints,
    create_semantic_constraints,
)

__all__.extend(
    [
        "Constraint",
        "ConstraintSeverity",
        "ConstraintViolation",
        "SemanticConstraints",
        "create_semantic_constraints",
    ]
)

# Real-Time Reflection
from gaap.core.reflection import (
    ExecutionSummary,
    RealTimeReflector,
    Reflection,
    ReflectionType,
    get_reflector,
)

__all__.extend(
    [
        # Reflection
        "RealTimeReflector",
        "Reflection",
        "ReflectionType",
        "ExecutionSummary",
        "get_reflector",
    ]
)

# SOP Governance
from gaap.core.governance import (
    Artifact as GovernanceArtifact,
    ArtifactStatus,
    RoleDefinition,
    SOPExecution,
    SOPGatekeeper,
    SOPStep,
    SOPStepStatus,
    SOPStore,
    create_sop_gatekeeper,
    create_sop_store,
)

__all__.extend(
    [
        # Governance
        "GovernanceArtifact",
        "ArtifactStatus",
        "RoleDefinition",
        "SOPExecution",
        "SOPGatekeeper",
        "SOPStep",
        "SOPStepStatus",
        "SOPStore",
        "create_sop_gatekeeper",
        "create_sop_store",
    ]
)

# Declarative Signatures (DSPy-inspired)
from gaap.core.signatures import (
    Example,
    ExecutionTrace,
    FieldType,
    Module,
    ModuleRegistry,
    OptimizationResult,
    Signature,
    SignatureField,
    Teleprompter,
    get_registry,
)

__all__.extend(
    [
        # Signatures
        "FieldType",
        "SignatureField",
        "Signature",
        "Example",
        "Teleprompter",
        "OptimizationResult",
        "Module",
        "ExecutionTrace",
        "ModuleRegistry",
        "get_registry",
    ]
)

# Artifact System (MetaGPT-inspired)
from gaap.core.artifacts import (
    Artifact,
    ArtifactBuilder,
    ArtifactLink,
    ArtifactMetadata,
    ArtifactRegistry,
    ArtifactStatus as ArtifactStatusType,
    ArtifactType,
    create_artifact_registry,
)

__all__.extend(
    [
        # Artifacts
        "ArtifactType",
        "ArtifactStatusType",
        "ArtifactMetadata",
        "Artifact",
        "ArtifactLink",
        "ArtifactBuilder",
        "ArtifactRegistry",
        "create_artifact_registry",
    ]
)
