# GAAP System GLM5
# 350+ lines
__version__ = "1.0.0"
__author__ = "GAAP Team"
__email__ = "team@gaap.io"

# =============================================================================
# Engine Imports
# =============================================================================

# =============================================================================
# Context Imports
# =============================================================================
from .context import (
    ContextOrchestrator,
    ExternalBrain,
    HierarchicalContextLoader,
    PKGAgent,
    SmartChunker,
)

# =============================================================================
# Core Imports
# =============================================================================
from .core import (
    BaseAgent,
    # Base
    BaseComponent,
    BaseProvider,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConfigBuilder,
    ConfigurationError,
    ContextBudget,
    ContextWindow,
    CriticEvaluation,
    CriticType,
    ExecutionMetrics,
    ExecutionStatus,
    # Config
    GAAPConfig,
    # Exceptions
    GAAPException,
    HealingLevel,
    LayerType,
    MADDecision,
    Message,
    MessageRole,
    ModelTier,
    ProviderError,
    ProviderType,
    RoutingError,
    # Types
    Task,
    TaskComplexity,
    TaskContext,
    TaskError,
    TaskPriority,
    TaskResult,
    TaskType,
)

# =============================================================================
# Constants
# =============================================================================
from .core.types import (
    CONTEXT_LIMITS,
    CRITIC_WEIGHTS,
    MODEL_COSTS,
)
from .gaap_engine import (
    GAAPEngine,
    GAAPRequest,
    GAAPResponse,
    create_engine,
    quick_chat,
)

# =============================================================================
# Healing Imports
# =============================================================================
from .healing import (
    ErrorCategory,
    ErrorClassifier,
    SelfHealingSystem,
)

# =============================================================================
# Layer Imports
# =============================================================================
from .layers import (
    ArchitectureParadigm,
    # Layer 1
    ArchitectureSpec,
    AtomicTask,
    # Layer 3
    ExecutionResult,
    GeneticTwin,
    IntentType,
    Layer0Interface,
    Layer1Strategic,
    Layer2Tactical,
    Layer3Execution,
    MADQualityPanel,
    # Layer 0
    StructuredIntent,
    # Layer 2
    TaskGraph,
)

# =============================================================================
# Memory Imports
# =============================================================================
from .memory import (
    HierarchicalMemory,
    MemoryTier,
)

# =============================================================================
# Provider Imports
# =============================================================================
from .providers import (
    G4FProvider,
    GeminiProvider,
    GroqProvider,
)

# =============================================================================
# Routing Imports
# =============================================================================
from .routing import (
    FallbackManager,
    RoutingStrategy,
    SmartRouter,
)

# =============================================================================
# Security Imports
# =============================================================================
from .security import (
    AuditTrail,
    CapabilityManager,
    PromptFirewall,
    RiskLevel,
)

# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Engine
    "GAAPEngine",
    "GAAPRequest",
    "GAAPResponse",
    "create_engine",
    "quick_chat",
    # Core Types
    "Task",
    "TaskResult",
    "TaskPriority",
    "TaskComplexity",
    "TaskType",
    "Message",
    "MessageRole",
    "LayerType",
    "ModelTier",
    "ExecutionStatus",
    "ExecutionMetrics",
    "CriticType",
    "MADDecision",
    # Exceptions
    "GAAPException",
    "ProviderError",
    "TaskError",
    # Config
    "GAAPConfig",
    "ConfigBuilder",
    # Layers
    "Layer0Interface",
    "Layer1Strategic",
    "Layer2Tactical",
    "Layer3Execution",
    "StructuredIntent",
    "ArchitectureSpec",
    "TaskGraph",
    "AtomicTask",
    "ExecutionResult",
    # Providers
    "BaseProvider",
    "G4FProvider",
    "GroqProvider",
    "GeminiProvider",
    # Routing
    "SmartRouter",
    "RoutingStrategy",
    "FallbackManager",
    # Context
    "ContextOrchestrator",
    "PKGAgent",
    "HierarchicalContextLoader",
    "SmartChunker",
    "ExternalBrain",
    # Healing
    "SelfHealingSystem",
    "ErrorClassifier",
    "ErrorCategory",
    # Memory
    "HierarchicalMemory",
    "MemoryTier",
    # Security
    "PromptFirewall",
    "AuditTrail",
    "CapabilityManager",
    "RiskLevel",
    # Constants
    "CONTEXT_LIMITS",
    "MODEL_COSTS",
    "CRITIC_WEIGHTS",
]
