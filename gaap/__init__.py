"""
GAAP Sovereign AGI v2.1.0
=========================

General-purpose AI Agent Platform with 4-layer OODA architecture.

Architecture:
    - Layer 0: Interface (Intent parsing, security firewall)
    - Layer 1: Strategic (ToT planning, MCTS, MAD consensus)
    - Layer 2: Tactical (Task decomposition, dependency analysis)
    - Layer 3: Execution (Tool calling, quality validation)

Features:
    - Multi-provider support (Groq, Gemini, Kilo, G4F)
    - Self-healing with 5 recovery levels
    - Hierarchical memory (Working, Episodic, Semantic, Procedural)
    - Constitutional AI with axiom enforcement
    - Real-time reflection and lesson learning

Usage:
    from gaap import GAAPEngine, create_engine

    engine = create_engine(budget=10.0)
    response = await engine.chat("Hello, world!")
"""

__version__ = "2.1.0-SOVEREIGN"
__author__ = "Sovereign Strategic Architect"
__email__ = "team@gaap.io"

# =============================================================================
# Engine Imports
# =============================================================================


# =============================================================================
# Core Imports
# =============================================================================
from .core import (  # Base; Config; Exceptions; Types
    BaseAgent,
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
    GAAPConfig,
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
from .layers import (  # Layer 1; Layer 3; Layer 0; Layer 2
    ArchitectureParadigm,
    ArchitectureSpec,
    AtomicTask,
    ExecutionResult,
    GeneticTwin,
    IntentType,
    Layer0Interface,
    Layer1Strategic,
    Layer2Tactical,
    Layer3Execution,
    MADQualityPanel,
    StructuredIntent,
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
    UnifiedGAAPProvider,
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
    "UnifiedGAAPProvider",
    # Routing
    "SmartRouter",
    "RoutingStrategy",
    "FallbackManager",
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
