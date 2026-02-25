"""
Layer 2 Configuration - Intelligent Defaults with Advanced Control
==================================================================

Evolution 2026: Implements the "Intelligent Default + Advanced Options" pattern.

Users get smart defaults (LLM decides based on context) but can override
with specific modes for maximum control when needed.

Usage:
    # Simple - intelligent defaults
    config = Layer2Config()

    # Advanced - full control
    config = Layer2Config(
        phase_reassessment_mode="full",
        dependency_depth="exhaustive",
        injection_autonomy="conservative",
    )
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal


class PhaseDiscoveryMode(Enum):
    """How phases are discovered from architecture spec"""

    AUTO = auto()  # LLM decides based on complexity
    STANDARD = auto()  # Fast, template-based
    DEEP = auto()  # LLM analyzes deeply


class PhaseReassessmentMode(Enum):
    """How phases are reassessed after completion"""

    AUTO = auto()  # LLM decides: risk_based or full
    RISK_BASED = auto()  # Analyze high-risk areas only
    FULL = auto()  # Complete reanalysis


class DependencyTiming(Enum):
    """When dependency resolution happens"""

    AUTO = auto()  # LLM decides: hybrid usually
    PRE = auto()  # Resolve all before execution
    JIT = auto()  # Resolve just-in-time per phase
    CONTINUOUS = auto()  # Re-resolve during execution
    HYBRID = auto()  # Pre for critical, JIT for others


class DependencyDepth(Enum):
    """How deep dependency analysis goes"""

    STANDARD = auto()  # Obvious dependencies
    DEEP = auto()  # Semantic analysis
    EXHAUSTIVE = auto()  # Full codebase analysis


class InjectionAutonomy(Enum):
    """How autonomous task injection is"""

    AUTO = auto()  # LLM decides based on risk
    AUTONOMOUS = auto()  # Always inject without asking
    SEMI = auto()  # Inject low-risk, ask for high-risk
    CONSERVATIVE = auto()  # Always ask before injecting


class SchemaDetailLevel(Enum):
    """How detailed task schemas are"""

    STANDARD = auto()  # Basic inputs/outputs
    DETAILED = auto()  # With types and constraints
    COMPREHENSIVE = auto()  # Full specifications with examples


class RiskAnalysisMode(Enum):
    """How thorough risk analysis is"""

    STANDARD = auto()  # Common risk patterns
    DEEP = auto()  # Extended analysis
    PARANOID = auto()  # Exhaustive risk detection


class ToolRecommendationDepth(Enum):
    """How tool recommendations are generated"""

    BASIC = auto()  # Simple tool list
    REASONED = auto()  # With reasoning for each
    EXHAUSTIVE = auto()  # Full alternatives and fallbacks


@dataclass
class Layer2Config:
    """
    Configuration for Layer 2 Tactical Planning.

    Provides intelligent defaults while allowing full user control.
    All "auto" modes mean LLM decides based on context.

    Attributes:
        # Phase Planning
        phase_discovery_mode: How phases are discovered
        phase_reassessment_mode: How phases are reassessed after completion

        # Dependency Resolution
        dependency_timing: When dependencies are resolved
        dependency_depth: How deep analysis goes

        # Task Injection
        injection_autonomy: How autonomous injection is
        injection_risk_threshold: Risk threshold for asking user (0.0-1.0)

        # Schema Generation
        schema_detail_level: How detailed task schemas are
        tool_recommendation_depth: How tool recommendations are generated

        # Risk Assessment
        risk_analysis_mode: How thorough risk analysis is

        # Learning
        learning_enabled: Whether tactical learning is enabled
        store_episodes: Whether to store execution episodes

        # Execution Limits
        max_phase_expansion_depth: Maximum depth of phase expansion
        parallel_task_limit: Maximum parallel tasks
        critical_path_optimization: Whether to optimize critical path

        # Advanced
        llm_temperature: Temperature for LLM calls
        fallback_enabled: Whether to use fallback strategies
    """

    # Phase Planning
    phase_discovery_mode: Literal["auto", "standard", "deep"] = "auto"
    phase_reassessment_mode: Literal["auto", "risk_based", "full"] = "auto"

    # Dependency Resolution
    dependency_timing: Literal["auto", "pre", "jit", "continuous", "hybrid"] = "auto"
    dependency_depth: Literal["standard", "deep", "exhaustive"] = "standard"

    # Task Injection
    injection_autonomy: Literal["auto", "autonomous", "semi", "conservative"] = "auto"
    injection_risk_threshold: float = 0.7

    # Schema Generation
    schema_detail_level: Literal["standard", "detailed", "comprehensive"] = "standard"
    tool_recommendation_depth: Literal["basic", "reasoned", "exhaustive"] = "reasoned"

    # Risk Assessment
    risk_analysis_mode: Literal["standard", "deep", "paranoid"] = "standard"

    # Learning
    learning_enabled: bool = True
    store_episodes: bool = True

    # JIT Tool Synthesis
    enable_jit_synthesis: bool = True

    # Execution Limits
    max_phase_expansion_depth: int = 3
    parallel_task_limit: int = 5
    critical_path_optimization: bool = True

    # Advanced
    llm_temperature: float = 0.3
    fallback_enabled: bool = True

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration"""
        if not 0.0 <= self.injection_risk_threshold <= 1.0:
            raise ValueError("injection_risk_threshold must be between 0.0 and 1.0")

        if self.parallel_task_limit < 1:
            raise ValueError("parallel_task_limit must be at least 1")

        if self.max_phase_expansion_depth < 1:
            raise ValueError("max_phase_expansion_depth must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "phase_discovery_mode": self.phase_discovery_mode,
            "phase_reassessment_mode": self.phase_reassessment_mode,
            "dependency_timing": self.dependency_timing,
            "dependency_depth": self.dependency_depth,
            "injection_autonomy": self.injection_autonomy,
            "injection_risk_threshold": self.injection_risk_threshold,
            "schema_detail_level": self.schema_detail_level,
            "tool_recommendation_depth": self.tool_recommendation_depth,
            "risk_analysis_mode": self.risk_analysis_mode,
            "learning_enabled": self.learning_enabled,
            "store_episodes": self.store_episodes,
            "enable_jit_synthesis": self.enable_jit_synthesis,
            "max_phase_expansion_depth": self.max_phase_expansion_depth,
            "parallel_task_limit": self.parallel_task_limit,
            "critical_path_optimization": self.critical_path_optimization,
            "llm_temperature": self.llm_temperature,
            "fallback_enabled": self.fallback_enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer2Config":
        """Create from dictionary"""
        return cls(
            phase_discovery_mode=data.get("phase_discovery_mode", "auto"),
            phase_reassessment_mode=data.get("phase_reassessment_mode", "auto"),
            dependency_timing=data.get("dependency_timing", "auto"),
            dependency_depth=data.get("dependency_depth", "standard"),
            injection_autonomy=data.get("injection_autonomy", "auto"),
            injection_risk_threshold=data.get("injection_risk_threshold", 0.7),
            schema_detail_level=data.get("schema_detail_level", "standard"),
            tool_recommendation_depth=data.get("tool_recommendation_depth", "reasoned"),
            risk_analysis_mode=data.get("risk_analysis_mode", "standard"),
            learning_enabled=data.get("learning_enabled", True),
            store_episodes=data.get("store_episodes", True),
            enable_jit_synthesis=data.get("enable_jit_synthesis", True),
            max_phase_expansion_depth=data.get("max_phase_expansion_depth", 3),
            parallel_task_limit=data.get("parallel_task_limit", 5),
            critical_path_optimization=data.get("critical_path_optimization", True),
            llm_temperature=data.get("llm_temperature", 0.3),
            fallback_enabled=data.get("fallback_enabled", True),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def high_quality(cls) -> "Layer2Config":
        """Preset for maximum quality (slower, more thorough)"""
        return cls(
            phase_discovery_mode="deep",
            phase_reassessment_mode="full",
            dependency_timing="hybrid",
            dependency_depth="exhaustive",
            injection_autonomy="semi",
            injection_risk_threshold=0.5,
            schema_detail_level="comprehensive",
            tool_recommendation_depth="exhaustive",
            risk_analysis_mode="paranoid",
            learning_enabled=True,
            store_episodes=True,
        )

    @classmethod
    def fast(cls) -> "Layer2Config":
        """Preset for speed (less thorough)"""
        return cls(
            phase_discovery_mode="standard",
            phase_reassessment_mode="risk_based",
            dependency_timing="jit",
            dependency_depth="standard",
            injection_autonomy="autonomous",
            injection_risk_threshold=0.8,
            schema_detail_level="standard",
            tool_recommendation_depth="basic",
            risk_analysis_mode="standard",
            learning_enabled=False,
            store_episodes=False,
        )

    @classmethod
    def balanced(cls) -> "Layer2Config":
        """Preset for balanced quality and speed"""
        return cls(
            phase_discovery_mode="auto",
            phase_reassessment_mode="auto",
            dependency_timing="auto",
            dependency_depth="deep",
            injection_autonomy="auto",
            injection_risk_threshold=0.7,
            schema_detail_level="detailed",
            tool_recommendation_depth="reasoned",
            risk_analysis_mode="deep",
            learning_enabled=True,
            store_episodes=True,
        )


def create_layer2_config(
    preset: Literal["default", "high_quality", "fast", "balanced"] = "default", **overrides: Any
) -> Layer2Config:
    """
    Factory function to create Layer2Config.

    Args:
        preset: Configuration preset
        **overrides: Override specific settings

    Returns:
        Layer2Config instance

    Example:
        >>> config = create_layer2_config("high_quality", parallel_task_limit=10)
    """
    if preset == "high_quality":
        config = Layer2Config.high_quality()
    elif preset == "fast":
        config = Layer2Config.fast()
    elif preset == "balanced":
        config = Layer2Config.balanced()
    else:
        config = Layer2Config()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
