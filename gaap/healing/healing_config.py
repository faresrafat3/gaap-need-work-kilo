"""
Healing Configuration - Configurable Recovery Settings
=======================================================

Evolution 2026: Intelligent defaults with full user control.

Key Features:
- Preset configurations (conservative, aggressive, fast, balanced)
- Configurable healing levels
- Reflexion settings
- Pattern detection thresholds
- Post-mortem memory settings

Usage:
    # Simple - intelligent defaults
    config = HealingConfig()

    # Conservative - maximum 3 levels, quick escalation
    config = HealingConfig.conservative()

    # Aggressive - try everything
    config = HealingConfig.aggressive()

    # Fast - minimal healing
    config = HealingConfig.fast()
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ReflexionConfig:
    """Configuration for ReflexionEngine"""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.3
    enable_deep_reflexion: bool = True
    deep_reflexion_threshold: int = 3
    cache_reflections: bool = True
    max_cache_size: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_deep_reflexion": self.enable_deep_reflexion,
            "deep_reflexion_threshold": self.deep_reflexion_threshold,
            "cache_reflections": self.cache_reflections,
            "max_cache_size": self.max_cache_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflexionConfig":
        return cls(
            enabled=data.get("enabled", True),
            model=data.get("model", "gpt-4o-mini"),
            max_tokens=data.get("max_tokens", 1000),
            temperature=data.get("temperature", 0.3),
            enable_deep_reflexion=data.get("enable_deep_reflexion", True),
            deep_reflexion_threshold=data.get("deep_reflexion_threshold", 3),
            cache_reflections=data.get("cache_reflections", True),
            max_cache_size=data.get("max_cache_size", 100),
        )


@dataclass
class SemanticClassifierConfig:
    """Configuration for SemanticErrorClassifier"""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    max_tokens: int = 50
    temperature: float = 0.1
    fallback_to_regex: bool = True
    cache_classifications: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_to_regex": self.fallback_to_regex,
            "cache_classifications": self.cache_classifications,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticClassifierConfig":
        return cls(
            enabled=data.get("enabled", True),
            model=data.get("model", "gpt-4o-mini"),
            max_tokens=data.get("max_tokens", 50),
            temperature=data.get("temperature", 0.1),
            fallback_to_regex=data.get("fallback_to_regex", True),
            cache_classifications=data.get("cache_classifications", True),
        )


@dataclass
class PostMortemConfig:
    """Configuration for post-mortem analysis"""

    enabled: bool = True
    store_in_episodic_memory: bool = True
    store_in_failure_store: bool = True
    include_reflection: bool = True
    include_stack_trace: bool = True
    max_trace_length: int = 500
    negative_weight: float = -0.5
    success_weight: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "store_in_episodic_memory": self.store_in_episodic_memory,
            "store_in_failure_store": self.store_in_failure_store,
            "include_reflection": self.include_reflection,
            "include_stack_trace": self.include_stack_trace,
            "max_trace_length": self.max_trace_length,
            "negative_weight": self.negative_weight,
            "success_weight": self.success_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PostMortemConfig":
        return cls(
            enabled=data.get("enabled", True),
            store_in_episodic_memory=data.get("store_in_episodic_memory", True),
            store_in_failure_store=data.get("store_in_failure_store", True),
            include_reflection=data.get("include_reflection", True),
            include_stack_trace=data.get("include_stack_trace", True),
            max_trace_length=data.get("max_trace_length", 500),
            negative_weight=data.get("negative_weight", -0.5),
            success_weight=data.get("success_weight", 0.3),
        )


@dataclass
class PatternDetectionConfig:
    """Configuration for failure pattern detection"""

    enabled: bool = True
    detection_threshold: int = 3
    time_window_hours: int = 24
    min_similarity_score: float = 0.7
    auto_escalate_patterns: bool = True
    pattern_cooldown_minutes: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "detection_threshold": self.detection_threshold,
            "time_window_hours": self.time_window_hours,
            "min_similarity_score": self.min_similarity_score,
            "auto_escalate_patterns": self.auto_escalate_patterns,
            "pattern_cooldown_minutes": self.pattern_cooldown_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternDetectionConfig":
        return cls(
            enabled=data.get("enabled", True),
            detection_threshold=data.get("detection_threshold", 3),
            time_window_hours=data.get("time_window_hours", 24),
            min_similarity_score=data.get("min_similarity_score", 0.7),
            auto_escalate_patterns=data.get("auto_escalate_patterns", True),
            pattern_cooldown_minutes=data.get("pattern_cooldown_minutes", 30),
        )


@dataclass
class HealingConfig:
    """
    Configuration for Self-Healing System.

    Provides intelligent defaults with full user control.

    Attributes:
        max_healing_level: Maximum healing level (1-5)
        max_retries_per_level: Retries allowed at each level
        base_delay_seconds: Base delay between retries
        max_delay_seconds: Maximum delay cap
        exponential_backoff: Use exponential backoff
        jitter: Add randomness to delays

        reflexion: ReflexionEngine configuration
        semantic_classifier: SemanticErrorClassifier configuration
        post_mortem: Post-mortem analysis configuration
        pattern_detection: Pattern detection configuration

        enable_learning: Learn from recovery patterns
        enable_observability: Emit healing events
        enable_parallel_recovery: Attempt parallel recovery strategies

    Presets:
        - conservative(): Maximum 3 levels, quick escalation
        - aggressive(): Try everything, all levels
        - fast(): Minimal healing, L2 max
        - balanced(): Intelligent defaults
        - development(): Relaxed for testing
    """

    max_healing_level: Literal[1, 2, 3, 4, 5] = 4
    max_retries_per_level: int = 1
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True

    reflexion: ReflexionConfig = field(default_factory=ReflexionConfig)
    semantic_classifier: SemanticClassifierConfig = field(default_factory=SemanticClassifierConfig)
    post_mortem: PostMortemConfig = field(default_factory=PostMortemConfig)
    pattern_detection: PatternDetectionConfig = field(default_factory=PatternDetectionConfig)

    enable_learning: bool = True
    enable_observability: bool = True
    enable_parallel_recovery: bool = False

    on_escalate_callback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_retries_per_level < 1:
            raise ValueError("max_retries_per_level must be at least 1")

        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be non-negative")

        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_healing_level": self.max_healing_level,
            "max_retries_per_level": self.max_retries_per_level,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "exponential_backoff": self.exponential_backoff,
            "jitter": self.jitter,
            "reflexion": self.reflexion.to_dict(),
            "semantic_classifier": self.semantic_classifier.to_dict(),
            "post_mortem": self.post_mortem.to_dict(),
            "pattern_detection": self.pattern_detection.to_dict(),
            "enable_learning": self.enable_learning,
            "enable_observability": self.enable_observability,
            "enable_parallel_recovery": self.enable_parallel_recovery,
            "on_escalate_callback": self.on_escalate_callback,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealingConfig":
        return cls(
            max_healing_level=data.get("max_healing_level", 4),
            max_retries_per_level=data.get("max_retries_per_level", 1),
            base_delay_seconds=data.get("base_delay_seconds", 1.0),
            max_delay_seconds=data.get("max_delay_seconds", 30.0),
            exponential_backoff=data.get("exponential_backoff", True),
            jitter=data.get("jitter", True),
            reflexion=ReflexionConfig.from_dict(data.get("reflexion", {})),
            semantic_classifier=SemanticClassifierConfig.from_dict(
                data.get("semantic_classifier", {})
            ),
            post_mortem=PostMortemConfig.from_dict(data.get("post_mortem", {})),
            pattern_detection=PatternDetectionConfig.from_dict(data.get("pattern_detection", {})),
            enable_learning=data.get("enable_learning", True),
            enable_observability=data.get("enable_observability", True),
            enable_parallel_recovery=data.get("enable_parallel_recovery", False),
            on_escalate_callback=data.get("on_escalate_callback"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def conservative(cls) -> "HealingConfig":
        """
        Preset for conservative healing.

        - Maximum 3 levels (L1-L3)
        - Quick escalation
        - No deep reflexion
        - Pattern detection enabled
        """
        return cls(
            max_healing_level=3,
            max_retries_per_level=1,
            base_delay_seconds=0.5,
            max_delay_seconds=10.0,
            reflexion=ReflexionConfig(
                enabled=True,
                enable_deep_reflexion=False,
            ),
            semantic_classifier=SemanticClassifierConfig(
                enabled=True,
            ),
            post_mortem=PostMortemConfig(
                enabled=True,
                store_in_failure_store=True,
            ),
            pattern_detection=PatternDetectionConfig(
                enabled=True,
                detection_threshold=2,
                auto_escalate_patterns=True,
            ),
            enable_learning=True,
        )

    @classmethod
    def aggressive(cls) -> "HealingConfig":
        """
        Preset for aggressive healing.

        - All 5 levels
        - Multiple retries per level
        - Deep reflexion enabled
        - All features enabled
        """
        return cls(
            max_healing_level=5,
            max_retries_per_level=2,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            reflexion=ReflexionConfig(
                enabled=True,
                enable_deep_reflexion=True,
                deep_reflexion_threshold=2,
            ),
            semantic_classifier=SemanticClassifierConfig(
                enabled=True,
            ),
            post_mortem=PostMortemConfig(
                enabled=True,
                store_in_episodic_memory=True,
                store_in_failure_store=True,
            ),
            pattern_detection=PatternDetectionConfig(
                enabled=True,
                detection_threshold=3,
                auto_escalate_patterns=False,
            ),
            enable_learning=True,
            enable_parallel_recovery=True,
        )

    @classmethod
    def fast(cls) -> "HealingConfig":
        """
        Preset for fast healing.

        - Maximum 2 levels (L1-L2)
        - No deep reflexion
        - Minimal features
        - Quick failures
        """
        return cls(
            max_healing_level=2,
            max_retries_per_level=1,
            base_delay_seconds=0.1,
            max_delay_seconds=5.0,
            reflexion=ReflexionConfig(
                enabled=False,
            ),
            semantic_classifier=SemanticClassifierConfig(
                enabled=False,
                fallback_to_regex=True,
            ),
            post_mortem=PostMortemConfig(
                enabled=False,
            ),
            pattern_detection=PatternDetectionConfig(
                enabled=False,
            ),
            enable_learning=False,
        )

    @classmethod
    def balanced(cls) -> "HealingConfig":
        """
        Preset for balanced healing.

        - 4 levels (L1-L4)
        - Intelligent defaults
        - All features enabled moderately
        """
        return cls(
            max_healing_level=4,
            max_retries_per_level=1,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            reflexion=ReflexionConfig(
                enabled=True,
                enable_deep_reflexion=True,
                deep_reflexion_threshold=3,
            ),
            semantic_classifier=SemanticClassifierConfig(
                enabled=True,
            ),
            post_mortem=PostMortemConfig(
                enabled=True,
            ),
            pattern_detection=PatternDetectionConfig(
                enabled=True,
                detection_threshold=3,
            ),
            enable_learning=True,
        )

    @classmethod
    def development(cls) -> "HealingConfig":
        """
        Preset for development/testing.

        - All levels enabled
        - Longer delays
        - All features enabled
        - No escalation callback
        """
        return cls(
            max_healing_level=5,
            max_retries_per_level=2,
            base_delay_seconds=0.0,
            max_delay_seconds=5.0,
            reflexion=ReflexionConfig(
                enabled=True,
                enable_deep_reflexion=True,
                temperature=0.5,
            ),
            semantic_classifier=SemanticClassifierConfig(
                enabled=True,
            ),
            post_mortem=PostMortemConfig(
                enabled=True,
            ),
            pattern_detection=PatternDetectionConfig(
                enabled=True,
                auto_escalate_patterns=False,
            ),
            enable_learning=True,
            enable_observability=False,
        )


def create_healing_config(
    preset: Literal[
        "default", "conservative", "aggressive", "fast", "balanced", "development"
    ] = "default",
    **overrides: Any,
) -> HealingConfig:
    """
    Factory function to create HealingConfig.

    Args:
        preset: Configuration preset
        **overrides: Override specific settings

    Returns:
        HealingConfig instance

    Example:
        >>> config = create_healing_config("aggressive", max_healing_level=3)
    """
    if preset == "conservative":
        config = HealingConfig.conservative()
    elif preset == "aggressive":
        config = HealingConfig.aggressive()
    elif preset == "fast":
        config = HealingConfig.fast()
    elif preset == "balanced":
        config = HealingConfig.balanced()
    elif preset == "development":
        config = HealingConfig.development()
    else:
        config = HealingConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
