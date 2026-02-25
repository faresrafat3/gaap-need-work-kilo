"""
Self-Healing System
===================

Provides automatic recovery from errors with 5 healing levels:

Levels:
    - L1_RETRY: Simple retry with exponential backoff
    - L2_REFINE: Prompt refinement with Reflexion
    - L3_PIVOT: Model/provider change
    - L4_STRATEGY_SHIFT: Task simplification
    - L5_HUMAN_ESCALATION: Human escalation

Components:
    - SelfHealingSystem: Main healing orchestrator
    - ErrorClassifier: Regex-based error categorization
    - SemanticErrorClassifier: LLM-powered error categorization
    - ReflexionEngine: Self-reflection for intelligent recovery
    - HealingConfig: Configuration with presets

Implements: docs/evolution_plan_2026/26_HEALING_AUDIT_SPEC.md

Usage:
    from gaap.healing import SelfHealingSystem, ReflexionEngine, HealingConfig

    # With config preset
    config = HealingConfig.aggressive()

    # With Reflexion for intelligent recovery
    reflexion = ReflexionEngine(llm_provider)
    healer = SelfHealingSystem(
        config=config,
        reflexion_engine=reflexion,
        failure_store=failure_store
    )
    result = await healer.heal(error, task, execute_func)
"""

from .healing_config import (
    HealingConfig,
    ReflexionConfig,
    SemanticClassifierConfig,
    PostMortemConfig,
    PatternDetectionConfig,
    create_healing_config,
)
from .healer import (
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
    HealingRecord,
    RecoveryAction,
    RecoveryResult,
    SelfHealingSystem,
    SemanticErrorClassifier,
)
from .reflexion import Reflection, ReflectionDepth, ReflexionEngine

__all__ = [
    "SelfHealingSystem",
    "ErrorClassifier",
    "ErrorCategory",
    "RecoveryResult",
    "RecoveryAction",
    "ErrorContext",
    "HealingRecord",
    "SemanticErrorClassifier",
    "ReflexionEngine",
    "Reflection",
    "ReflectionDepth",
    "HealingConfig",
    "ReflexionConfig",
    "SemanticClassifierConfig",
    "PostMortemConfig",
    "PatternDetectionConfig",
    "create_healing_config",
]
