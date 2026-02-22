"""
Self-Healing System
===================

Provides automatic recovery from errors with 5 healing levels:

Levels:
    - L1_RETRY: Simple retry with exponential backoff
    - L2_REFINE: Prompt refinement and retry
    - L3_DECOMPOSE: Task decomposition and retry
    - L4_ALTERNATIVE: Alternative provider/model
    - L5_ESCALATE: Human escalation

Components:
    - SelfHealingSystem: Main healing orchestrator
    - ErrorClassifier: Categorizes errors for appropriate healing
    - RecoveryResult: Result of healing attempt

Usage:
    from gaap.healing import SelfHealingSystem

    healer = SelfHealingSystem(max_level=HealingLevel.L4)
    result = await healer.heal(error, context)
"""

from .healer import ErrorCategory, ErrorClassifier, RecoveryResult, SelfHealingSystem

__all__ = [
    "SelfHealingSystem",
    "ErrorClassifier",
    "ErrorCategory",
    "RecoveryResult",
]
