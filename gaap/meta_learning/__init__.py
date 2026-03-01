"""
Meta-Learning Module - Recursive Wisdom & Self-Improvement
===========================================================

Implements: docs/evolution_plan_2026/42_META_LEARNING_AUDIT_SPEC.md

The agent learns from its own experiences:
- Wisdom Distillation: Extract principles from successes
- Failure Store: Learn from mistakes with corrective actions
- Axiom Bridge: Propose new constitutional rules
- Confidence Scoring: Know what it doesn't know

Usage:
    from gaap.meta_learning import MetaLearner, WisdomDistiller, FailureStore

    learner = MetaLearner()
    await learner.run_dream_cycle()
"""

from gaap.meta_learning.axiom_bridge import (
    AxiomBridge,
    AxiomProposal,
    ProposalStatus,
)
from gaap.meta_learning.confidence import (
    ConfidenceCalculator,
    ConfidenceFactors,
)
from gaap.meta_learning.failure_store import (
    CorrectiveAction,
    FailedTrace,
    FailureStore,
    FailureType,
)
from gaap.meta_learning.meta_learner import (
    DreamCycleResult,
    MetaLearner,
)
from gaap.meta_learning.wisdom_distiller import (
    DistillationResult,
    ProjectHeuristic,
    WisdomDistiller,
)

__all__ = [
    "MetaLearner",
    "DreamCycleResult",
    "WisdomDistiller",
    "ProjectHeuristic",
    "DistillationResult",
    "FailureStore",
    "FailedTrace",
    "CorrectiveAction",
    "FailureType",
    "AxiomBridge",
    "AxiomProposal",
    "ProposalStatus",
    "ConfidenceCalculator",
    "ConfidenceFactors",
]
