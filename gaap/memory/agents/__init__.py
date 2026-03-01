"""
Memory Agents Module
====================

Intelligent agents for memory operations.

Components:
- RetrievalAgent: Smart retrieval with clarification
- SpecialistAgent: Domain specialization
- ConsolidationAgent: Memory consolidation (REAP)
- MemoryRepair: Fixing incorrect memories

Reference: docs/evolution_plan_2026/25_MEMORY_AUDIT_SPEC.md
"""

from .retrieval_agent import RetrievalAgent, RetrievalContext, RetrievalResult
from .specialist_agent import DomainDecision, SpecialistAgent

__all__ = [
    "RetrievalAgent",
    "RetrievalContext",
    "RetrievalResult",
    "SpecialistAgent",
    "DomainDecision",
]
