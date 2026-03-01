"""
Memory Evolution Module
======================

Evolution and consolidation of memories.

Components:
- REAPEngine: Review, Extract, Abstract, Prune
- ClarificationSystem: Smart clarification with suggestions

Reference: docs/evolution_plan_2026/01_MEMORY_AND_DREAMING.md
"""

from .clarification import ClarificationRequest, ClarificationSystem
from .reap_engine import REAPEngine, REAPResult

__all__ = [
    "REAPEngine",
    "REAPResult",
    "ClarificationSystem",
    "ClarificationRequest",
]
