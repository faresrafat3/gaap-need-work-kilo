"""
Graph of Thoughts (GoT) Strategic Engine
========================================

.. deprecated::
    This module has been moved to gaap.layers.strategic.got_engine.
    Import from there for new code. This module is kept for backwards compatibility.

A graph-based reasoning engine that replaces Tree of Thoughts (ToT).
Unlike ToT's linear tree structure, GoT allows:
- Multiple parents per node (for aggregation/merging)
- Cross-level connections
- Cycles for iterative refinement

Key Components:
    - ThoughtNode: Node with multiple parents/children
    - ThoughtEdge: Edge connecting thought nodes
    - GoTGraph: Graph container
    - GoTStrategic: Main engine for graph-based exploration

Usage:
    from gaap.layers.strategic.got_engine import (
        GoTStrategic,
        ThoughtNode,
        ThoughtEdge,
        GoTGraph,
    )

    got = GoTStrategic(provider=provider)
    spec = await got.explore(intent, context)
"""

# Re-export from new location for backwards compatibility
from gaap.layers.strategic.got_engine import (
    EdgeType,
    GoTGraph,
    GoTStrategic,
    ThoughtEdge,
    ThoughtNode,
    ThoughtStatus,
    create_got_strategic,
)

__all__ = [
    "EdgeType",
    "GoTGraph",
    "GoTStrategic",
    "ThoughtEdge",
    "ThoughtNode",
    "ThoughtStatus",
    "create_got_strategic",
]
