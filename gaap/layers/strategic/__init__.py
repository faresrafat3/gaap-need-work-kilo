"""
Strategic Layer Components

This subpackage contains strategic planning components including:
- Monte Carlo Tree Search (MCTS) engine
- Tree of Thoughts (ToT) engine
- Graph of Thoughts (GoT) engine
- MAD Architecture Panel
- Strategic Planner
"""

# ToT imports - lazy import to avoid circular dependency
from typing import TYPE_CHECKING

# MCTS imports (self-contained, no circular deps)
from gaap.layers.strategic.mcts_engine import (
    MCTSConfig,
    MCTSNode,
    MCTSPhase,
    MCTSStrategic,
    NodeType,
    ValueOracle,
    create_mcts,
    create_mcts_for_priority,
)

if TYPE_CHECKING:
    from gaap.layers.strategic.got_engine import (
        EdgeType,
        GoTGraph,
        GoTStrategic,
        ThoughtEdge,
        ThoughtNode,
        ThoughtStatus,
        create_got_strategic,
    )
    from gaap.layers.strategic.tot_engine import ToTNode, ToTPath, ToTStrategic

__all__ = [
    "MCTSConfig",
    "MCTSNode",
    "MCTSPhase",
    "MCTSStrategic",
    "NodeType",
    "ValueOracle",
    "create_mcts",
    "create_mcts_for_priority",
]


# Lazy load ToT/GoT classes to avoid circular imports
def __getattr__(name: str):
    if name in ("ToTNode", "ToTPath", "ToTStrategic"):
        from gaap.layers.strategic.tot_engine import ToTNode, ToTPath, ToTStrategic

        return locals()[name]
    if name in (
        "EdgeType",
        "GoTGraph",
        "GoTStrategic",
        "ThoughtEdge",
        "ThoughtNode",
        "ThoughtStatus",
        "create_got_strategic",
    ):
        from gaap.layers.strategic.got_engine import (
            EdgeType,
            GoTGraph,
            GoTStrategic,
            ThoughtEdge,
            ThoughtNode,
            ThoughtStatus,
            create_got_strategic,
        )

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
