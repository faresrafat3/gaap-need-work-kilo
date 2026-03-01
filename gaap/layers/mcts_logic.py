"""
Monte Carlo Tree Search (MCTS) Strategic Planning Module.

**DEPRECATED**: This module is kept for backward compatibility.
Please import from `gaap.layers.strategic.mcts_engine` instead.

Implements: docs/evolution_plan_2026/48_MCTS_PLANNING.md

MCTS extends Tree of Thoughts with simulation and value estimation
to find the globally optimal decision path.

Features:
    - UCT (Upper Confidence Bound for Trees) selection
    - Value Oracle for success probability prediction
    - Rollout simulation for outcome estimation
    - Backpropagation for value updates
    - Configurable iterations based on task priority

Classes:
    - MCTSNode: Node in the MCTS search tree
    - MCTSConfig: Configuration parameters
    - ValueOracle: Success probability predictor
    - MCTSStrategic: Main MCTS implementation

Usage:
    # New recommended import:
    from gaap.layers.strategic.mcts_engine import MCTSStrategic, MCTSConfig

    # Legacy import (still works):
    from gaap.layers.mcts_logic import MCTSStrategic, MCTSConfig
"""

import warnings

# Re-export everything from the new location for backward compatibility
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

# Issue deprecation warning when this module is imported
warnings.warn(
    "gaap.layers.mcts_logic is deprecated. Import from gaap.layers.strategic.mcts_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
