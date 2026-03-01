"""
Monte Carlo Tree Search (MCTS) Engine Module.

Unified MCTS implementation extracted from layer1_strategic.py and mcts_logic.py.
Implements the four-phase MCTS loop: Selection, Expansion, Simulation, Backpropagation.

Features:
    - UCT (Upper Confidence Bound for Trees) selection
    - Value Oracle for success probability prediction
    - Rollout simulation for outcome estimation
    - Configurable iterations based on task priority/complexity
    - Backward compatibility with original APIs

Classes:
    - MCTSPhase: Execution phases enumeration
    - NodeType: Types of MCTS nodes
    - MCTSConfig: Configuration parameters
    - MCTSNode: Node in the MCTS search tree
    - ValueOracle: Success probability predictor
    - MCTSStrategic: Main MCTS implementation

Functions:
    - create_mcts: Create MCTS with specified parameters
    - create_mcts_for_priority: Create MCTS configured for task priority

Usage:
    from gaap.layers.strategic.mcts_engine import MCTSStrategic, MCTSConfig

    config = MCTSConfig(iterations=50)
    mcts = MCTSStrategic(config, provider=provider)
    best_node, stats = await mcts.search(intent, context)
    path = mcts.get_best_path()
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from gaap.core.types import TaskComplexity, TaskPriority


class MCTSPhase(Enum):
    """MCTS execution phases."""

    SELECTION = "selection"
    EXPANSION = "expansion"
    SIMULATION = "simulation"
    BACKPROPAGATION = "backpropagation"


class NodeType(Enum):
    """Types of MCTS nodes."""

    DECISION = "decision"
    CHANCE = "chance"
    TERMINAL = "terminal"


@dataclass
class MCTSConfig:
    """
    Configuration for MCTS search.

    Attributes:
        iterations: Number of MCTS iterations (default: 50 for CRITICAL)
        exploration_weight: UCT exploration parameter (default: 1.414, sqrt(2))
        expansion_factor: Children to generate per expansion (default: 3)
        max_depth: Maximum tree depth (default: 5)
        rollout_depth: Depth for simulation rollouts (default: 5)
        parallel_rollouts: Whether to run rollouts in parallel (default: True)
        min_iterations: Minimum iterations for non-critical tasks (default: 10)

    Example:
        >>> config = MCTSConfig(iterations=100, max_depth=6)
        >>> print(config.exploration_weight)
        1.414
    """

    iterations: int = 50
    exploration_weight: float = 1.414
    expansion_factor: int = 3
    max_depth: int = 5
    rollout_depth: int = 5
    parallel_rollouts: bool = True
    min_iterations: int = 10

    @classmethod
    def for_priority(cls, priority: TaskPriority) -> MCTSConfig:
        """
        Create config based on task priority.

        Iteration counts by priority:
            - CRITICAL: 50+ iterations (ensure no catastrophic failure branches missed)
            - HIGH: 30 iterations
            - NORMAL: 20 iterations
            - LOW/BACKGROUND: 10 iterations

        Args:
            priority: Task priority level

        Returns:
            MCTSConfig configured for the priority level
        """
        iteration_map = {
            TaskPriority.CRITICAL: 50,
            TaskPriority.HIGH: 30,
            TaskPriority.NORMAL: 20,
            TaskPriority.LOW: 10,
            TaskPriority.BACKGROUND: 10,
        }
        return cls(iterations=iteration_map.get(priority, 20))

    @classmethod
    def for_complexity(cls, complexity: TaskComplexity) -> MCTSConfig:
        """
        Create config based on task complexity.

        Args:
            complexity: Task complexity level

        Returns:
            MCTSConfig configured for the complexity level
        """
        if complexity == TaskComplexity.ARCHITECTURAL:
            return cls(iterations=50, max_depth=6)
        elif complexity == TaskComplexity.COMPLEX:
            return cls(iterations=30, max_depth=5)
        elif complexity == TaskComplexity.MODERATE:
            return cls(iterations=20, max_depth=4)
        return cls(iterations=10, max_depth=3)


@dataclass
class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a decision point or state in the search space.
    Tracks visit counts and accumulated values for UCT-based selection.

    Attributes:
        id: Unique node identifier
        content: Decision content/description
        parent: Parent node (None for root)
        children: List of child nodes
        visits: Number of times this node was visited
        value: Accumulated value from rollouts
        depth: Depth in the tree (0 for root)
        node_type: Type of node (decision/chance/terminal)
        expanded: Whether this node has been expanded
        pruned: Whether this branch was pruned
        metadata: Additional node metadata

    Example:
        >>> root = MCTSNode(content="Root Decision")
        >>> child = root.add_child("Option A")
        >>> child.update(0.8)
        >>> print(child.average_value)
        0.8
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    node_type: NodeType = NodeType.DECISION
    expanded: bool = False
    pruned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """Check if node is terminal."""
        return self.node_type == NodeType.TERMINAL

    @property
    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None

    @property
    def average_value(self) -> float:
        """Get average value (Q-value). Returns 0.0 if never visited."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """
        Calculate UCT (Upper Confidence Bound for Trees) score.

        Formula: UCT = Q(s,a) + C * sqrt(ln(N(parent)) / N(s))
        Where:
            - Q(s,a) = average value (exploitation)
            - C = exploration weight
            - N = visit count

        Args:
            exploration_weight: Balance between exploration and exploitation

        Returns:
            UCT score for node selection. Returns infinity if never visited.
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.average_value

        exploitation = self.average_value
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def add_child(self, content: str, node_type: NodeType = NodeType.DECISION) -> MCTSNode:
        """
        Add a child node.

        Args:
            content: Content for the new child
            node_type: Type of the new node

        Returns:
            The newly created child node
        """
        child = MCTSNode(
            content=content,
            parent=self,
            depth=self.depth + 1,
            node_type=node_type,
        )
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """
        Update node statistics after a rollout.

        Args:
            reward: Reward value from the rollout
        """
        self.visits += 1
        self.value += reward

    def get_path(self) -> Sequence[MCTSNode]:
        """
        Get the path from root to this node.

        Returns:
            Sequence of nodes from root to this node (inclusive)
        """
        path: list[MCTSNode] = [self]
        current: MCTSNode | None = self.parent
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def get_best_child(self, exploration_weight: float = 1.414) -> MCTSNode | None:
        """
        Get the best child based on UCT score.

        Args:
            exploration_weight: UCT exploration parameter

        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))

    def get_most_visited_child(self) -> MCTSNode | None:
        """
        Get the most visited child (best after search completes).

        Returns:
            Most visited child node or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)

    def prune(self) -> None:
        """Mark this node and all descendants as pruned."""
        self.pruned = True
        for child in self.children:
            child.prune()

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "visits": self.visits,
            "value": self.value,
            "avg_value": self.average_value,
            "depth": self.depth,
            "node_type": self.node_type.value,
            "children": [c.to_dict() for c in self.children],
            "pruned": self.pruned,
        }


class ValueOracle:
    """
    Value Oracle: Predicts success probability of a decision node.

    The Oracle estimates the "Success Probability" of a decision path using
    both heuristic scoring and LLM-based evaluation when available.

    Attributes:
        provider: LLM provider for value estimation
        model: Model to use for LLM evaluation
        HEURISTIC_WEIGHTS: Weights for heuristic scoring factors

    Example:
        >>> oracle = ValueOracle(provider=llm_provider)
        >>> value = await oracle.evaluate(node, context, intent)
    """

    HEURISTIC_WEIGHTS = {
        "scalability_match": 0.15,
        "complexity_match": 0.15,
        "security_risk": 0.20,
        "cost_efficiency": 0.15,
        "maintainability": 0.15,
        "team_fit": 0.20,
    }

    VALUE_PROMPT = """You are a Value Oracle for architectural decisions.
Given the context and decision, predict the probability of success (0.0 to 1.0).

## Context
{context}

## Decision Path
{decision_path}

## Proposed Action
{action}

## Instructions
Consider:
1. Does this align with the stated goals?
2. Are there any obvious risks or blockers?
3. Is this the right level of complexity?

Return ONLY a float between 0.0 and 1.0 representing success probability.
No explanation, no other text."""

    def __init__(self, provider: Any = None, model: str | None = None) -> None:
        """
        Initialize the Value Oracle.

        Args:
            provider: LLM provider for value estimation
            model: Model to use for evaluation (default: llama-3.3-70b-versatile)
        """
        self.provider = provider
        self.model = model or "llama-3.3-70b-versatile"
        self._logger = logging.getLogger("gaap.mcts.oracle")
        self._cache: dict[str, float] = {}

    async def evaluate(
        self,
        node: MCTSNode,
        context: dict[str, Any],
        intent: Any = None,
    ) -> float:
        """
        Evaluate the success probability of a node.

        Tries LLM evaluation first if provider available, falls back to heuristics.
        Results are cached for performance.

        Args:
            node: Node to evaluate
            context: Evaluation context
            intent: Original intent/requirements

        Returns:
            Success probability (0.0 to 1.0)
        """
        cache_key = f"{node.id}:{node.content}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.provider is not None:
            try:
                value = await self._llm_evaluate(node, context, intent)
                self._cache[cache_key] = value
                return value
            except Exception as e:
                self._logger.warning(f"LLM evaluation failed, using heuristic: {e}")

        value = self._heuristic_evaluate(node, context, intent)
        self._cache[cache_key] = value
        return value

    async def _llm_evaluate(self, node: MCTSNode, context: dict[str, Any], intent: Any) -> float:
        """Use LLM to evaluate node value."""
        from gaap.core.types import Message, MessageRole

        path = " -> ".join([n.content for n in node.get_path()])
        prompt = self.VALUE_PROMPT.format(
            context=str(context)[:500],
            decision_path=path,
            action=node.content,
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a precise value oracle."),
            Message(role=MessageRole.USER, content=prompt),
        ]

        response = await self.provider.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.3,
            max_tokens=10,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_value(raw)

    def _parse_value(self, raw: str) -> float:
        """Parse LLM response to float value (0.0 to 1.0)."""
        import re

        numbers = re.findall(r"-?[\d.]+", raw)
        if numbers:
            value = float(numbers[0])
            return max(0.0, min(1.0, value))
        return 0.5

    def _heuristic_evaluate(self, node: MCTSNode, context: dict[str, Any], intent: Any) -> float:
        """Heuristic evaluation without LLM."""
        score = 0.5
        content_lower = node.content.lower()

        # Architecture-specific heuristics
        if "microservices" in content_lower:
            if context.get("scalability_required"):
                score += 0.15
            if context.get("budget_conscious"):
                score -= 0.20
        elif "monolith" in content_lower:
            if context.get("budget_conscious"):
                score += 0.15
            if context.get("scalability_required"):
                score -= 0.10

        # Infrastructure heuristics
        if "kubernetes" in content_lower or "docker" in content_lower:
            if context.get("team_ops_expertise"):
                score += 0.10
            else:
                score -= 0.10

        # Security bonus
        if "security" in content_lower:
            score += 0.10

        # Depth penalty (deeper paths are riskier)
        if node.depth > 3:
            score -= 0.05 * (node.depth - 3)

        return max(0.0, min(1.0, score))

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()


class MCTSStrategic:
    """
    Monte Carlo Tree Search Strategic Planner.

    Implements the 4-phase MCTS loop:
    1. SELECTION: Traverse tree using UCT to find promising leaf
    2. EXPANSION: Generate new child nodes from the selected node
    3. SIMULATION: Rollout to estimate value of the expansion
    4. BACKPROPAGATION: Update values up the tree from the simulation

    Attributes:
        config: MCTS configuration
        oracle: Value oracle for evaluation
        simulator: Optional simulator for rollouts
        provider: LLM provider for expansion

    Example:
        >>> mcts = MCTSStrategic(config=MCTSConfig(iterations=50))
        >>> best_node, stats = await mcts.search(intent, context)
        >>> path = mcts.get_best_path()
    """

    def __init__(
        self,
        config: MCTSConfig | None = None,
        provider: Any = None,
        simulator: Any = None,
    ) -> None:
        """
        Initialize MCTS Strategic.

        Args:
            config: MCTS configuration (default: MCTSConfig())
            provider: LLM provider for expansion
            simulator: Optional simulator for rollouts
        """
        self.config = config or MCTSConfig()
        self.oracle = ValueOracle(provider=provider)
        self.simulator = simulator
        self.provider = provider
        self._root: MCTSNode | None = None
        self._logger = logging.getLogger("gaap.mcts")
        self._stats: dict[str, Any] = {
            "iterations": 0,
            "nodes_created": 0,
            "rollouts": 0,
            "llm_calls": 0,
        }

    async def search(
        self,
        intent: Any,
        context: dict[str, Any] | None = None,
        root_content: str = "Root Decision",
        timeout_seconds: float = 30.0,
    ) -> tuple[MCTSNode, dict[str, Any]]:
        """
        Perform MCTS search.

        Runs the 4-phase MCTS loop for the configured number of iterations.
        Returns the best node found and search statistics.

        Args:
            intent: Intent/requirements to plan for
            context: Additional context dictionary
            root_content: Content for root node

        Returns:
            Tuple of (best result node, statistics dictionary)
        """
        context = context or {}
        self._root = MCTSNode(content=root_content, node_type=NodeType.DECISION)

        self._stats = {
            "iterations": 0,
            "nodes_created": 1,
            "rollouts": 0,
            "llm_calls": 0,
        }

        start_time = time.time()
        for i in range(self.config.iterations):
            # Check timeout
            if time.time() - start_time >= timeout_seconds:
                self._logger.warning(f"MCTS search timeout after {i} iterations")
                break
            self._stats["iterations"] = i + 1

            # Phase 1: Selection
            node = self._select(self._root)

            # Phase 2: Expansion
            if not node.is_terminal and node.depth < self.config.max_depth:
                await self._expand(node, intent, context)

            # Phase 3: Simulation
            reward = await self._simulate(node, intent, context)

            # Phase 4: Backpropagation
            self._backpropagate(node, reward)

            if i % 10 == 0:
                self._logger.debug(f"MCTS iteration {i + 1}/{self.config.iterations}")

        # Select best child of root as result
        best_child = self._root.get_most_visited_child()
        result_node = best_child if best_child else self._root

        self._stats["final_value"] = result_node.average_value
        self._stats["total_visits"] = self._root.visits
        self._stats["tree_depth"] = self._get_tree_depth(self._root)
        self._stats["timeout_reached"] = time.time() - start_time >= timeout_seconds

        return result_node, self._stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCT until leaf found.

        Args:
            node: Starting node (typically root)

        Returns:
            Selected leaf node for expansion
        """
        while not node.is_leaf and not node.is_terminal:
            best_child = node.get_best_child(self.config.exploration_weight)
            if best_child is None:
                break
            node = best_child
        return node

    async def _expand(self, node: MCTSNode, intent: Any, context: dict[str, Any]) -> None:
        """
        Expansion phase: Generate new child nodes.

        Args:
            node: Node to expand
            intent: Intent for expansion context
            context: Additional context
        """
        if node.expanded:
            return

        actions = await self._generate_actions(node, intent, context)

        for action in actions[: self.config.expansion_factor]:
            node.add_child(content=action)
            self._stats["nodes_created"] += 1
            self._logger.debug(f"Expanded node {node.id} with child: {action}")

        node.expanded = True

    async def _generate_actions(
        self, node: MCTSNode, intent: Any, context: dict[str, Any]
    ) -> list[str]:
        """
        Generate possible actions from a node.

        Args:
            node: Node to generate actions from
            intent: Intent for context
            context: Additional context

        Returns:
            List of possible actions/decisions
        """
        level = node.depth
        return self._get_default_actions(level)

    def _get_default_actions(self, level: int) -> list[str]:
        """
        Get default actions based on decision level.

        Level mapping:
            0: Architecture paradigms
            1: Data strategies
            2: Communication patterns
            3: Deployment platforms
            4: Monitoring solutions
            5: Observability strategies
        """
        action_map: dict[int, list[str]] = {
            0: ["microservices", "modular_monolith", "monolith", "serverless"],
            1: ["single_database", "polyglot", "cqrs", "event_sourcing"],
            2: ["rest", "graphql", "grpc", "message_queue"],
            3: ["kubernetes", "docker_compose", "serverless_platform", "vm"],
            4: ["prometheus", "datadog", "cloudwatch", "custom"],
            5: ["centralized_logging", "distributed_tracing", "both"],
        }
        return action_map.get(level, ["explore_alternative"])

    async def _simulate(self, node: MCTSNode, intent: Any, context: dict[str, Any]) -> float:
        """
        Simulation phase: Rollout to estimate value.

        Args:
            node: Node to simulate from
            intent: Intent for simulation context
            context: Additional context

        Returns:
            Reward value from simulation (0.0 to 1.0)
        """
        self._stats["rollouts"] += 1

        if self.simulator is not None:
            try:
                return await self._simulator_rollout(node, intent, context)
            except Exception as e:
                self._logger.warning(f"Simulator rollout failed: {e}")

        return await self._oracle_rollout(node, intent, context)

    async def _simulator_rollout(
        self, node: MCTSNode, intent: Any, context: dict[str, Any]
    ) -> float:
        """Rollout using the external simulator if available."""
        import random

        current = node
        path = [current.content]
        reward = 0.0

        for _ in range(self.config.rollout_depth):
            if current.is_terminal:
                break

            actions = self._get_default_actions(current.depth)
            if not actions:
                break

            action = random.choice(actions)
            path.append(action)

            if self.simulator:
                sim_result = await self.simulator.simulate_action(action)
                reward += 1.0 - sim_result.risk_score

        if self.oracle:
            final_value = await self.oracle.evaluate(node, context, intent)
            reward = reward / max(len(path), 1) + final_value * 0.5

        return max(0.0, min(1.0, reward))

    async def _oracle_rollout(self, node: MCTSNode, intent: Any, context: dict[str, Any]) -> float:
        """Rollout using the Value Oracle for evaluation."""
        value = await self.oracle.evaluate(node, context, intent)
        if self.provider is not None:
            self._stats["llm_calls"] += 1
        return value

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update values up the tree.

        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate up
        """
        current: MCTSNode | None = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _get_tree_depth(self, node: MCTSNode) -> int:
        """Calculate the maximum depth of the tree from given node."""
        if node.is_leaf:
            return node.depth
        return max(self._get_tree_depth(child) for child in node.children)

    def get_best_path(self) -> list[MCTSNode]:
        """
        Get the best path through the tree (most visited path).

        Returns:
            List of nodes representing the best path from root to leaf
        """
        if self._root is None:
            return []

        path = [self._root]
        current = self._root

        while current.children:
            best = current.get_most_visited_child()
            if best is None:
                break
            path.append(best)
            current = best

        return path

    def get_tree_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search tree.

        Returns:
            Dictionary with tree statistics including visits, values, depth
        """
        if self._root is None:
            return {"error": "No tree created yet"}

        most_visited = self._root.get_most_visited_child()
        best_child_visits = most_visited.visits if most_visited else 0

        return {
            "root_visits": self._root.visits,
            "root_value": self._root.value,
            "children_count": len(self._root.children),
            "best_child_visits": best_child_visits,
            "tree_depth": self._get_tree_depth(self._root),
            **self._stats,
        }


def create_mcts(
    iterations: int = 50,
    provider: Any = None,
    simulator: Any = None,
    exploration_weight: float = 1.414,
) -> MCTSStrategic:
    """
    Create an MCTS instance with specified parameters.

    Args:
        iterations: Number of MCTS iterations
        provider: LLM provider for oracle evaluation
        simulator: Optional simulator for rollouts
        exploration_weight: UCT exploration weight (default: sqrt(2))

    Returns:
        Configured MCTSStrategic instance

    Example:
        >>> mcts = create_mcts(iterations=100, provider=llm)
        >>> best_node, stats = await mcts.search(intent, context)
    """
    config = MCTSConfig(
        iterations=iterations,
        exploration_weight=exploration_weight,
    )
    return MCTSStrategic(config=config, provider=provider, simulator=simulator)


def create_mcts_for_priority(
    priority: TaskPriority,
    provider: Any = None,
    simulator: Any = None,
) -> MCTSStrategic:
    """
    Create an MCTS instance configured for a task priority.

    Args:
        priority: Task priority level
        provider: LLM provider for oracle evaluation
        simulator: Optional simulator for rollouts

    Returns:
        MCTSStrategic configured for the priority level

    Example:
        >>> mcts = create_mcts_for_priority(TaskPriority.CRITICAL, provider=llm)
        >>> best_node, stats = await mcts.search(intent, context)
    """
    config = MCTSConfig.for_priority(priority)
    return MCTSStrategic(config=config, provider=provider, simulator=simulator)
