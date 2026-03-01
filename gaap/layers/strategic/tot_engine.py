"""
Tree of Thoughts (ToT) Engine for Strategic Planning

This module implements the Tree of Thoughts (ToT) algorithm for exploring
architectural decision spaces. It provides a tree-based search mechanism
that explores multiple reasoning paths to find optimal architecture specifications.

Features:
    - Multi-level tree exploration (depth=5, branching=4)
    - Adaptive option generation based on intent type
    - Path evaluation and pruning
    - Conversion of paths to architecture specifications

Classes:
    ToTNode: Node in the tree of thoughts
    ToTPath: Path representation through the tree
    ToTStrategic: Main ToT implementation

Usage:
    from gaap.layers.strategic.tot_engine import ToTStrategic

    tot = ToTStrategic(max_depth=5, branching_factor=4)
    spec, root = await tot.explore(intent, context)
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from gaap.layers.strategic.types import (
    ArchitectureDecision,
    ArchitectureParadigm,
    ArchitectureSpec,
    CommunicationPattern,
    DataStrategy,
)

if TYPE_CHECKING:
    from gaap.layers.layer0_interface import StructuredIntent

try:
    from gaap.core.logging import get_standard_logger as get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


@dataclass
class ToTNode:
    """
    Node in the Tree of Thoughts.

    Represents a single thought/decision point in the exploration tree.
    Each node contains content, a score, and references to parent/child nodes.

    Attributes:
        id: Unique identifier for this node
        level: Depth level in the tree (0 = root)
        content: The thought content/decision at this node
        score: Evaluation score (0.0-1.0)
        children: List of child nodes
        parent: Reference to parent node (None for root)
        explored: Whether this node's subtree has been explored
        pruned: Whether this node was pruned from consideration

    Example:
        >>> root = ToTNode(id="root", level=0, content="Design system")
        >>> child = ToTNode(id="root_0", level=1, content="microservices", parent=root)
        >>> root.children.append(child)
    """

    id: str
    level: int
    content: str
    score: float = 0.0
    children: list[ToTNode] = field(default_factory=list)
    parent: Optional[ToTNode] = None
    explored: bool = False
    pruned: bool = False

    def get_path(self) -> list[ToTNode]:
        """
        Get the path from root to this node.

        Returns:
            List of nodes from root to this node (inclusive)
        """
        path = [self]
        current = self.parent
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_depth(self) -> int:
        """
        Get the depth of this node in the tree.

        Returns:
            Depth level (root = 0)
        """
        return self.level


class ToTPath:
    """
    Represents a path through the Tree of Thoughts.

    A path is a sequence of nodes from root to a leaf, representing
    a complete reasoning chain for an architecture decision.

    Attributes:
        nodes: List of nodes in the path

    Example:
        >>> path = ToTPath([root, child1, grandchild])
        >>> score = path.total_score()
    """

    def __init__(self, nodes: list[ToTNode]) -> None:
        """
        Initialize a path with a sequence of nodes.

        Args:
            nodes: Ordered list of nodes from root to leaf
        """
        self.nodes = nodes

    def total_score(self) -> float:
        """
        Calculate the total score of this path.

        Returns:
            Average score of all nodes in the path
        """
        if not self.nodes:
            return 0.0
        return sum(node.score for node in self.nodes) / len(self.nodes)

    def length(self) -> int:
        """
        Get the length of this path.

        Returns:
            Number of nodes in the path
        """
        return len(self.nodes)

    def get_leaf(self) -> ToTNode:
        """
        Get the leaf node of this path.

        Returns:
            The last node in the path
        """
        return self.nodes[-1]

    def get_root(self) -> ToTNode:
        """
        Get the root node of this path.

        Returns:
            The first node in the path
        """
        return self.nodes[0]

    def to_spec(self, intent: StructuredIntent) -> ArchitectureSpec:
        """
        Convert this path to an architecture specification.

        Args:
            intent: The structured intent that initiated exploration

        Returns:
            Architecture specification derived from this path
        """
        return ToTStrategic._path_nodes_to_spec(self.nodes, intent)


class ToTStrategic:
    """
    Tree of Thoughts strategic explorer.

    Implements a tree-based search algorithm for exploring architectural
    decision spaces across 5 levels:
    - L0: Architectural paradigm
    - L1: Data strategy
    - L2: Communication pattern
    - L3: Infrastructure
    - L4: Monitoring and security

    The algorithm:
    1. Builds a tree by generating options at each level
    2. Evaluates and prunes low-scoring options
    3. Explores promising branches depth-first
    4. Selects the best path based on cumulative scores
    5. Converts the path to an architecture specification

    Attributes:
        max_depth: Maximum tree depth (default: 5)
        branching_factor: Maximum children per node (default: 4)

    Example:
        >>> tot = ToTStrategic(max_depth=5, branching_factor=4)
        >>> spec, root = await tot.explore(intent, context)
        >>> print(f"Explored {tot.get_explored_count()} nodes")
    """

    def __init__(self, max_depth: int = 5, branching_factor: int = 4) -> None:
        """
        Initialize the ToT strategic explorer.

        Args:
            max_depth: Maximum tree depth (levels to explore)
            branching_factor: Maximum number of children per node
        """
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self._logger = get_logger("gaap.layer1.tot")
        self._explored_nodes = 0

    async def explore(
        self,
        intent: StructuredIntent,
        context: dict[str, Any] | None = None,
    ) -> tuple[ArchitectureSpec, ToTNode]:
        """
        Explore the solution space and find optimal architecture.

        Builds a tree of possible architectural decisions, evaluates paths,
        and returns the best architecture specification.

        Args:
            intent: Structured intent from Layer 0
            context: Optional context dictionary for exploration

        Returns:
            Tuple of (ArchitectureSpec, root_node)

        Example:
            >>> spec, tree = await tot.explore(intent)
            >>> print(spec.paradigm)
        """
        # Create root node
        root = ToTNode(
            id="root",
            level=0,
            content=intent.explicit_goals[0] if intent.explicit_goals else "Design system",
        )

        # Build the tree
        await self._build_tree(root, intent, context)

        # Evaluate and select best path
        best_path = self._select_best_path(root)

        # Convert to specification
        spec = self._path_to_spec(best_path, intent)
        spec.explored_paths = self._explored_nodes

        return spec, root

    async def _build_tree(
        self,
        node: ToTNode,
        intent: StructuredIntent,
        context: dict[str, Any] | None,
    ) -> None:
        """
        Recursively build the tree of thoughts.

        Generates options for the current level, evaluates them,
        and recursively explores the best candidates.

        Args:
            node: Current node to expand
            intent: Structured intent for context
            context: Optional exploration context
        """
        if node.level >= self.max_depth:
            return

        # Generate options for this level
        options = self._generate_options(node.level, intent)

        # Evaluate and prune
        scored_options = []
        for opt in options[: self.branching_factor]:
            score = self._evaluate_option(opt, node.level, intent)
            if score > 0.3:  # Prune low scores
                child = ToTNode(
                    id=f"{node.id}_{len(node.children)}",
                    level=node.level + 1,
                    content=opt,
                    score=score,
                    parent=node,
                )
                node.children.append(child)
                scored_options.append((child, score))
                self._explored_nodes += 1

        # Sort and explore best options
        scored_options.sort(key=lambda x: x[1], reverse=True)

        for child, _ in scored_options[:2]:  # Explore top 2
            child.explored = True
            await self._build_tree(child, intent, context)

    def _generate_options(self, level: int, intent: StructuredIntent) -> list[str]:
        """
        Generate options for a given level based on intent type.

        Different intent types (research, diagnostic, architecture)
        have different option mappings at each level.

        Args:
            level: Current tree level (0-4)
            intent: Structured intent for context

        Returns:
            List of option strings for this level
        """
        # Import here to avoid circular imports
        from gaap.layers.layer0_interface import IntentType

        # 1. Research Thinking Stream
        if intent.intent_type in (IntentType.RESEARCH, IntentType.ANALYSIS):
            research_map = {
                0: [
                    "systematic_review",
                    "deep_dive",
                    "comparative_analysis",
                    "exploratory_research",
                ],
                1: [
                    "academic_papers",
                    "technical_docs",
                    "source_code",
                    "market_data",
                    "public_apis",
                ],
                2: ["cross_reference", "empirical_testing", "expert_validation", "logical_proof"],
                3: [
                    "qualitative_analysis",
                    "quantitative_analysis",
                    "statistical_modeling",
                    "pattern_recognition",
                ],
                4: ["detailed_report", "comparison_matrix", "executive_summary", "raw_data_dump"],
            }
            return research_map.get(level, [])

        # 2. Diagnostic Thinking Stream
        elif intent.intent_type in (IntentType.DEBUGGING, IntentType.CODE_REVIEW):
            diagnostic_map = {
                0: ["reproduce_first", "log_analysis_first", "code_audit_first", "trace_analysis"],
                1: [
                    "error_logs",
                    "system_metrics",
                    "stack_traces",
                    "network_traffic",
                    "database_state",
                ],
                2: [
                    "binary_search_isolation",
                    "component_mocking",
                    "traffic_replay",
                    "state_injection",
                ],
                3: [
                    "logic_error_hypothesis",
                    "resource_leak_hypothesis",
                    "concurrency_hypothesis",
                    "config_hypothesis",
                ],
                4: [
                    "point_fix",
                    "architectural_refactor",
                    "configuration_change",
                    "dependency_update",
                ],
            }
            return diagnostic_map.get(level, [])

        # 3. Default Architecture Thinking Stream (Standard)
        options_map: dict[int, list[Any]] = {
            0: list(ArchitectureParadigm),
            1: list(DataStrategy),
            2: list(CommunicationPattern),
            3: ["kubernetes", "docker", "serverless", "vm"],
            4: ["prometheus", "datadog", "cloudwatch", "custom"],
        }

        options = options_map.get(level, [])
        return [opt.value if hasattr(opt, "value") else str(opt) for opt in options]

    def _evaluate_option(self, option: str, level: int, intent: StructuredIntent) -> float:
        """
        Evaluate an option and return a score.

        Scoring is based on implicit requirements and the option's
        suitability for the given level.

        Args:
            option: The option to evaluate
            level: Current tree level
            intent: Structured intent with requirements

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.5  # Base score

        implicit = intent.implicit_requirements

        # Level-specific adjustments
        if level == 0:  # Architectural paradigm
            if implicit.scalability and option == "microservices":
                score += 0.3
            if implicit.budget == "budget_conscious" and option in ("monolith", "modular_monolith"):
                score += 0.2
            if len(intent.explicit_goals) > 3 and option == "microservices":
                score += 0.1

        elif level == 1:  # Data strategy
            if implicit.scalability and option == "cqrs":
                score += 0.2
            if implicit.performance == "real_time" and option == "event_sourcing":
                score += 0.2

        elif level == 2:  # Communication pattern
            if implicit.performance == "high_throughput" and option == "grpc":
                score += 0.3
            if option == "rest":
                score += 0.1  # REST is safe default

        return min(score, 1.0)

    def _select_best_path(self, root: ToTNode) -> list[ToTNode]:
        """
        Select the best path from root to leaf.

        Uses greedy selection at each level, choosing the highest
        scoring child at each step.

        Args:
            root: Root node of the tree

        Returns:
            List of nodes representing the best path
        """
        path = [root]
        current = root

        while current.children:
            # Select best child
            best_child = max(current.children, key=lambda x: x.score)
            path.append(best_child)
            current = best_child

        return path

    def _select_best_path_globally(self, root: ToTNode) -> list[ToTNode]:
        """
        Select the best path considering all leaf nodes.

        Evaluates all complete paths and selects the one with
        the highest cumulative score.

        Args:
            root: Root node of the tree

        Returns:
            List of nodes representing the globally best path
        """
        all_paths = self._get_all_paths(root)
        if not all_paths:
            return [root]

        # Score each path by average node score
        def path_score(path: list[ToTNode]) -> float:
            if not path:
                return 0.0
            return sum(n.score for n in path) / len(path)

        return max(all_paths, key=path_score)

    def _get_all_paths(self, node: ToTNode) -> list[list[ToTNode]]:
        """
        Get all paths from node to leaves.

        Args:
            node: Starting node

        Returns:
            List of all paths (each path is a list of nodes)
        """
        if not node.children:
            return [[node]]

        paths = []
        for child in node.children:
            for child_path in self._get_all_paths(child):
                paths.append([node] + child_path)
        return paths

    def _path_to_spec(self, path: list[ToTNode], intent: StructuredIntent) -> ArchitectureSpec:
        """
        Convert a path to an architecture specification.

        Delegates to static method for implementation.

        Args:
            path: List of nodes from root to leaf
            intent: Original structured intent

        Returns:
            Architecture specification
        """
        return self._path_nodes_to_spec(path, intent)

    @staticmethod
    def _path_nodes_to_spec(path: list[ToTNode], intent: StructuredIntent) -> ArchitectureSpec:
        """
        Convert path nodes to architecture specification.

        Static implementation that can be used by ToTPath as well.

        Args:
            path: List of nodes from root to leaf
            intent: Original structured intent

        Returns:
            Architecture specification
        """
        # Import here to avoid circular imports
        from gaap.layers.layer0_interface import IntentType

        spec = ArchitectureSpec(
            spec_id=f"spec_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
        )

        # 1. Handling Research Stream
        if intent.intent_type in (IntentType.RESEARCH, IntentType.ANALYSIS):
            for node in path[1:]:
                if node.level == 1:
                    spec.metadata["research_methodology"] = node.content
                elif node.level == 2:
                    spec.metadata["source_strategy"] = node.content
                elif node.level == 3:
                    spec.metadata["verification_method"] = node.content

            spec.paradigm = ArchitectureParadigm.LAYERED  # Default safe placeholder
            spec.decisions.append(
                ArchitectureDecision(
                    aspect="research_strategy",
                    choice=spec.metadata.get("research_methodology", "exploratory"),
                    reasoning="Tailored research approach for high-fidelity information gathering.",
                    trade_offs=["Speed vs. Accuracy"],
                    confidence=0.9,
                )
            )

        # 2. Handling Diagnostic Stream
        elif intent.intent_type in (IntentType.DEBUGGING, IntentType.CODE_REVIEW):
            for node in path[1:]:
                if node.level == 1:
                    spec.metadata["diagnostic_approach"] = node.content
                elif node.level == 2:
                    spec.metadata["data_collection"] = node.content
                elif node.level == 3:
                    spec.metadata["isolation_method"] = node.content

            spec.paradigm = ArchitectureParadigm.MODULAR_MONOLITH  # Placeholder
            spec.decisions.append(
                ArchitectureDecision(
                    aspect="diagnostic_strategy",
                    choice=spec.metadata.get("diagnostic_approach", "log_analysis"),
                    reasoning="Focused diagnostic path to identify root cause.",
                    trade_offs=["Completeness vs. MTTR"],
                    confidence=0.85,
                )
            )

        # 3. Standard Architecture Stream
        else:
            for node in path[1:]:
                if node.level == 1:
                    with contextlib.suppress(BaseException):
                        spec.paradigm = ArchitectureParadigm(node.content)
                elif node.level == 2:
                    with contextlib.suppress(BaseException):
                        spec.data_strategy = DataStrategy(node.content)
                elif node.level == 3:
                    with contextlib.suppress(BaseException):
                        spec.communication = CommunicationPattern(node.content)

            spec.decisions.append(
                ArchitectureDecision(
                    aspect="architecture_paradigm",
                    choice=spec.paradigm.value,
                    reasoning=f"Selected based on requirements: {intent.implicit_requirements.scalability or 'balanced'}",
                    trade_offs=["Complexity trade-off", "Team expertise required"],
                    confidence=0.85,
                )
            )

        spec.selected_path_score = sum(n.score for n in path) / len(path) if path else 0
        return spec

    def evaluate_path(self, path: ToTPath) -> float:
        """
        Evaluate a path and return its overall score.

        Public API for path evaluation. Considers node scores,
        path length, and other heuristics.

        Args:
            path: Path to evaluate

        Returns:
            Overall path score (0.0-1.0)
        """
        if not path.nodes:
            return 0.0

        # Base score: average of node scores
        base_score = path.total_score()

        # Bonus for complete paths (reach max depth)
        if path.length() >= self.max_depth:
            base_score += 0.1

        # Penalty for very short paths
        if path.length() < 3:
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    def get_explored_count(self) -> int:
        """
        Get the number of nodes explored in the last search.

        Returns:
            Count of explored nodes
        """
        return self._explored_nodes

    def reset(self) -> None:
        """
        Reset the explorer state.

        Clears the explored node counter for a fresh search.
        """
        self._explored_nodes = 0

    async def explore_with_branching(
        self,
        intent: StructuredIntent,
        context: dict[str, Any] | None = None,
        num_branches: int = 2,
    ) -> list[tuple[ArchitectureSpec, ToTNode]]:
        """
        Explore multiple branches and return top candidates.

        Instead of returning just the best path, returns the top N
        candidates for further evaluation.

        Args:
            intent: Structured intent from Layer 0
            context: Optional context dictionary
            num_branches: Number of top branches to return

        Returns:
            List of (spec, root) tuples, sorted by score
        """
        spec, root = await self.explore(intent, context)

        # Get all paths and score them
        all_paths = self._get_all_paths(root)
        scored_paths = []

        for path in all_paths:
            path_obj = ToTPath(path)
            score = self.evaluate_path(path_obj)
            scored_paths.append((score, path))

        # Sort by score descending
        scored_paths.sort(key=lambda x: x[0], reverse=True)

        # Return top N
        results = []
        for score, path in scored_paths[:num_branches]:
            branch_spec = self._path_nodes_to_spec(path, intent)
            branch_spec.selected_path_score = score
            results.append((branch_spec, root))

        return results if results else [(spec, root)]


# Backward compatibility aliases
TreeOfThoughts = ToTStrategic
