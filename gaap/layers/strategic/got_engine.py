"""
Graph of Thoughts (GoT) Engine
==============================

A graph-based reasoning engine that enables complex multi-step reasoning
through graph structures. Unlike Tree of Thoughts (ToT), GoT allows:
- Multiple parents per node (for aggregation/merging)
- Cross-level connections
- Cycles for iterative refinement
- Evidence-based scoring

Key Components:
    - ThoughtNode: Node representing a single thought/decision
    - ThoughtEdge: Edge connecting thoughts with relationship metadata
    - GoTGraph: Container for the complete graph structure
    - GoTStrategic: Main engine for graph-based exploration

Usage:
    from gaap.layers.strategic.got_engine import (
        GoTStrategic,
        ThoughtNode,
        ThoughtEdge,
        GoTGraph,
    )

    got = GoTStrategic(provider=provider, max_nodes=50, max_generations=4)
    spec = await got.explore(intent, context)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer0_interface import StructuredIntent
from gaap.layers.strategic.types import (
    ArchitectureDecision,
    ArchitectureParadigm,
    ArchitectureSpec,
    CommunicationPattern,
    DataStrategy,
)

logger = get_logger("gaap.layer1.got")


class ThoughtStatus(Enum):
    """Status of a thought node in the graph.

    Attributes:
        DRAFT: Initial state, not yet evaluated
        EVALUATED: Has been scored and validated
        AGGREGATED: Combined from multiple parent nodes
        REFINED: Improved based on feedback
        PRUNED: Removed from consideration due to low score
        SELECTED: Chosen as the final solution
    """

    DRAFT = auto()
    EVALUATED = auto()
    AGGREGATED = auto()
    REFINED = auto()
    PRUNED = auto()
    SELECTED = auto()


class EdgeType(Enum):
    """Type of relationship between thought nodes.

    Attributes:
        GENERATES: Parent generates child (hierarchical)
        SUPPORTS: Node supports another (evidential)
        CONTRADICTS: Node contradicts another (conflict)
        REFINES: Node refines/improves another
        MERGES: Result of merging multiple nodes
    """

    GENERATES = auto()
    SUPPORTS = auto()
    CONTRADICTS = auto()
    REFINES = auto()
    MERGES = auto()


@dataclass
class ThoughtEdge:
    """Edge connecting two thought nodes in the graph.

    Edges carry metadata about the relationship between thoughts,
    enabling rich graph traversal and reasoning.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of relationship
        weight: Edge weight for scoring (0.0-1.0)
        metadata: Additional relationship metadata

    Example:
        >>> edge = ThoughtEdge(
        ...     source_id="node_1",
        ...     target_id="node_2",
        ...     edge_type=EdgeType.GENERATES,
        ...     weight=0.8,
        ... )
    """

    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.GENERATES
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert edge to dictionary representation.

        Returns:
            Dictionary with edge attributes
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.name,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class ThoughtNode:
    """A node in the Graph of Thoughts.

    Unlike ToT's tree nodes, GoT nodes can have multiple parents,
    enabling aggregation of multiple solutions and complex reasoning patterns.

    Attributes:
        id: Unique identifier
        content: The architecture decision or reasoning text
        parents: List of parent node IDs (for aggregation)
        children: List of child node IDs
        score: Evaluation score (0.0-1.0)
        valid: Whether this node is still valid
        evidence: List of evidence supporting this thought
        generation: Which generation/level this node belongs to
        status: Current status of the node
        metadata: Additional metadata

    Example:
        >>> node = ThoughtNode(
        ...     id="thought_1",
        ...     content="Use microservices architecture",
        ...     generation=1,
        ... )
    """

    id: str
    content: ArchitectureDecision | str
    parents: list[ThoughtNode] = field(default_factory=list)
    children: list[ThoughtNode] = field(default_factory=list)
    score: float = 0.0
    valid: bool = True
    evidence: list[str] = field(default_factory=list)
    generation: int = 0
    status: ThoughtStatus = ThoughtStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_id(self) -> str:
        """Generate a unique hash ID for this node.

        Returns:
            12-character hexadecimal hash string
        """
        content_str = str(self.content)[:100]
        return hashlib.md5(f"{self.id}:{content_str}:{self.generation}".encode()).hexdigest()[:12]

    def add_child(self, child: ThoughtNode) -> None:
        """Add a child node and establish bidirectional relationship.

        Args:
            child: The child node to add
        """
        self.children.append(child)
        child.parents.append(self)

    def add_parent(self, parent: ThoughtNode) -> None:
        """Add a parent node and establish bidirectional relationship.

        Args:
            parent: The parent node to add
        """
        self.parents.append(parent)
        parent.children.append(self)

    def get_ancestors(self) -> list[ThoughtNode]:
        """Get all ancestor nodes (parents, grandparents, etc.).

        Returns:
            List of ancestor nodes in traversal order
        """
        ancestors: list[ThoughtNode] = []
        visited: set[str] = set()

        def traverse(node: ThoughtNode) -> None:
            for parent in node.parents:
                if parent.id not in visited:
                    visited.add(parent.id)
                    ancestors.append(parent)
                    traverse(parent)

        traverse(self)
        return ancestors

    def get_descendants(self) -> list[ThoughtNode]:
        """Get all descendant nodes (children, grandchildren, etc.).

        Returns:
            List of descendant nodes in traversal order
        """
        descendants: list[ThoughtNode] = []
        visited: set[str] = set()

        def traverse(node: ThoughtNode) -> None:
            for child in node.children:
                if child.id not in visited:
                    visited.add(child.id)
                    descendants.append(child)
                    traverse(child)

        traverse(self)
        return descendants

    def get_siblings(self) -> list[ThoughtNode]:
        """Get sibling nodes (share at least one parent).

        Returns:
            List of sibling nodes (excluding self)
        """
        siblings: set[ThoughtNode] = set()
        for parent in self.parents:
            for child in parent.children:
                if child.id != self.id:
                    siblings.add(child)
        return list(siblings)

    def get_depth(self) -> int:
        """Calculate the depth of this node in the graph.

        Returns:
            Depth level (0 for root nodes)
        """
        if not self.parents:
            return 0
        return max(parent.get_depth() for parent in self.parents) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary representation.

        Returns:
            Dictionary with node attributes
        """
        content_dict: dict[str, Any]
        if isinstance(self.content, ArchitectureDecision):
            content_dict = {
                "aspect": self.content.aspect,
                "choice": self.content.choice,
                "reasoning": self.content.reasoning,
                "confidence": self.content.confidence,
            }
        else:
            content_dict = {"value": str(self.content)}

        return {
            "id": self.id,
            "content": content_dict,
            "parent_ids": [p.id for p in self.parents],
            "child_ids": [c.id for c in self.children],
            "score": self.score,
            "valid": self.valid,
            "evidence": self.evidence,
            "generation": self.generation,
            "status": self.status.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_decision(
        cls,
        decision: ArchitectureDecision,
        generation: int = 0,
        node_id: str | None = None,
    ) -> ThoughtNode:
        """Create a ThoughtNode from an ArchitectureDecision.

        Args:
            decision: The architecture decision
            generation: Generation level
            node_id: Optional explicit node ID

        Returns:
            New ThoughtNode instance
        """
        return cls(
            id=node_id or f"decision_{int(time.time() * 1000)}",
            content=decision,
            generation=generation,
            status=ThoughtStatus.DRAFT,
        )


@dataclass
class GoTGraph:
    """Container for the complete Graph of Thoughts.

    Manages the graph structure, node access, and graph-wide operations.

    Attributes:
        nodes: Dictionary mapping node IDs to ThoughtNodes
        root: The root node of the graph
        selected: The currently selected best node
        max_nodes: Maximum number of nodes allowed
        edges: List of edges between nodes

    Example:
        >>> graph = GoTGraph(max_nodes=50)
        >>> graph.add_node(root_node)
        >>> best = graph.get_best_node()
    """

    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    root: ThoughtNode | None = None
    selected: ThoughtNode | None = None
    max_nodes: int = 50
    edges: list[ThoughtEdge] = field(default_factory=list)

    def add_node(self, node: ThoughtNode) -> bool:
        """Add a node to the graph.

        Args:
            node: The node to add

        Returns:
            True if added successfully, False if at capacity
        """
        if len(self.nodes) >= self.max_nodes:
            return False
        self.nodes[node.id] = node
        return True

    def get_node(self, node_id: str) -> ThoughtNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up

        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def add_edge(self, edge: ThoughtEdge) -> bool:
        """Add an edge to the graph.

        Args:
            edge: The edge to add

        Returns:
            True if added successfully
        """
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            self.edges.append(edge)
            # Establish node relationships
            source = self.nodes[edge.source_id]
            target = self.nodes[edge.target_id]
            if target not in source.children:
                source.children.append(target)
            if source not in target.parents:
                target.parents.append(source)
            return True
        return False

    def get_best_node(self) -> ThoughtNode | None:
        """Get the highest-scoring valid node.

        Returns:
            The best node, or None if no valid nodes exist
        """
        valid_nodes = [n for n in self.nodes.values() if n.valid and n.score > 0]
        if not valid_nodes:
            return None
        return max(valid_nodes, key=lambda n: n.score)

    def get_nodes_by_generation(self, generation: int) -> list[ThoughtNode]:
        """Get all nodes at a specific generation level.

        Args:
            generation: The generation number to filter by

        Returns:
            List of nodes at that generation
        """
        return [n for n in self.nodes.values() if n.generation == generation]

    def get_leaves(self) -> list[ThoughtNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf nodes
        """
        return [n for n in self.nodes.values() if not n.children]

    def prune_low_score_nodes(self, threshold: float = 0.3) -> int:
        """Remove nodes with scores below the threshold.

        Args:
            threshold: Minimum score to keep (default: 0.3)

        Returns:
            Number of nodes pruned
        """
        pruned = 0
        for node in self.nodes.values():
            if node.score < threshold and node.status != ThoughtStatus.SELECTED:
                node.valid = False
                node.status = ThoughtStatus.PRUNED
                pruned += 1
        return pruned

    def merge_nodes(
        self,
        nodes: list[ThoughtNode],
        merged_content: ArchitectureDecision | str,
        node_id: str | None = None,
    ) -> ThoughtNode:
        """Merge multiple nodes into a new aggregated node.

        Args:
            nodes: Nodes to merge
            merged_content: Content for the merged node
            node_id: Optional explicit node ID

        Returns:
            The new merged node
        """
        new_node = ThoughtNode(
            id=node_id or f"merged_{int(time.time() * 1000)}",
            content=merged_content,
            parents=nodes.copy(),
            generation=max(n.generation for n in nodes) + 1,
            status=ThoughtStatus.AGGREGATED,
            evidence=[f"Merged from: {', '.join(n.id for n in nodes)}"],
        )

        for node in nodes:
            node.add_child(new_node)

        self.add_node(new_node)
        return new_node

    def get_path_to_root(self, node: ThoughtNode) -> list[ThoughtNode]:
        """Get the path from a node back to the root.

        Args:
            node: Starting node

        Returns:
            List of nodes from node to root (inclusive)
        """
        path = [node]
        current = node

        while current.parents:
            # Take first parent for path construction
            current = current.parents[0]
            path.append(current)

        return path

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary with graph structure
        """
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "root_id": self.root.id if self.root else None,
            "selected_id": self.selected.id if self.selected else None,
            "max_nodes": self.max_nodes,
            "node_count": len(self.nodes),
        }


class GoTStrategic:
    """Graph of Thoughts Strategic Engine.

    Replaces ToT with a more flexible graph structure that supports:
    - Generating diverse solutions
    - Aggregating best parts from multiple nodes
    - Refining nodes based on feedback
    - Evidence-based scoring

    The engine builds a graph of thoughts through multiple generations,
    scoring and selecting the best architecture specification.

    Attributes:
        provider: LLM provider for generation
        max_nodes: Maximum nodes in the graph
        max_generations: Maximum generations to explore
        model: Model to use for LLM calls
        _graph: The current GoT graph being built

    Example:
        >>> got = GoTStrategic(provider=provider, max_nodes=50, max_generations=4)
        >>> spec = await got.explore(intent, context)
    """

    def __init__(
        self,
        provider: Any = None,
        max_nodes: int = 50,
        max_generations: int = 4,
        model: str | None = None,
    ):
        self.provider = provider
        self.max_nodes = max_nodes
        self.max_generations = max_generations
        self.model = model or "llama-3.3-70b-versatile"
        self._logger = logger
        self._graph: GoTGraph | None = None

    async def generate(
        self,
        intent: StructuredIntent,
        n: int = 3,
        context: dict[str, Any] | None = None,
    ) -> list[ThoughtNode]:
        """Generate n diverse solution candidates.

        Args:
            intent: The structured intent to address
            n: Number of solutions to generate
            context: Additional context for generation

        Returns:
            List of ThoughtNode candidates
        """
        if not self.provider:
            return self._generate_fallback(intent, n)

        prompt = self._build_generation_prompt(intent, context)

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=self._get_generation_system_prompt()),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=4096,
            )

            if not response.choices:
                return self._generate_fallback(intent, n)

            content = response.choices[0].message.content
            return self._parse_generation_response(content, intent, n)

        except Exception as e:
            self._logger.warning(f"LLM generation failed: {e}, using fallback")
            return self._generate_fallback(intent, n)

    async def aggregate(
        self,
        nodes: list[ThoughtNode],
        intent: StructuredIntent,
    ) -> ThoughtNode:
        """Aggregate multiple nodes into a new combined solution.

        Args:
            nodes: Nodes to aggregate
            intent: The original intent

        Returns:
            A new aggregated ThoughtNode
        """
        if not nodes:
            raise ValueError("Cannot aggregate empty node list")

        if len(nodes) == 1:
            return nodes[0]

        if not self.provider:
            return self._aggregate_best_parts(nodes)

        prompt = self._build_aggregation_prompt(nodes, intent)

        try:
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=self._get_aggregation_system_prompt(),
                ),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=2048,
            )

            if not response.choices:
                return self._aggregate_best_parts(nodes)

            content = response.choices[0].message.content
            return self._parse_aggregation_response(content, nodes)

        except Exception as e:
            self._logger.warning(f"LLM aggregation failed: {e}, using fallback")
            return self._aggregate_best_parts(nodes)

    async def refine(
        self,
        node: ThoughtNode,
        feedback: str,
        intent: StructuredIntent,
    ) -> ThoughtNode:
        """Refine a node based on feedback.

        Args:
            node: Node to refine
            feedback: Feedback for improvement
            intent: The original intent

        Returns:
            A new refined ThoughtNode
        """
        if not self.provider:
            return self._refine_fallback(node, feedback)

        prompt = self._build_refinement_prompt(node, feedback, intent)

        try:
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=self._get_refinement_system_prompt(),
                ),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.4,
                max_tokens=2048,
            )

            if not response.choices:
                return self._refine_fallback(node, feedback)

            content = response.choices[0].message.content
            return self._parse_refinement_response(content, node)

        except Exception as e:
            self._logger.warning(f"LLM refinement failed: {e}, using fallback")
            return self._refine_fallback(node, feedback)

    async def score_node(
        self,
        node: ThoughtNode,
        intent: StructuredIntent,
        criteria: list[str] | None = None,
    ) -> float:
        """Score a node with evidence.

        Args:
            node: Node to score
            intent: The original intent
            criteria: Optional scoring criteria

        Returns:
            Score between 0.0 and 1.0
        """
        if not self.provider:
            return self._score_fallback(node, intent)

        prompt = self._build_scoring_prompt(node, intent, criteria)

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=self._get_scoring_system_prompt()),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
            )

            if not response.choices:
                return self._score_fallback(node, intent)

            content = response.choices[0].message.content
            score, evidence = self._parse_scoring_response(content)
            node.evidence = evidence
            return score

        except Exception as e:
            self._logger.warning(f"LLM scoring failed: {e}, using fallback")
            return self._score_fallback(node, intent)

    async def explore(
        self,
        intent: StructuredIntent,
        context: dict[str, Any],
        wisdom: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
    ) -> ArchitectureSpec:
        """Main entry point - builds the graph and returns the best spec.

        This method builds a graph of thoughts through multiple generations,
        scoring nodes and aggregating the best solutions.

        Args:
            intent: The structured intent
            context: Execution context
            wisdom: Relevant wisdom/heuristics from memory
            pitfalls: Relevant pitfalls from failure store

        Returns:
            The best ArchitectureSpec found
        """
        start_time = time.time()

        self._graph = GoTGraph(max_nodes=self.max_nodes)

        root_content = ArchitectureDecision(
            aspect="root",
            choice=intent.explicit_goals[0] if intent.explicit_goals else "Design system",
            reasoning="Initial problem statement",
            trade_offs=[],
            confidence=1.0,
        )
        self._graph.root = ThoughtNode(
            id="root",
            content=root_content,
            generation=0,
            status=ThoughtStatus.DRAFT,
        )
        self._graph.add_node(self._graph.root)

        enhanced_context = context.copy()
        if wisdom:
            enhanced_context["wisdom"] = [
                w.principle for w in wisdom[:3] if hasattr(w, "principle")
            ]
        if pitfalls:
            enhanced_context["pitfalls"] = [
                f"{p[0].error}: {p[1][0].solution if p[1] else 'No solution'}" for p in pitfalls[:3]
            ]

        for generation in range(1, self.max_generations + 1):
            parent_nodes = self._graph.get_nodes_by_generation(generation - 1)
            parent_nodes = [n for n in parent_nodes if n.valid]

            if not parent_nodes:
                break

            candidates = await self.generate(intent, n=3, context=enhanced_context)

            for candidate in candidates:
                candidate.generation = generation
                parent = (
                    max(parent_nodes, key=lambda p: p.score) if parent_nodes else self._graph.root
                )
                parent.add_child(candidate)
                self._graph.add_node(candidate)

                candidate.score = await self.score_node(candidate, intent)
                candidate.status = ThoughtStatus.EVALUATED

            if generation >= 2:
                best_nodes = sorted(
                    [n for n in self._graph.get_nodes_by_generation(generation) if n.valid],
                    key=lambda n: n.score,
                    reverse=True,
                )[:2]

                if len(best_nodes) >= 2:
                    aggregated = await self.aggregate(best_nodes, intent)
                    aggregated.generation = generation
                    for bn in best_nodes:
                        bn.add_child(aggregated)
                    self._graph.add_node(aggregated)
                    aggregated.score = await self.score_node(aggregated, intent)
                    aggregated.status = ThoughtStatus.AGGREGATED

        best_node = self._graph.get_best_node()
        if best_node:
            best_node.status = ThoughtStatus.SELECTED
            self._graph.selected = best_node

        spec = self._node_to_spec(best_node, intent)

        elapsed = (time.time() - start_time) * 1000
        self._logger.info(
            f"GoT exploration complete: nodes={len(self._graph.nodes)}, "
            f"best_score={best_node.score if best_node else 0:.2f}, time={elapsed:.0f}ms"
        )

        return spec

    def _node_to_spec(self, node: ThoughtNode | None, intent: StructuredIntent) -> ArchitectureSpec:
        """Convert best node to ArchitectureSpec.

        Args:
            node: The selected best node
            intent: Original structured intent

        Returns:
            Complete ArchitectureSpec
        """
        spec = ArchitectureSpec(
            spec_id=f"spec_got_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
        )

        if not node:
            return spec

        spec.explored_paths = len(self._graph.nodes) if self._graph else 1
        spec.selected_path_score = node.score

        if isinstance(node.content, ArchitectureDecision):
            spec.decisions.append(node.content)

        content_str = str(node.content).lower() if node.content else ""

        for paradigm in ArchitectureParadigm:
            if paradigm.value in content_str:
                spec.paradigm = paradigm
                break

        for strategy in DataStrategy:
            if strategy.value in content_str:
                spec.data_strategy = strategy
                break

        for comm in CommunicationPattern:
            if comm.value in content_str:
                spec.communication = comm
                break

        if node.evidence:
            spec.metadata["evidence"] = node.evidence

        return spec

    def _generate_fallback(self, intent: StructuredIntent, n: int) -> list[ThoughtNode]:
        """Fallback generation without LLM."""
        paradigms = list(ArchitectureParadigm)[:n]
        nodes = []

        for i, paradigm in enumerate(paradigms):
            decision = ArchitectureDecision(
                aspect="paradigm",
                choice=paradigm.value,
                reasoning=f"Fallback option {i + 1}",
                trade_offs=["Complexity", "Cost"],
                confidence=0.6,
            )
            node = ThoughtNode(
                id=f"fallback_{i}",
                content=decision,
                generation=1,
            )
            nodes.append(node)

        return nodes

    def _aggregate_best_parts(self, nodes: list[ThoughtNode]) -> ThoughtNode:
        """Fallback aggregation - combine best scoring aspects."""
        best = max(nodes, key=lambda n: n.score)

        aggregated_decision = ArchitectureDecision(
            aspect="aggregated_paradigm",
            choice=(
                best.content.choice
                if isinstance(best.content, ArchitectureDecision)
                else str(best.content)
            ),
            reasoning=f"Aggregated from {len(nodes)} solutions",
            trade_offs=[],
            confidence=sum(n.score for n in nodes) / len(nodes),
        )

        aggregated = ThoughtNode(
            id=f"agg_{int(time.time() * 1000)}",
            content=aggregated_decision,
            parents=nodes.copy(),
            generation=max(n.generation for n in nodes) + 1,
            status=ThoughtStatus.AGGREGATED,
            evidence=[f"Combined from nodes: {', '.join(n.id for n in nodes)}"],
        )

        return aggregated

    def _refine_fallback(self, node: ThoughtNode, feedback: str) -> ThoughtNode:
        """Fallback refinement without LLM."""
        refined_content = node.content
        if isinstance(node.content, ArchitectureDecision):
            refined_content = ArchitectureDecision(
                aspect=node.content.aspect,
                choice=node.content.choice,
                reasoning=f"{node.content.reasoning} [Refined: {feedback[:50]}]",
                trade_offs=node.content.trade_offs,
                confidence=min(node.content.confidence + 0.05, 1.0),
            )

        return ThoughtNode(
            id=f"refined_{node.id}",
            content=refined_content,
            parents=[node],
            generation=node.generation + 1,
            status=ThoughtStatus.REFINED,
            metadata={"refinement_feedback": feedback},
        )

    def _score_fallback(self, node: ThoughtNode, intent: StructuredIntent) -> float:
        """Fallback scoring based on content matching."""
        base_score = 0.5

        if intent.explicit_goals:
            content_str = str(node.content).lower()
            for goal in intent.explicit_goals:
                if any(word in content_str for word in goal.lower().split()):
                    base_score += 0.1

        implicit = intent.implicit_requirements
        if implicit.scalability:
            content_str = str(node.content).lower()
            if "microservices" in content_str or "distributed" in content_str:
                base_score += 0.15

        return min(base_score, 1.0)

    def _get_generation_system_prompt(self) -> str:
        """System prompt for generation phase."""
        return """You are an expert software architect. Generate diverse architecture solutions.

For each solution, provide:
1. Architecture paradigm (monolith, modular_monolith, microservices, serverless, event_driven, layered, hexagonal)
2. Data strategy (single_database, polyglot, cqrs, event_sourcing)
3. Communication pattern (rest, graphql, grpc, message_queue, event_bus)
4. Reasoning for each choice
5. Trade-offs considered

Output ONLY valid JSON array:
[
  {
    "paradigm": "...",
    "data_strategy": "...",
    "communication": "...",
    "reasoning": "...",
    "trade_offs": ["...", "..."]
  }
]

Generate diverse solutions, not variations of the same idea."""

    def _build_generation_prompt(
        self,
        intent: StructuredIntent,
        context: dict[str, Any] | None,
    ) -> str:
        """Build prompt for generation phase."""
        wisdom_hints = ""
        if context and "wisdom" in context:
            wisdom_hints = f"\nRelevant wisdom:\n" + "\n".join(f"- {w}" for w in context["wisdom"])

        pitfalls_hints = ""
        if context and "pitfalls" in context:
            pitfalls_hints = f"\nPitfalls to avoid:\n" + "\n".join(
                f"- {p}" for p in context["pitfalls"]
            )

        return f"""Generate 3 diverse architecture solutions for:

Goals: {intent.explicit_goals}
Constraints: {intent.constraints}
Intent Type: {intent.intent_type.name}

Requirements:
- Performance: {intent.implicit_requirements.performance or "Not specified"}
- Security: {intent.implicit_requirements.security or "Standard"}
- Scalability: {intent.implicit_requirements.scalability or "Not required"}
- Budget: {intent.implicit_requirements.budget or "Not constrained"}
- Timeline: {intent.implicit_requirements.timeline or "Flexible"}
{wisdom_hints}
{pitfalls_hints}

Return ONLY a JSON array of 3 different architecture proposals."""

    def _parse_generation_response(
        self,
        content: str,
        intent: StructuredIntent,
        n: int,
    ) -> list[ThoughtNode]:
        """Parse LLM response into ThoughtNodes."""
        import re

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._generate_fallback(intent, n)
            else:
                return self._generate_fallback(intent, n)

        nodes = []
        for i, item in enumerate(data[:n]):
            decision = ArchitectureDecision(
                aspect=item.get("aspect", "architecture"),
                choice=item.get("paradigm", "modular_monolith"),
                reasoning=item.get("reasoning", "Generated solution"),
                trade_offs=item.get("trade_offs", []),
                confidence=0.7,
            )
            node = ThoughtNode(
                id=f"gen_{i}_{int(time.time() * 1000)}",
                content=decision,
                generation=1,
                metadata={
                    "data_strategy": item.get("data_strategy"),
                    "communication": item.get("communication"),
                },
            )
            nodes.append(node)

        return nodes if nodes else self._generate_fallback(intent, n)

    def _get_aggregation_system_prompt(self) -> str:
        """System prompt for aggregation phase."""
        return """You are an expert software architect. Combine the best parts of multiple solutions.

Analyze each solution, identify their strengths, and create a new solution that combines the best aspects.

Output ONLY valid JSON:
{
  "paradigm": "...",
  "data_strategy": "...",
  "communication": "...",
  "reasoning": "Combined best aspects: ...",
  "trade_offs": ["...", "..."],
  "combined_from": ["aspect from solution 1", "aspect from solution 2"]
}"""

    def _build_aggregation_prompt(
        self,
        nodes: list[ThoughtNode],
        intent: StructuredIntent,
    ) -> str:
        """Build prompt for aggregation phase."""
        solutions = []
        for i, node in enumerate(nodes):
            content = node.content
            if isinstance(content, ArchitectureDecision):
                solutions.append(f"""
Solution {i + 1} (score: {node.score:.2f}):
- Paradigm: {content.choice}
- Reasoning: {content.reasoning}
- Trade-offs: {content.trade_offs}
""")
            else:
                solutions.append(f"Solution {i + 1}: {content}")

        return f"""Combine these architecture solutions into one optimal solution:

{"".join(solutions)}

Original intent: {intent.explicit_goals}

Create a new solution that combines the best aspects of each."""

    def _parse_aggregation_response(
        self,
        content: str,
        nodes: list[ThoughtNode],
    ) -> ThoughtNode:
        """Parse aggregation response."""
        import re

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._aggregate_best_parts(nodes)
            else:
                return self._aggregate_best_parts(nodes)

        decision = ArchitectureDecision(
            aspect=data.get("aspect", "aggregated_architecture"),
            choice=data.get("paradigm", "modular_monolith"),
            reasoning=data.get("reasoning", "Aggregated solution"),
            trade_offs=data.get("trade_offs", []),
            confidence=sum(n.score for n in nodes) / len(nodes),
        )

        aggregated = ThoughtNode(
            id=f"agg_{int(time.time() * 1000)}",
            content=decision,
            parents=nodes.copy(),
            generation=max(n.generation for n in nodes) + 1,
            status=ThoughtStatus.AGGREGATED,
            evidence=data.get("combined_from", []),
        )

        return aggregated

    def _get_refinement_system_prompt(self) -> str:
        """System prompt for refinement phase."""
        return """You are an expert software architect. Refine an architecture solution based on feedback.

Improve the solution while maintaining its core strengths.

Output ONLY valid JSON:
{
  "paradigm": "...",
  "data_strategy": "...",
  "communication": "...",
  "reasoning": "Refined: ...",
  "trade_offs": ["...", "..."],
  "improvements": ["what was improved based on feedback"]
}"""

    def _build_refinement_prompt(
        self,
        node: ThoughtNode,
        feedback: str,
        intent: StructuredIntent,
    ) -> str:
        """Build prompt for refinement phase."""
        content = node.content
        if isinstance(content, ArchitectureDecision):
            solution_str = f"""
Current solution:
- Paradigm: {content.choice}
- Reasoning: {content.reasoning}
- Trade-offs: {content.trade_offs}
- Confidence: {content.confidence}
"""
        else:
            solution_str = f"Current solution: {content}"

        return f"""Refine this architecture solution based on feedback:

{solution_str}

Feedback: {feedback}

Original intent: {intent.explicit_goals}

Improve the solution while keeping what works."""

    def _parse_refinement_response(
        self,
        content: str,
        node: ThoughtNode,
    ) -> ThoughtNode:
        """Parse refinement response."""
        import re

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._refine_fallback(node, "LLM parse error")
            else:
                return self._refine_fallback(node, "LLM parse error")

        original = node.content
        if isinstance(original, ArchitectureDecision):
            decision = ArchitectureDecision(
                aspect=original.aspect,
                choice=data.get("paradigm", original.choice),
                reasoning=data.get("reasoning", original.reasoning),
                trade_offs=data.get("trade_offs", original.trade_offs),
                confidence=min(original.confidence + 0.05, 1.0),
            )
        else:
            decision = ArchitectureDecision(
                aspect="refined",
                choice=data.get("paradigm", "modular_monolith"),
                reasoning=data.get("reasoning", str(original)),
                trade_offs=data.get("trade_offs", []),
                confidence=0.7,
            )

        refined = ThoughtNode(
            id=f"refined_{node.id}",
            content=decision,
            parents=[node],
            generation=node.generation + 1,
            status=ThoughtStatus.REFINED,
            evidence=data.get("improvements", []),
        )

        return refined

    def _get_scoring_system_prompt(self) -> str:
        """System prompt for scoring phase."""
        return """You are an expert software architect evaluator. Score architecture solutions objectively.

Provide a score (0-100) and specific evidence for your evaluation.

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "specific evidence point 1",
    "specific evidence point 2",
    "specific evidence point 3"
  ],
  "reasoning": "brief explanation"
}"""

    def _build_scoring_prompt(
        self,
        node: ThoughtNode,
        intent: StructuredIntent,
        criteria: list[str] | None,
    ) -> str:
        """Build prompt for scoring phase."""
        content = node.content
        if isinstance(content, ArchitectureDecision):
            solution_str = f"""
Solution to evaluate:
- Aspect: {content.aspect}
- Choice: {content.choice}
- Reasoning: {content.reasoning}
- Trade-offs: {content.trade_offs}
"""
        else:
            solution_str = f"Solution to evaluate: {content}"

        criteria_str = ""
        if criteria:
            criteria_str = f"\nScoring criteria:\n" + "\n".join(f"- {c}" for c in criteria)

        return f"""Evaluate this architecture solution:

{solution_str}

Original goals: {intent.explicit_goals}
Constraints: {intent.constraints}
Requirements:
- Performance: {intent.implicit_requirements.performance or "Not specified"}
- Security: {intent.implicit_requirements.security or "Standard"}
- Scalability: {intent.implicit_requirements.scalability or "Not required"}
{criteria_str}

Score from 0-100 and provide at least 3 specific evidence points."""

    def _parse_scoring_response(self, content: str) -> tuple[float, list[str]]:
        """Parse scoring response."""
        import re

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return 0.5, ["LLM parse error - using default score"]
            else:
                return 0.5, ["LLM parse error - using default score"]

        score = data.get("score", 50) / 100.0
        evidence = data.get("evidence", [data.get("reasoning", "No evidence provided")])

        return score, evidence


def create_got_strategic(
    provider: Any = None,
    max_nodes: int = 50,
    max_generations: int = 4,
) -> GoTStrategic:
    """Create a GoTStrategic instance with the specified configuration.

    Args:
        provider: LLM provider for generation
        max_nodes: Maximum nodes in the graph (default: 50)
        max_generations: Maximum generations to explore (default: 4)

    Returns:
        Configured GoTStrategic instance

    Example:
        >>> got = create_got_strategic(provider=provider, max_nodes=50)
        >>> spec = await got.explore(intent, context)
    """
    return GoTStrategic(
        provider=provider,
        max_nodes=max_nodes,
        max_generations=max_generations,
    )
