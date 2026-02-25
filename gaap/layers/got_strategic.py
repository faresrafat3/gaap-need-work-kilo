"""
Graph of Thoughts (GoT) Strategic Engine
========================================

A graph-based reasoning engine that replaces Tree of Thoughts (ToT).
Unlike ToT's linear tree structure, GoT allows:
- Multiple parents per node (for aggregation/merging)
- Cross-level connections
- Cycles for iterative refinement

Key Components:
    - ThoughtNode: Node with multiple parents/children
    - GoTStrategic: Main engine for graph-based exploration

Usage:
    from gaap.layers.got_strategic import GoTStrategic, ThoughtNode

    got = GoTStrategic(provider=provider)
    spec = await got.explore(intent, context)
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer0_interface import StructuredIntent
from gaap.layers.layer1_strategic import (
    ArchitectureDecision,
    ArchitectureParadigm,
    ArchitectureSpec,
    CommunicationPattern,
    DataStrategy,
)

logger = get_logger("gaap.layer1.got")


class ThoughtStatus(Enum):
    """Status of a thought node"""

    DRAFT = auto()
    EVALUATED = auto()
    AGGREGATED = auto()
    REFINED = auto()
    PRUNED = auto()
    SELECTED = auto()


@dataclass
class ThoughtNode:
    """
    A node in the Graph of Thoughts.

    Unlike ToT's tree nodes, GoT nodes can have multiple parents,
    enabling aggregation of multiple solutions.

    Attributes:
        id: Unique identifier
        content: The architecture decision or reasoning
        parents: List of parent nodes (for aggregation)
        children: List of child nodes
        score: Evaluation score (0.0-1.0)
        valid: Whether this node is still valid
        evidence: List of evidence supporting this thought
        generation: Which generation/level this node belongs to
        status: Current status of the node
        metadata: Additional metadata
    """

    id: str
    content: ArchitectureDecision | str
    parents: list["ThoughtNode"] = field(default_factory=list)
    children: list["ThoughtNode"] = field(default_factory=list)
    score: float = 0.0
    valid: bool = True
    evidence: list[str] = field(default_factory=list)
    generation: int = 0
    status: ThoughtStatus = ThoughtStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_id(self) -> str:
        content_str = str(self.content)[:100]
        return hashlib.md5(f"{self.id}:{content_str}:{self.generation}".encode()).hexdigest()[:12]

    def add_child(self, child: "ThoughtNode") -> None:
        self.children.append(child)
        child.parents.append(self)

    def get_ancestors(self) -> list["ThoughtNode"]:
        ancestors: list["ThoughtNode"] = []
        visited: set[str] = set()

        def traverse(node: "ThoughtNode") -> None:
            for parent in node.parents:
                if parent.id not in visited:
                    visited.add(parent.id)
                    ancestors.append(parent)
                    traverse(parent)

        traverse(self)
        return ancestors

    def get_descendants(self) -> list["ThoughtNode"]:
        descendants: list["ThoughtNode"] = []
        visited: set[str] = set()

        def traverse(node: "ThoughtNode") -> None:
            for child in node.children:
                if child.id not in visited:
                    visited.add(child.id)
                    descendants.append(child)
                    traverse(child)

        traverse(self)
        return descendants

    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class GoTGraph:
    """Container for the complete Graph of Thoughts"""

    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    root: ThoughtNode | None = None
    selected: ThoughtNode | None = None
    max_nodes: int = 50

    def add_node(self, node: ThoughtNode) -> bool:
        if len(self.nodes) >= self.max_nodes:
            return False
        self.nodes[node.id] = node
        return True

    def get_best_node(self) -> ThoughtNode | None:
        valid_nodes = [n for n in self.nodes.values() if n.valid and n.score > 0]
        if not valid_nodes:
            return None
        return max(valid_nodes, key=lambda n: n.score)

    def get_nodes_by_generation(self, generation: int) -> list[ThoughtNode]:
        return [n for n in self.nodes.values() if n.generation == generation]

    def prune_low_score_nodes(self, threshold: float = 0.3) -> int:
        pruned = 0
        for node in self.nodes.values():
            if node.score < threshold and node.status != ThoughtStatus.SELECTED:
                node.valid = False
                node.status = ThoughtStatus.PRUNED
                pruned += 1
        return pruned


class GoTStrategic:
    """
    Graph of Thoughts Strategic Engine.

    Replaces ToT with a more flexible graph structure that supports:
    - Generating diverse solutions
    - Aggregating best parts from multiple nodes
    - Refining nodes based on feedback
    - Evidence-based scoring

    Attributes:
        provider: LLM provider for generation
        max_nodes: Maximum nodes in the graph
        max_generations: Maximum generations to explore
        model: Model to use for LLM calls
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
        """
        Generate n diverse solution candidates using LLM.

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
        """
        Aggregate multiple nodes into a new combined solution using LLM.

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
                Message(role=MessageRole.SYSTEM, content=self._get_aggregation_system_prompt()),
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
        """
        Refine a node based on feedback using LLM.

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
                Message(role=MessageRole.SYSTEM, content=self._get_refinement_system_prompt()),
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
        """
        Score a node with evidence using LLM.

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
        """
        Main entry point - builds the graph and returns the best spec.

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
        """Convert best node to ArchitectureSpec"""
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
        """Fallback generation without LLM"""
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
        """Fallback aggregation - combine best scoring aspects"""
        best = max(nodes, key=lambda n: n.score)

        aggregated_decision = ArchitectureDecision(
            aspect="aggregated_paradigm",
            choice=best.content.choice
            if isinstance(best.content, ArchitectureDecision)
            else str(best.content),
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
        """Fallback refinement"""
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
        """Fallback scoring based on content matching"""
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
        """Parse LLM response into ThoughtNodes"""
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
        """Parse aggregation response"""
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
        """Parse refinement response"""
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
        """Parse scoring response"""
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
    """Create a GoTStrategic instance."""
    return GoTStrategic(
        provider=provider,
        max_nodes=max_nodes,
        max_generations=max_generations,
    )
