"""
Contrastive Reasoning for GAAP System

Provides structured contrastive analysis for complex decision-making.

Classes:
    - ContrastivePath: A single reasoning path
    - ContrastiveResult: Result of contrastive reasoning
    - ContrastiveReasoner: Contrastive reasoning engine

Usage:
    from gaap.core.contrastive import ContrastiveReasoner

    reasoner = ContrastiveReasoner()
    result = reasoner.reason_about("Should we use microservices or monolith?")
    print(result.final_decision)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from gaap.core.logging import get_standard_logger as get_logger


@dataclass
class ContrastivePath:
    """
    A single reasoning path in contrastive analysis.

    Attributes:
        name: Name/label for this path
        reasoning: The reasoning behind this path
        pros: Advantages of this path
        cons: Disadvantages of this path
        risks: Risks associated with this path
        estimated_cost: Estimated cost/effort (relative scale 0-1)
        confidence: Confidence in this analysis (0-1)
        dependencies: Dependencies for this path
        timeline: Estimated timeline
    """

    name: str
    reasoning: str = ""
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    estimated_cost: float = 0.5
    confidence: float = 0.5
    dependencies: list[str] = field(default_factory=list)
    timeline: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reasoning": self.reasoning,
            "pros": self.pros,
            "cons": self.cons,
            "risks": self.risks,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "timeline": self.timeline,
        }

    def score(self) -> float:
        """Calculate overall score for this path."""
        pro_score = len(self.pros) * 0.1
        con_score = len(self.cons) * 0.05
        risk_score = len(self.risks) * 0.08
        cost_penalty = self.estimated_cost * 0.2

        base = 0.5 + pro_score - con_score - risk_score - cost_penalty
        return max(0.0, min(1.0, base * self.confidence))


@dataclass
class ContrastiveResult:
    """
    Result of contrastive reasoning process.

    Attributes:
        path_a: First reasoning path
        path_b: Second reasoning path
        synthesis: Synthesized insights from comparing paths
        final_decision: The final decision/recommendation
        decision_rationale: Rationale for the final decision
        confidence: Overall confidence in the result
        alternatives: Alternative options considered
    """

    path_a: ContrastivePath
    path_b: ContrastivePath
    synthesis: str = ""
    final_decision: str = ""
    decision_rationale: str = ""
    confidence: float = 0.5
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_a": self.path_a.to_dict(),
            "path_b": self.path_b.to_dict(),
            "synthesis": self.synthesis,
            "final_decision": self.final_decision,
            "decision_rationale": self.decision_rationale,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }

    def get_winning_path(self) -> ContrastivePath:
        """Determine the winning path based on scores."""
        score_a = self.path_a.score()
        score_b = self.path_b.score()
        return self.path_a if score_a >= score_b else self.path_b


class ContrastiveReasoner:
    """
    Contrastive reasoning engine for complex decision-making.

    Generates and compares alternative reasoning paths to reach
    well-considered decisions.

    Attributes:
        min_confidence: Minimum confidence threshold
        max_paths: Maximum number of alternative paths
    """

    ARCHITECTURE_PATHS: dict[str, tuple[ContrastivePath, ContrastivePath]] = {
        "monolith_vs_microservices": (
            ContrastivePath(
                name="Monolith",
                reasoning="Single unified application with shared codebase",
                pros=[
                    "Simpler deployment and operations",
                    "Easier debugging and tracing",
                    "Lower initial complexity",
                    "Better performance for small scale",
                    "Shared code reduces duplication",
                ],
                cons=[
                    "Limited horizontal scaling",
                    "Single point of failure",
                    "Technology lock-in",
                    "Larger deployment units",
                ],
                risks=[
                    "May need major refactor for scale",
                    "Team coordination overhead as codebase grows",
                ],
                estimated_cost=0.3,
                timeline="weeks",
            ),
            ContrastivePath(
                name="Microservices",
                reasoning="Distributed services with independent deployment",
                pros=[
                    "Independent scaling",
                    "Technology flexibility",
                    "Team autonomy",
                    "Fault isolation",
                    "Independent deployment",
                ],
                cons=[
                    "Increased operational complexity",
                    "Network latency",
                    "Distributed system challenges",
                    "Higher initial cost",
                ],
                risks=[
                    "Data consistency challenges",
                    "Service mesh complexity",
                    "Monitoring overhead",
                    "Team skill requirements",
                ],
                estimated_cost=0.7,
                timeline="months",
            ),
        ),
        "sql_vs_nosql": (
            ContrastivePath(
                name="SQL Database",
                reasoning="Relational database with ACID guarantees",
                pros=[
                    "ACID transactions",
                    "Mature ecosystem",
                    "Clear data model",
                    "Strong consistency",
                    "Rich query capabilities",
                ],
                cons=[
                    "Horizontal scaling limitations",
                    "Schema rigidity",
                    "Potential performance bottleneck",
                ],
                risks=[
                    "May need sharding for scale",
                    "Schema migrations can be complex",
                ],
                estimated_cost=0.3,
                timeline="days",
            ),
            ContrastivePath(
                name="NoSQL Database",
                reasoning="Document/key-value store with flexible schema",
                pros=[
                    "Horizontal scalability",
                    "Flexible schema",
                    "High write throughput",
                    "Good for unstructured data",
                ],
                cons=[
                    "Weaker consistency guarantees",
                    "Limited query capabilities",
                    "Less mature tooling",
                ],
                risks=[
                    "Data consistency issues",
                    "Vendor lock-in",
                    "Learning curve",
                ],
                estimated_cost=0.4,
                timeline="days",
            ),
        ),
        "build_vs_buy": (
            ContrastivePath(
                name="Build In-House",
                reasoning="Develop custom solution internally",
                pros=[
                    "Full control over features",
                    "No licensing costs",
                    "Tailored to exact needs",
                    "Team learns and grows",
                ],
                cons=[
                    "Time to market",
                    "Maintenance burden",
                    "Opportunity cost",
                    "Requires expertise",
                ],
                risks=[
                    "Scope creep",
                    "Quality issues without review",
                    "Key person dependency",
                ],
                estimated_cost=0.8,
                timeline="months",
            ),
            ContrastivePath(
                name="Buy/Use Existing",
                reasoning="Use existing solution or service",
                pros=[
                    "Faster time to market",
                    "Proven solution",
                    "Vendor support",
                    "Lower initial cost",
                ],
                cons=[
                    "Licensing costs",
                    "Limited customization",
                    "Vendor dependency",
                    "May not fit perfectly",
                ],
                risks=[
                    "Vendor lock-in",
                    "Service discontinuation",
                    "Hidden costs",
                ],
                estimated_cost=0.4,
                timeline="days",
            ),
        ),
    }

    DECISION_KEYWORDS: dict[str, list[str]] = {
        "architecture": ["architecture", "design", "structure", "pattern", "system"],
        "database": ["database", "storage", "persistence", "data store", "db"],
        "build_buy": ["build", "buy", "make", "implement", "use", "adopt"],
        "language": ["language", "framework", "stack", "technology"],
        "deployment": ["deploy", "cloud", "host", "infrastructure", "kubernetes"],
    }

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_paths: int = 3,
        provider: Any = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_paths = max_paths
        self._provider = provider
        self._logger = get_logger("gaap.core.contrastive")
        self._reasoning_count = 0

    def generate_paths(
        self,
        decision_context: str,
    ) -> tuple[ContrastivePath, ContrastivePath]:
        """
        Generate two contrastive paths for a decision.

        Args:
            decision_context: Context describing the decision

        Returns:
            Tuple of two contrastive paths
        """
        context_lower = decision_context.lower()

        for key, paths in self.ARCHITECTURE_PATHS.items():
            keywords = self.DECISION_KEYWORDS.get(key, [])
            if any(kw in context_lower for kw in keywords):
                self._logger.debug(f"Matched decision type: {key}")
                return paths

        return self._generate_generic_paths(decision_context)

    def synthesize(
        self,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
    ) -> str:
        """
        Synthesize insights from comparing two paths.

        Args:
            path_a: First path
            path_b: Second path

        Returns:
            Synthesis string
        """
        synthesis_parts = []

        synthesis_parts.append(f"Comparing {path_a.name} vs {path_b.name}:")

        unique_pros_a = [p for p in path_a.pros if p not in path_b.pros]
        unique_pros_b = [p for p in path_b.pros if p not in path_a.pros]

        if unique_pros_a:
            synthesis_parts.append(
                f"  {path_a.name} unique advantages: {', '.join(unique_pros_a[:3])}"
            )
        if unique_pros_b:
            synthesis_parts.append(
                f"  {path_b.name} unique advantages: {', '.join(unique_pros_b[:3])}"
            )

        shared_risks = set(path_a.risks) & set(path_b.risks)
        if shared_risks:
            synthesis_parts.append(f"  Shared risks: {', '.join(list(shared_risks)[:2])}")

        score_a = path_a.score()
        score_b = path_b.score()

        if abs(score_a - score_b) < 0.1:
            synthesis_parts.append(
                "  Both paths have similar scores; context-specific factors should decide."
            )
        else:
            winner = path_a if score_a > score_b else path_b
            synthesis_parts.append(
                f"  {winner.name} scores higher ({max(score_a, score_b):.2f} vs {min(score_a, score_b):.2f})"
            )

        return "\n".join(synthesis_parts)

    def reason_about(
        self,
        decision: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ContrastiveResult:
        """
        Perform contrastive reasoning on a decision.

        Args:
            decision: The decision to reason about
            context: Additional context for reasoning

        Returns:
            ContrastiveResult with paths and final decision
        """
        self._reasoning_count += 1
        self._logger.debug(f"Reasoning about: {decision[:50]}...")

        path_a, path_b = self.generate_paths(decision)

        if context:
            path_a = self._enrich_path(path_a, context)
            path_b = self._enrich_path(path_b, context)

        synthesis = self.synthesize(path_a, path_b)

        final_decision = self._make_decision(path_a, path_b, synthesis)
        decision_rationale = self._generate_rationale(path_a, path_b, final_decision)

        confidence = self._calculate_confidence(path_a, path_b)

        result = ContrastiveResult(
            path_a=path_a,
            path_b=path_b,
            synthesis=synthesis,
            final_decision=final_decision,
            decision_rationale=decision_rationale,
            confidence=confidence,
            alternatives=[path_a.name, path_b.name],
        )

        self._logger.info(
            f"Contrastive reasoning complete: {final_decision[:30]}... "
            f"(confidence: {confidence:.2f})"
        )

        return result

    def _generate_generic_paths(
        self,
        decision_context: str,
    ) -> tuple[ContrastivePath, ContrastivePath]:
        """Generate generic conservative vs aggressive paths."""
        path_a = ContrastivePath(
            name="Conservative Approach",
            reasoning="Take a careful, incremental approach minimizing risk",
            pros=[
                "Lower risk of failure",
                "Easier to reverse",
                "More predictable outcomes",
                "Lower initial investment",
            ],
            cons=[
                "Slower progress",
                "May miss opportunities",
                "Could become obsolete",
            ],
            risks=[
                "Competitor advantage",
                "Technology drift",
            ],
            estimated_cost=0.3,
            timeline="short",
        )

        path_b = ContrastivePath(
            name="Aggressive Approach",
            reasoning="Take bold action for maximum potential gain",
            pros=[
                "Faster time to value",
                "Competitive advantage",
                "Full commitment",
                "Modern approach",
            ],
            cons=[
                "Higher risk",
                "More resources required",
                "Harder to reverse",
            ],
            risks=[
                "Implementation challenges",
                "Budget overrun",
                "Team skill gaps",
            ],
            estimated_cost=0.7,
            timeline="medium",
        )

        return path_a, path_b

    def _enrich_path(
        self,
        path: ContrastivePath,
        context: dict[str, Any],
    ) -> ContrastivePath:
        """Enrich a path with additional context."""
        budget = context.get("budget")
        if budget:
            if budget == "budget_conscious":
                path.estimated_cost *= 1.2
            elif budget == "unlimited":
                path.estimated_cost *= 0.8

        timeline = context.get("timeline")
        if timeline:
            if timeline == "urgent":
                path.timeline = "critical"
            elif timeline == "relaxed":
                path.timeline = "flexible"

        return path

    def _make_decision(
        self,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
        synthesis: str,
    ) -> str:
        """Make a final decision based on paths and synthesis."""
        score_a = path_a.score()
        score_b = path_b.score()

        if score_a > score_b + 0.15:
            return f"Recommend {path_a.name}: {path_a.reasoning}"
        elif score_b > score_a + 0.15:
            return f"Recommend {path_b.name}: {path_b.reasoning}"
        else:
            return f"Context-dependent: Either {path_a.name} or {path_b.name} could work depending on specific requirements"

    def _generate_rationale(
        self,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
        decision: str,
    ) -> str:
        """Generate rationale for the decision."""
        winner = self._get_winner(path_a, path_b)

        rationale_parts = [
            f"Decision: {decision}",
            f"",
            f"Key factors:",
        ]

        for pro in winner.pros[:3]:
            rationale_parts.append(f"  + {pro}")

        loser = path_b if winner == path_a else path_a
        for con in loser.cons[:2]:
            rationale_parts.append(f"  - Alternative {loser.name}: {con}")

        rationale_parts.append(f"")
        rationale_parts.append(
            f"Primary risk: {winner.risks[0] if winner.risks else 'None identified'}"
        )

        return "\n".join(rationale_parts)

    def _get_winner(
        self,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
    ) -> ContrastivePath:
        """Get the winning path."""
        return path_a if path_a.score() >= path_b.score() else path_b

    def _calculate_confidence(
        self,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
    ) -> float:
        """Calculate overall confidence in the result."""
        score_diff = abs(path_a.score() - path_b.score())

        path_confidence = (path_a.confidence + path_b.confidence) / 2

        separation_bonus = score_diff * 0.5

        confidence = path_confidence + separation_bonus
        return max(0.0, min(1.0, confidence))

    def get_statistics(self) -> dict[str, Any]:
        """Get reasoning statistics."""
        return {
            "reasoning_count": self._reasoning_count,
            "min_confidence": self.min_confidence,
            "max_paths": self.max_paths,
        }

    def add_custom_paths(
        self,
        key: str,
        path_a: ContrastivePath,
        path_b: ContrastivePath,
    ) -> None:
        """Add custom paths for a decision type."""
        self.ARCHITECTURE_PATHS[key] = (path_a, path_b)
        self._logger.info(f"Added custom paths for: {key}")
