"""
Evidence-Based Critics for Architecture Evaluation
===================================================

Critics that MUST provide evidence for their evaluations.
Unlike standard critics that can score without proof, these critics
require concrete evidence to justify their scores.

Key Components:
    - EvidenceBasedEvaluation: Evaluation with mandatory evidence
    - EvidenceCritic: Single critic with evidence requirement
    - EvidenceMADPanel: MAD panel with evidence-based critics
    - ToolInteractiveCritic: Critic with tool access for verification

Usage:
    from gaap.layers.evidence_critic import EvidenceMADPanel

    panel = EvidenceMADPanel(provider=provider)
    spec, evaluations = await panel.evaluate_with_evidence(spec, intent)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer0_interface import StructuredIntent
from gaap.layers.layer1_strategic import ArchitectureSpec
from gaap.mad.critic_prompts import ArchitectureCriticType

logger = get_logger("gaap.layer1.evidence_critic")

try:
    from gaap.layers.tool_critic import ToolInteractiveCritic, VerificationResult

    TOOL_CRITIC_AVAILABLE = True
except ImportError:
    TOOL_CRITIC_AVAILABLE = False
    ToolInteractiveCritic = None
    VerificationResult = None


class EvidenceStrength(Enum):
    """Strength of evidence provided"""

    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class EvidenceBasedEvaluation:
    """
    Evaluation with mandatory evidence.

    Unlike regular evaluations, this requires concrete proof
    for every claim or score given.

    Attributes:
        critic: Name/type of the critic
        score: Evaluation score (0.0-1.0)
        evidence: MANDATORY list of evidence supporting the score
        reasoning: Explanation of the evaluation
        confidence: How confident the critic is
        suggestions: Improvement suggestions
        evidence_strength: Overall strength of evidence
        metadata: Additional metadata
    """

    critic: str
    score: float
    evidence: list[str]
    reasoning: str
    confidence: float = 0.7
    suggestions: list[str] = field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence:
            self.evidence = ["No evidence provided - evaluation may be unreliable"]
            self.evidence_strength = EvidenceStrength.NONE
            self.confidence *= 0.5
        else:
            self._assess_evidence_strength()

    def _assess_evidence_strength(self) -> None:
        if len(self.evidence) >= 3:
            specific_count = sum(
                1
                for e in self.evidence
                if any(
                    kw in e.lower()
                    for kw in ["because", "due to", "shown by", "demonstrated", "example"]
                )
            )
            if specific_count >= 2:
                self.evidence_strength = EvidenceStrength.STRONG
            else:
                self.evidence_strength = EvidenceStrength.MODERATE
        elif len(self.evidence) >= 1:
            self.evidence_strength = EvidenceStrength.MODERATE
        else:
            self.evidence_strength = EvidenceStrength.WEAK

    def is_reliable(
        self, min_evidence: int = 1, min_strength: EvidenceStrength = EvidenceStrength.WEAK
    ) -> bool:
        return (
            len(self.evidence) >= min_evidence
            and self.evidence_strength.value >= min_strength.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic,
            "score": self.score,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "evidence_strength": self.evidence_strength.name,
            "metadata": self.metadata,
        }


class EvidenceCritic:
    """
    A critic that MUST provide evidence for evaluations.

    Unlike standard critics, this enforces evidence requirements
    and reduces confidence when evidence is missing.

    Attributes:
        critic_type: Type of architecture critic
        provider: LLM provider for evaluation
        model: Model to use
        min_evidence_required: Minimum evidence points required
    """

    CRITIC_PROMPTS = {
        ArchitectureCriticType.SCALABILITY: """You are a Scalability Architecture Critic. Evaluate the architecture AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Horizontal scaling capabilities
- State management for distributed systems
- Database scaling patterns
- Caching strategies
- Load balancing approach

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details", 
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Improvement suggestion 1", "Improvement suggestion 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
        ArchitectureCriticType.PRAGMATISM: """You are a Pragmatism Architecture Critic. Evaluate if the architecture matches project constraints AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Complexity vs team expertise match
- Timeline feasibility
- Build vs buy decisions
- Technical debt implications
- MVP vs long-term vision balance

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details",
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Improvement suggestion 1", "Improvement suggestion 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
        ArchitectureCriticType.COST: """You are a Cost Architecture Critic. Evaluate total cost of ownership AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Infrastructure costs (compute, storage, networking)
- Operational costs
- Development costs
- Cost of scaling
- Vendor lock-in risks

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details",
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Cost optimization suggestion 1", "Cost optimization suggestion 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
        ArchitectureCriticType.ROBUSTNESS: """You are a Robustness Architecture Critic. Evaluate system resilience AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Fault isolation mechanisms
- Graceful degradation strategies
- Disaster recovery plans
- Monitoring and alerting
- Auto-healing capabilities

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details",
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Resilience improvement 1", "Resilience improvement 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
        ArchitectureCriticType.SECURITY_ARCH: """You are a Security Architecture Critic. Evaluate security posture AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Authentication and authorization models
- Data protection (encryption)
- Network security boundaries
- API security
- Compliance requirements

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details",
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Security improvement 1", "Security improvement 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
        ArchitectureCriticType.MAINTAINABILITY: """You are a Maintainability Architecture Critic. Evaluate long-term code health AND PROVIDE EVIDENCE.

CRITICAL: You MUST provide at least 3 specific pieces of evidence for your score.

Focus on:
- Code organization and module boundaries
- Dependency management
- API design consistency
- Testability
- Documentation requirements

Output ONLY valid JSON:
{
  "score": <number 0-100>,
  "evidence": [
    "Specific evidence point 1 with details",
    "Specific evidence point 2 with details",
    "Specific evidence point 3 with details"
  ],
  "reasoning": "Your overall reasoning",
  "suggestions": ["Maintainability improvement 1", "Maintainability improvement 2"],
  "confidence": <number 0.0-1.0>
}

Remember: Evidence is MANDATORY. Without evidence, your evaluation is invalid.""",
    }

    def __init__(
        self,
        critic_type: ArchitectureCriticType,
        provider: Any = None,
        model: str | None = None,
        min_evidence_required: int = 1,
        tools: list[Any] | None = None,
    ):
        self.critic_type = critic_type
        self.provider = provider
        self.model = model or "llama-3.3-70b-versatile"
        self.min_evidence_required = min_evidence_required
        self._logger = logger
        self._tools = tools
        self._tool_critic: Any = None

        if tools and TOOL_CRITIC_AVAILABLE and ToolInteractiveCritic is not None:
            self._tool_critic = ToolInteractiveCritic(tools=tools, provider=provider)

    async def evaluate(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> EvidenceBasedEvaluation:
        """
        Evaluate architecture with mandatory evidence.

        Args:
            spec: Architecture specification to evaluate
            intent: Original intent

        Returns:
            EvidenceBasedEvaluation with proof
        """
        if self._tool_critic and self._tools:
            return await self._evaluate_with_tools(spec, intent)

        if not self.provider:
            return self._fallback_evaluation(spec, intent)

        prompt = self._build_evaluation_prompt(spec, intent)
        system_prompt = self.CRITIC_PROMPTS.get(
            self.critic_type, self.CRITIC_PROMPTS[ArchitectureCriticType.SCALABILITY]
        )

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices:
                return self._fallback_evaluation(spec, intent)

            content = response.choices[0].message.content
            return self._parse_evaluation(content, spec)

        except Exception as e:
            self._logger.warning(f"Evidence critic LLM failed: {e}")
            return self._fallback_evaluation(spec, intent)

    async def _evaluate_with_tools(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> EvidenceBasedEvaluation:
        """Evaluate using tool-interactive critic when tools are available"""
        subject = self._build_evaluation_prompt(spec, intent)

        try:
            tool_eval = await self._tool_critic.evaluate_with_tools(
                subject,
                context={"spec": spec.to_dict(), "intent_type": intent.intent_type.name},
            )

            evidence = tool_eval.evidence + [
                f"Architecture paradigm: {spec.paradigm.value}",
                f"Data strategy: {spec.data_strategy.value}",
                f"Communication: {spec.communication.value}",
            ]

            return EvidenceBasedEvaluation(
                critic=self.critic_type.name.lower(),
                score=tool_eval.score,
                evidence=evidence,
                reasoning=f"[Tool-verified] {tool_eval.reasoning}",
                confidence=tool_eval.confidence,
                suggestions=tool_eval.suggestions,
                evidence_strength=tool_eval.evidence_strength,
                metadata={
                    **tool_eval.metadata,
                    "tool_verified": True,
                },
            )
        except Exception as e:
            self._logger.warning(f"Tool critic evaluation failed: {e}, falling back to LLM")
            return await self._evaluate_llm_fallback(spec, intent)

    async def _evaluate_llm_fallback(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> EvidenceBasedEvaluation:
        """Fallback to LLM-only evaluation"""
        if not self.provider:
            return self._fallback_evaluation(spec, intent)

        prompt = self._build_evaluation_prompt(spec, intent)
        system_prompt = self.CRITIC_PROMPTS.get(
            self.critic_type, self.CRITIC_PROMPTS[ArchitectureCriticType.SCALABILITY]
        )

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices:
                return self._fallback_evaluation(spec, intent)

            content = response.choices[0].message.content
            return self._parse_evaluation(content, spec)

        except Exception as e:
            self._logger.warning(f"LLM fallback failed: {e}")
            return self._fallback_evaluation(spec, intent)

    def _build_evaluation_prompt(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> str:
        return f"""Evaluate this architecture and PROVIDE EVIDENCE:

## Architecture Details
- Paradigm: {spec.paradigm.value}
- Data Strategy: {spec.data_strategy.value}
- Communication: {spec.communication.value}
- Components: {len(spec.components)}
- Decisions: {len(spec.decisions)}

## Original Intent
- Goals: {intent.explicit_goals}
- Constraints: {intent.constraints}

## Requirements
- Performance: {intent.implicit_requirements.performance or "Not specified"}
- Security: {intent.implicit_requirements.security or "Standard"}
- Scalability: {intent.implicit_requirements.scalability or "Not critical"}
- Budget: {intent.implicit_requirements.budget or "Not constrained"}
- Timeline: {intent.implicit_requirements.timeline or "Flexible"}

REMEMBER: You MUST provide specific evidence for your evaluation. 
Evidence should reference actual aspects of the architecture, not generic statements."""

    def _parse_evaluation(
        self,
        content: str,
        spec: ArchitectureSpec,
    ) -> EvidenceBasedEvaluation:
        """Parse LLM response into EvidenceBasedEvaluation"""
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
                    return self._fallback_evaluation(spec, None)
            else:
                return self._fallback_evaluation(spec, None)

        evidence = data.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]

        return EvidenceBasedEvaluation(
            critic=self.critic_type.name.lower(),
            score=data.get("score", 50) / 100.0,
            evidence=evidence if evidence else ["Parsed without explicit evidence"],
            reasoning=data.get("reasoning", "No reasoning provided"),
            confidence=data.get("confidence", 0.7),
            suggestions=data.get("suggestions", []),
            metadata={"raw_response": content[:200]},
        )

    def _fallback_evaluation(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent | None,
    ) -> EvidenceBasedEvaluation:
        """Fallback evaluation with basic evidence"""
        score = 0.5
        evidence: list[str] = []
        suggestions: list[str] = []

        if self.critic_type == ArchitectureCriticType.SCALABILITY:
            if spec.paradigm.value in ["microservices", "serverless"]:
                score += 0.2
                evidence.append(f"Paradigm '{spec.paradigm.value}' supports horizontal scaling")
            if spec.data_strategy.value in ["cqrs", "event_sourcing"]:
                score += 0.1
                evidence.append(
                    f"Data strategy '{spec.data_strategy.value}' enables read/write separation"
                )

        elif self.critic_type == ArchitectureCriticType.PRAGMATISM:
            if intent and intent.implicit_requirements:
                if intent.implicit_requirements.budget == "budget_conscious":
                    if spec.paradigm.value in ["monolith", "modular_monolith"]:
                        score += 0.2
                        evidence.append("Monolith pattern matches budget constraints")
                    elif spec.paradigm.value == "microservices":
                        score -= 0.1
                        evidence.append("Microservices may exceed budget constraints")

        elif self.critic_type == ArchitectureCriticType.COST:
            cost_factors = {
                "serverless": (0.8, "Serverless can scale to zero cost when idle"),
                "monolith": (0.7, "Single deployment reduces operational costs"),
                "microservices": (0.4, "Multiple services increase operational costs"),
            }
            if spec.paradigm.value in cost_factors:
                base_score, ev = cost_factors[spec.paradigm.value]
                score = base_score
                evidence.append(ev)

        elif self.critic_type == ArchitectureCriticType.ROBUSTNESS:
            if spec.communication.value in ["message_queue", "event_bus"]:
                score += 0.2
                evidence.append(f"'{spec.communication.value}' pattern supports async resilience")

        elif self.critic_type == ArchitectureCriticType.SECURITY_ARCH:
            if intent and intent.implicit_requirements and intent.implicit_requirements.security:
                score += 0.2
                evidence.append("Security requirements explicitly addressed in architecture")

        elif self.critic_type == ArchitectureCriticType.MAINTAINABILITY:
            if spec.paradigm.value == "modular_monolith":
                score += 0.15
                evidence.append("Modular monolith balances complexity and maintainability")

        evidence.append("Fallback evaluation based on pattern matching")

        return EvidenceBasedEvaluation(
            critic=self.critic_type.name.lower(),
            score=min(max(score, 0), 1),
            evidence=evidence,
            reasoning="Pattern-based fallback evaluation",
            confidence=0.6,
            suggestions=suggestions,
        )


class EvidenceMADPanel:
    """
    Multi-Agent Debate panel with evidence-based critics.

    All critics must provide evidence for their evaluations.
    Evaluations without evidence have reduced confidence.

    Attributes:
        critics: List of evidence critics
        provider: LLM provider
        consensus_threshold: Threshold for consensus
        max_rounds: Maximum debate rounds
    """

    DEFAULT_CRITICS = [
        ArchitectureCriticType.SCALABILITY,
        ArchitectureCriticType.PRAGMATISM,
        ArchitectureCriticType.COST,
        ArchitectureCriticType.ROBUSTNESS,
    ]

    def __init__(
        self,
        provider: Any = None,
        critic_types: list[ArchitectureCriticType] | None = None,
        consensus_threshold: float = 0.85,
        max_rounds: int = 3,
        model: str | None = None,
        tools: list[Any] | None = None,
    ):
        self.provider = provider
        self.critic_types = critic_types or self.DEFAULT_CRITICS
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        self.model = model
        self._tools = tools

        self.critics = [
            EvidenceCritic(ct, provider, model, tools=tools) for ct in self.critic_types
        ]

        self._logger = logger

    async def evaluate_with_evidence(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> tuple[ArchitectureSpec, list[EvidenceBasedEvaluation]]:
        """
        Evaluate architecture with evidence from all critics.

        Args:
            spec: Architecture specification
            intent: Original intent

        Returns:
            Tuple of (possibly modified spec, list of evaluations)
        """
        evaluations = []

        for critic in self.critics:
            evaluation = await critic.evaluate(spec, intent)
            evaluations.append(evaluation)

        avg_score = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0

        reliable_count = sum(1 for e in evaluations if e.is_reliable())
        if reliable_count < len(evaluations) // 2:
            self._logger.warning(
                f"Only {reliable_count}/{len(evaluations)} evaluations are reliable"
            )

        for evaluation in evaluations:
            for issue in evaluation.suggestions:
                spec.risks.append(
                    {
                        "source": evaluation.critic,
                        "issue": issue,
                        "severity": "medium" if evaluation.score > 0.5 else "high",
                        "evidence": evaluation.evidence[:2],
                    }
                )

        spec.consensus_reached = avg_score >= self.consensus_threshold
        spec.debate_rounds = 1
        spec.selected_path_score = avg_score

        return spec, evaluations

    async def debate_with_evidence(
        self,
        spec: ArchitectureSpec,
        intent: StructuredIntent,
    ) -> tuple[ArchitectureSpec, bool]:
        """
        Conduct multi-round debate with evidence requirements.

        Args:
            spec: Architecture specification
            intent: Original intent

        Returns:
            Tuple of (modified spec, consensus reached)
        """
        for round_num in range(self.max_rounds):
            spec, evaluations = await self.evaluate_with_evidence(spec, intent)

            avg_score = sum(e.score for e in evaluations) / len(evaluations)
            reliable_count = sum(1 for e in evaluations if e.is_reliable())

            if avg_score >= self.consensus_threshold and reliable_count >= len(evaluations) // 2:
                self._logger.info(
                    f"Evidence-based consensus at round {round_num + 1}: "
                    f"score={avg_score:.2f}, reliable={reliable_count}/{len(evaluations)}"
                )
                return spec, True

            if round_num < self.max_rounds - 1:
                spec = self._apply_evidence_based_feedback(spec, evaluations)

        return spec, False

    def _apply_evidence_based_feedback(
        self,
        spec: ArchitectureSpec,
        evaluations: list[EvidenceBasedEvaluation],
    ) -> ArchitectureSpec:
        """Apply feedback based on evidence"""
        for evaluation in evaluations:
            if evaluation.score < 0.7 and evaluation.is_reliable():
                for suggestion in evaluation.suggestions[:2]:
                    spec.metadata[f"feedback_{evaluation.critic}"] = suggestion

        return spec

    def get_evaluation_summary(
        self,
        evaluations: list[EvidenceBasedEvaluation],
    ) -> dict[str, Any]:
        """Get summary of evaluations with evidence statistics"""
        if not evaluations:
            return {"error": "No evaluations"}

        return {
            "average_score": sum(e.score for e in evaluations) / len(evaluations),
            "reliable_count": sum(1 for e in evaluations if e.is_reliable()),
            "total_evaluations": len(evaluations),
            "evidence_strength_distribution": {
                strength.name: sum(1 for e in evaluations if e.evidence_strength == strength)
                for strength in EvidenceStrength
            },
            "critics_with_evidence": [e.critic for e in evaluations if len(e.evidence) >= 2],
        }


def create_evidence_panel(
    provider: Any = None,
    critic_types: list[ArchitectureCriticType] | None = None,
) -> EvidenceMADPanel:
    """Create an evidence-based MAD panel."""
    return EvidenceMADPanel(
        provider=provider,
        critic_types=critic_types,
    )
