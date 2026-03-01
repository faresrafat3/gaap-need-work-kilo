"""
MAD (Multi-Agent Debate) Architecture Panel.

Provides multi-critic evaluation of architecture specifications through structured
debate rounds with consensus calculation.

Usage:
    from gaap.layers.strategic.mad_panel import MADArchitecturePanel, CriticEvaluation

    panel = MADArchitecturePanel(max_rounds=3, consensus_threshold=0.85)
    spec, reached = await panel.debate(architecture_spec, intent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.mad.critic_prompts import (
    ARCH_SYSTEM_PROMPTS,
    ArchitectureCriticType,
    build_architecture_prompt,
)
from gaap.mad.response_parser import (
    CriticParseError,
    fallback_architecture_evaluation,
    parse_architecture_critic_response,
)

if TYPE_CHECKING:
    pass


@dataclass
class CriticEvaluation:
    """Single critic evaluation result."""

    critic: str
    score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation to dictionary."""
        return {
            "critic": self.critic,
            "score": self.score,
            "raw_score": self.raw_score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "reasoning": self.reasoning,
        }


class MADArchitecturePanel:
    """Multi-Agent Debate Architecture Panel with critic evaluation."""

    ARCH_CRITICS: list[ArchitectureCriticType] = [
        ArchitectureCriticType.SCALABILITY,
        ArchitectureCriticType.PRAGMATISM,
        ArchitectureCriticType.COST,
        ArchitectureCriticType.ROBUSTNESS,
    ]

    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.85,
        provider: Any = None,
        critic_model: Optional[str] = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.provider = provider
        self.critic_model = critic_model or "llama-3.3-70b-versatile"
        self._logger = get_logger("gaap.layer1.mad")
        self._llm_failures = 0
        self._evaluations_history: list[list[CriticEvaluation]] = []

    async def debate(self, spec: Any, intent: Any) -> tuple[Any, bool]:
        """Conduct multi-round debate until consensus or max rounds."""
        for round_num in range(self.max_rounds):
            evaluations = await self.evaluate(spec, intent)
            self._evaluations_history.append(evaluations)

            avg_score = sum(e.score for e in evaluations) / len(evaluations)

            if avg_score >= self.consensus_threshold:
                spec.consensus_reached = True
                spec.debate_rounds = round_num + 1
                self._logger.info(
                    f"MAD consensus reached at round {round_num + 1}: score={avg_score:.2f}"
                )
                return spec, True

            spec = self._apply_critiques(spec, evaluations)

        spec.debate_rounds = self.max_rounds
        spec.consensus_reached = False
        return spec, False

    async def evaluate(self, spec: Any, intent: Any) -> list[CriticEvaluation]:
        """Evaluate architecture with all critics (LLM or fallback)."""
        if self.provider is None:
            return self._evaluate_fallback(spec, intent)

        evaluations = []
        for critic_type in self.ARCH_CRITICS:
            result = await self._evaluate_with_llm(spec, intent, critic_type)
            evaluations.append(
                CriticEvaluation(
                    critic=result["critic"],
                    score=result["score"],
                    issues=result.get("issues", []),
                    suggestions=result.get("suggestions", []),
                    reasoning=result.get("reasoning", ""),
                    raw_score=result.get("raw_score", result["score"] * 100),
                )
            )
        return evaluations

    def get_consensus(self, evaluations: list[CriticEvaluation]) -> dict[str, Any]:
        """Calculate consensus metrics from evaluations."""
        if not evaluations:
            return {
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "consensus_reached": False,
                "approved_count": 0,
                "total_critics": 0,
            }

        scores = [e.score for e in evaluations]
        avg_score = sum(scores) / len(scores)
        approved = sum(1 for e in evaluations if e.score >= self.consensus_threshold)

        return {
            "average_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "consensus_reached": avg_score >= self.consensus_threshold,
            "approved_count": approved,
            "total_critics": len(evaluations),
        }

    def get_evaluation_history(self) -> list[list[CriticEvaluation]]:
        """Get history of evaluation rounds."""
        return self._evaluations_history.copy()

    def reset_history(self) -> None:
        """Clear evaluation history."""
        self._evaluations_history.clear()

    async def _evaluate_with_llm(
        self, spec: Any, intent: Any, critic_type: ArchitectureCriticType
    ) -> dict[str, Any]:
        """Evaluate with a specific critic using LLM."""
        try:
            system_prompt = ARCH_SYSTEM_PROMPTS.get(
                critic_type, ARCH_SYSTEM_PROMPTS[ArchitectureCriticType.SCALABILITY]
            )
            user_prompt = build_architecture_prompt(spec, intent)

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.critic_model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices or not response.choices[0].message.content:
                self._logger.warning(f"LLM call failed for {critic_type.name}, using fallback")
                self._llm_failures += 1
                return self._get_fallback_eval(spec, intent, critic_type)

            parsed = parse_architecture_critic_response(
                response.choices[0].message.content, critic_type
            )

            return {
                "critic": critic_type.name.lower(),
                "score": parsed["score"] / 100.0,
                "raw_score": parsed["score"],
                "issues": parsed["issues"],
                "suggestions": parsed["suggestions"],
                "reasoning": parsed["reasoning"],
            }

        except CriticParseError as e:
            self._logger.warning(f"Parse error for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._get_fallback_eval(spec, intent, critic_type)
        except Exception as e:
            self._logger.warning(f"LLM evaluation failed for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._get_fallback_eval(spec, intent, critic_type)

    def _get_fallback_eval(
        self, spec: Any, intent: Any, critic_type: ArchitectureCriticType
    ) -> dict[str, Any]:
        """Get fallback evaluation when LLM fails."""
        result = fallback_architecture_evaluation(critic_type, spec, intent)
        return {
            "critic": critic_type.name.lower(),
            "score": result["score"] / 100.0,
            "raw_score": result["score"],
            "issues": result["issues"],
            "suggestions": result["suggestions"],
            "reasoning": result["reasoning"],
        }

    def _evaluate_fallback(self, spec: Any, intent: Any) -> list[CriticEvaluation]:
        """Evaluate using fallback logic (no LLM)."""
        return [
            self._scalability_eval(spec, intent),
            self._pragmatism_eval(spec, intent),
            self._cost_eval(spec, intent),
            self._robustness_eval(spec, intent),
        ]

    def _scalability_eval(self, spec: Any, intent: Any) -> CriticEvaluation:
        """Scalability critic."""
        score = 0.5
        issues, suggestions = [], []

        paradigm = getattr(spec, "paradigm", None)
        paradigm_val = getattr(paradigm, "value", str(paradigm)) if paradigm else ""
        data_strategy = getattr(spec, "data_strategy", None)
        ds_val = getattr(data_strategy, "value", str(data_strategy)) if data_strategy else ""

        if paradigm_val == "microservices":
            score += 0.3
        elif paradigm_val == "monolith":
            score -= 0.2
            issues.append("Monolith may limit horizontal scaling")
            suggestions.append("Consider horizontal scaling patterns")
        if ds_val == "cqrs":
            score += 0.1

        score = min(max(score, 0.0), 1.0)
        return CriticEvaluation(
            critic="scalability",
            score=score,
            raw_score=score * 100,
            issues=issues,
            suggestions=suggestions if score < 0.7 else [],
            reasoning="Based on paradigm analysis",
        )

    def _pragmatism_eval(self, spec: Any, intent: Any) -> CriticEvaluation:
        """Pragmatism critic."""
        score = 0.7
        issues, suggestions = [], []

        paradigm = getattr(spec, "paradigm", None)
        paradigm_val = getattr(paradigm, "value", str(paradigm)) if paradigm else ""
        implicit_req = getattr(intent, "implicit_requirements", None)
        budget = getattr(implicit_req, "budget", None) if implicit_req else None
        timeline = getattr(implicit_req, "timeline", None) if implicit_req else None

        if paradigm_val == "microservices" and budget == "budget_conscious":
            score -= 0.3
            issues.append("Microservices may be over-engineering for budget")
            suggestions.append("Start simple, evolve later")
        if timeline == "urgent" and paradigm_val != "modular_monolith":
            score -= 0.1
            issues.append("Modular monolith may be faster")
            suggestions.append("Consider modular monolith")

        score = min(max(score, 0.0), 1.0)
        return CriticEvaluation(
            critic="pragmatism",
            score=score,
            raw_score=score * 100,
            issues=issues,
            suggestions=suggestions if score < 0.7 else [],
            reasoning="Based on constraints analysis",
        )

    def _cost_eval(self, spec: Any, intent: Any) -> CriticEvaluation:
        """Cost critic."""
        issues, suggestions = [], []
        cost_factors = {
            "serverless": 0.7,
            "monolith": 0.8,
            "modular_monolith": 0.75,
            "microservices": 0.4,
            "event_driven": 0.6,
            "hexagonal": 0.65,
            "layered": 0.75,
        }
        paradigm = getattr(spec, "paradigm", None)
        paradigm_val = getattr(paradigm, "value", str(paradigm)) if paradigm else ""
        score = cost_factors.get(paradigm_val, 0.5)

        implicit_req = getattr(intent, "implicit_requirements", None)
        budget = getattr(implicit_req, "budget", None) if implicit_req else None

        if budget == "budget_conscious" and paradigm_val == "microservices":
            issues.append("High operational costs expected")
            suggestions.append("Consider managed services")

        score = min(max(score, 0.0), 1.0)
        return CriticEvaluation(
            critic="cost",
            score=score,
            raw_score=score * 100,
            issues=issues,
            suggestions=suggestions if score < 0.6 else [],
            reasoning="Based on cost model",
        )

    def _robustness_eval(self, spec: Any, intent: Any) -> CriticEvaluation:
        """Robustness critic."""
        score = 0.6
        issues, suggestions = [], []

        implicit_req = getattr(intent, "implicit_requirements", None)
        security = getattr(implicit_req, "security", None) if implicit_req else None
        if security:
            score += 0.2

        communication = getattr(spec, "communication", None)
        comm_val = getattr(communication, "value", str(communication)) if communication else ""
        if comm_val in ("message_queue", "event_bus"):
            score += 0.1

        score = min(max(score, 0.0), 1.0)
        if score < 0.7:
            suggestions.extend(["Add circuit breakers", "Implement retry logic"])

        return CriticEvaluation(
            critic="robustness",
            score=score,
            raw_score=score * 100,
            issues=issues,
            suggestions=suggestions,
            reasoning="Based on resilience patterns",
        )

    def _apply_critiques(self, spec: Any, evaluations: list[CriticEvaluation]) -> Any:
        """Apply critiques to improve specification."""
        for eval_result in evaluations:
            for issue in eval_result.issues:
                risks = getattr(spec, "risks", None)
                if risks is not None:
                    risks.append(
                        {
                            "source": eval_result.critic,
                            "issue": issue,
                            "severity": "medium" if eval_result.score > 0.5 else "high",
                        }
                    )
        return spec

    def get_stats(self) -> dict[str, Any]:
        """Get panel statistics."""
        return {
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "llm_failures": self._llm_failures,
            "evaluation_rounds": len(self._evaluations_history),
            "has_provider": self.provider is not None,
            "critic_model": self.critic_model,
            "active_critics": [c.name for c in self.ARCH_CRITICS],
        }
