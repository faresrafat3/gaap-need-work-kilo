"""
World Model - Predict Outcomes Before Execution
================================================

Simulates potential outcomes of actions before executing them.
Enables "thinking before acting" to avoid errors and choose optimal paths.

Usage:
    model = WorldModel(provider)
    prediction = await model.predict_outcome(action, context)
    if prediction.risk_level > 0.7:
        # Choose safer alternative
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gaap.core.types import Message, MessageRole
from gaap.memory import VECTOR_MEMORY_AVAILABLE, LessonStore
from gaap.providers.base_provider import BaseProvider

logger = logging.getLogger("gaap.world_model")


class RiskLevel(Enum):
    """Risk levels for predicted outcomes"""

    SAFE = 0.0
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


@dataclass
class Prediction:
    """Predicted outcome of an action"""

    success_probability: float
    risk_level: float
    potential_issues: list[str]
    suggestions: list[str]
    similar_past_outcomes: list[str]
    confidence: float
    reasoning: str
    processing_time_ms: float = 0.0


@dataclass
class Action:
    """Action to be predicted"""

    name: str
    description: str
    task_type: str = "general"
    parameters: dict[str, Any] = field(default_factory=dict)
    is_destructive: bool = False
    requires_external_access: bool = False


class WorldModel:
    """
    Predicts outcomes of actions before execution.

    Features:
    - Uses past lessons from vector memory
    - LLM-based reasoning for novel situations
    - Risk assessment
    - Suggests safer alternatives
    """

    def __init__(
        self,
        provider: BaseProvider | None = None,
        lesson_store: LessonStore | None = None,
        enable_llm_prediction: bool = True,
    ):
        self._provider = provider
        self._enable_llm = enable_llm_prediction and provider is not None

        if VECTOR_MEMORY_AVAILABLE:
            self._lesson_store: LessonStore | None = lesson_store or LessonStore()
        else:
            self._lesson_store = None

        self._logger = logger
        self._predictions_made = 0

    async def predict_outcome(
        self,
        action: Action,
        context: dict[str, Any] | None = None,
    ) -> Prediction:
        """
        Predict the outcome of an action.

        Args:
            action: The action to predict
            context: Additional context (project state, constraints, etc.)

        Returns:
            Prediction with risk level and suggestions
        """
        start_time = time.time()
        self._predictions_made += 1

        similar_outcomes = await self._get_similar_past_outcomes(action)

        base_risk = self._calculate_base_risk(action, similar_outcomes)

        if self._enable_llm:
            try:
                llm_prediction = await asyncio.wait_for(
                    self._llm_predict(action, context, similar_outcomes), timeout=10.0
                )
            except Exception as e:
                self._logger.warning(f"LLM prediction failed: {e}, using heuristic fallback")
                llm_prediction = self._heuristic_predict(action, context, similar_outcomes)
            risk_level = (base_risk + llm_prediction.get("risk", base_risk)) / 2
            issues = llm_prediction.get("issues", [])
            suggestions = llm_prediction.get("suggestions", [])
            reasoning = llm_prediction.get("reasoning", "")
            confidence = llm_prediction.get("confidence", 0.5)
        else:
            risk_level = base_risk
            issues = self._rule_based_issues(action)
            suggestions = self._rule_based_suggestions(action, similar_outcomes)
            reasoning = "Rule-based prediction (no LLM)"
            confidence = 0.6 if similar_outcomes else 0.3

        success_prob = max(0, 1.0 - risk_level)

        processing_time = (time.time() - start_time) * 1000

        return Prediction(
            success_probability=success_prob,
            risk_level=risk_level,
            potential_issues=issues,
            suggestions=suggestions,
            similar_past_outcomes=similar_outcomes[:5],
            confidence=confidence,
            reasoning=reasoning,
            processing_time_ms=processing_time,
        )

    async def should_proceed(self, prediction: Prediction) -> tuple[bool, str]:
        """
        Decide whether to proceed with an action based on prediction.

        Args:
            prediction: The prediction result

        Returns:
            (should_proceed, reason)
        """
        if prediction.risk_level >= RiskLevel.CRITICAL.value:
            return False, "Risk level too high - critical safety concern"

        if prediction.risk_level >= RiskLevel.HIGH.value:
            if prediction.confidence < 0.5:
                return False, "High risk with low confidence - need more information"
            return (
                True,
                "High risk but proceeding with caution - consider alternatives",
            )

        if prediction.success_probability < 0.3:
            return False, "Low success probability - likely to fail"

        if prediction.potential_issues and prediction.risk_level >= RiskLevel.MEDIUM.value:
            return True, f"Proceeding with awareness of: {prediction.potential_issues[0]}"

        return True, "Safe to proceed"

    async def _get_similar_past_outcomes(self, action: Action) -> list[str]:
        """Get similar past outcomes from lesson store"""
        if not self._lesson_store:
            return []

        query = f"{action.task_type} {action.name}"
        results = self._lesson_store.search(query, n=5)
        return [r.content for r in results if r.score > 0.1]

    def _calculate_base_risk(self, action: Action, similar_outcomes: list[str]) -> float:
        """Calculate base risk from action properties and past outcomes"""
        risk = 0.0

        if action.is_destructive:
            risk += 0.3

        if action.requires_external_access:
            risk += 0.2

        if "delete" in action.name.lower() or "remove" in action.name.lower():
            risk += 0.25

        if "write" in action.name.lower() or "create" in action.name.lower():
            risk += 0.1

        if similar_outcomes:
            failed_count = sum(
                1 for o in similar_outcomes if "fail" in o.lower() or "error" in o.lower()
            )
            failure_rate = failed_count / len(similar_outcomes)
            risk += failure_rate * 0.2

        return min(risk, 1.0)

    async def _llm_predict(
        self,
        action: Action,
        context: dict[str, Any] | None,
        similar_outcomes: list[str],
    ) -> dict[str, Any]:
        """Use LLM to predict outcome"""
        if not self._provider:
            return {}

        system_prompt = """You are a predictive AI that forecasts outcomes of actions.
Analyze the action and predict:
1. Risk level (0.0-1.0)
2. Potential issues
3. Suggestions for safer execution
4. Confidence in prediction (0.0-1.0)
5. Brief reasoning

Respond in JSON format:
{"risk": 0.3, "issues": ["..."], "suggestions": ["..."], "confidence": 0.8, "reasoning": "..."}"""

        context_str = f"\nContext: {context}" if context else ""
        similar_str = f"\nSimilar past outcomes: {similar_outcomes[:3]}" if similar_outcomes else ""

        user_prompt = f"""Predict outcome for action:
Name: {action.name}
Description: {action.description}
Type: {action.task_type}
Destructive: {action.is_destructive}
External Access: {action.requires_external_access}
Parameters: {action.parameters}{context_str}{similar_str}"""

        try:
            response = await self._provider.chat_completion(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=system_prompt),
                    Message(role=MessageRole.USER, content=user_prompt),
                ],
                model=self._provider.default_model,
                temperature=0.3,
                max_tokens=500,
            )

            if response.choices and response.choices[0].message.content:
                import json

                content = response.choices[0].message.content
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])  # type: ignore[no-any-return]
        except Exception as e:
            self._logger.debug(f"LLM prediction failed: {e}")

        return {}

    def _rule_based_issues(self, action: Action) -> list[str]:
        """Generate issues based on rules"""
        issues = []

        if action.is_destructive:
            issues.append("Destructive action - changes cannot be easily undone")

        if action.requires_external_access:
            issues.append("Requires external access - may fail due to network/auth issues")

        for param, value in action.parameters.items():
            if isinstance(value, str):
                if "/" in value and (".." in value or value.startswith("/")):
                    issues.append(f"Parameter '{param}' may access sensitive paths")

        return issues

    def _rule_based_suggestions(self, action: Action, similar_outcomes: list[str]) -> list[str]:
        """Generate suggestions based on rules and past outcomes"""
        suggestions = []

        if action.is_destructive:
            suggestions.append("Create backup before destructive operation")

        if action.requires_external_access:
            suggestions.append("Implement retry logic with exponential backoff")

        for outcome in similar_outcomes:
            if "timeout" in outcome.lower():
                suggestions.append("Increase timeout or optimize for faster execution")
                break

        for outcome in similar_outcomes:
            if "rate limit" in outcome.lower():
                suggestions.append("Implement rate limiting to avoid throttling")
                break

        return suggestions

    def _heuristic_predict(
        self,
        action: Action,
        context: dict[str, Any] | None,
        similar_outcomes: list[str],
    ) -> dict[str, Any]:
        """Heuristic fallback prediction when LLM fails"""
        risk = self._calculate_base_risk(action, similar_outcomes)
        issues = self._rule_based_issues(action)
        suggestions = self._rule_based_suggestions(action, similar_outcomes)

        return {
            "risk": risk,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.6 if similar_outcomes else 0.3,
            "reasoning": "Heuristic prediction (LLM failed)",
        }

    def get_stats(self) -> dict[str, Any]:
        """Get prediction statistics"""
        return {
            "predictions_made": self._predictions_made,
            "llm_enabled": self._enable_llm,
            "lesson_store_available": self._lesson_store is not None,
        }
