# mypy: ignore-errors
import json
import re
from typing import Any

from gaap.core.types import CriticType
from gaap.mad.critic_prompts import ArchitectureCriticType


class CriticParseError(Exception):
    """Error parsing critic response."""

    pass


def parse_critic_response(response: str, critic_type: CriticType) -> dict[str, Any]:
    """
    Parse LLM response for critic evaluation.

    Args:
        response: Raw LLM response
        critic_type: Type of critic

    Returns:
        Dictionary with score, approved, issues, suggestions, reasoning
    """
    cleaned = _clean_response(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise CriticParseError(f"Failed to parse JSON: {e}. Response: {cleaned[:200]}") from e

    return _validate_and_normalize(data, critic_type)


def _clean_response(response: str) -> str:
    """Clean LLM response to extract JSON."""
    response = response.strip()

    response = re.sub(r"^```json\s*", "", response)
    response = re.sub(r"^```\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        return json_match.group(0)

    return response


def _validate_and_normalize(data: dict[str, Any], critic_type: CriticType) -> dict[str, Any]:
    """Validate and normalize critic response data."""

    score = data.get("score", 70.0)
    if isinstance(score, str):
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 70.0

    score = max(0.0, min(100.0, float(score)))

    approved = data.get("approved")
    if approved is None:
        approved = score >= 70.0
    elif not isinstance(approved, bool):
        approved = bool(approved)

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    issues = [str(i) for i in issues if i]

    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []
    suggestions = [str(s) for s in suggestions if s]

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "score": score,
        "approved": approved,
        "issues": issues,
        "suggestions": suggestions,
        "reasoning": reasoning,
        "critic_type": critic_type,
    }


def fallback_evaluation(critic_type: CriticType, artifact: str) -> dict[str, Any]:
    """
    Fallback evaluation when LLM fails.

    Uses simple heuristics as backup.
    """
    score = 70.0
    issues = []
    suggestions = []
    reasoning = "Fallback evaluation due to LLM parse failure"

    if critic_type == CriticType.SECURITY:
        dangerous = ["eval(", "exec(", "pickle.loads", "subprocess("]
        for pattern in dangerous:
            if pattern in artifact:
                score -= 15
                issues.append(f"Potential dangerous pattern: {pattern}")
                suggestions.append(f"Review use of {pattern}")
        if (
            ("password" in artifact.lower() or "api_key" in artifact.lower())
            and "=" in artifact
            and ('"' in artifact or "'" in artifact)
        ):
            score -= 20
            issues.append("Potential hardcoded secret detected")
            suggestions.append("Use environment variables for secrets")

    elif critic_type == CriticType.PERFORMANCE:
        if "for " in artifact and " in " in artifact and artifact.count("for ") > 3:
            score -= 10
            issues.append("Multiple nested loops detected")
            suggestions.append("Consider optimization for large datasets")
        if "+=" in artifact and "for " in artifact:
            score -= 5
            issues.append("Potential string concatenation in loop")
            suggestions.append("Use join() or list comprehension")

    elif critic_type == CriticType.LOGIC:
        if "return" not in artifact:
            score -= 10
            issues.append("No return statement found")
            suggestions.append("Ensure function returns expected value")
        if "error" in artifact.lower() and "except" not in artifact.lower():
            score -= 5
            issues.append("Error mentioned but no exception handling")
            suggestions.append("Add proper error handling")

    elif critic_type == CriticType.STYLE:
        if len(artifact.split("\n")) > 100:
            score -= 5
            issues.append("Large code block may benefit from splitting")
            suggestions.append("Consider breaking into smaller functions")
        if "    " not in artifact and "\t" not in artifact:
            score -= 5
            issues.append("No indentation detected")
            suggestions.append("Follow proper indentation")

    score = max(0.0, min(100.0, score))
    approved = score >= 70.0

    return {
        "score": score,
        "approved": approved,
        "issues": issues,
        "suggestions": suggestions,
        "reasoning": reasoning,
        "critic_type": critic_type,
    }


def parse_architecture_critic_response(
    response: str, critic_type: ArchitectureCriticType
) -> dict[str, Any]:
    """
    Parse LLM response for architecture critic evaluation.
    """
    cleaned = _clean_response(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise CriticParseError(f"Failed to parse JSON: {e}. Response: {cleaned[:200]}") from e

    score = data.get("score", 70.0)
    if isinstance(score, str):
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 70.0

    score = max(0.0, min(100.0, float(score)))

    approved = data.get("approved")
    if approved is None:
        approved = score >= 70.0
    elif not isinstance(approved, bool):
        approved = bool(approved)

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    issues = [str(i) for i in issues if i]

    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []
    suggestions = [str(s) for s in suggestions if s]

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "score": score,
        "approved": approved,
        "issues": issues,
        "suggestions": suggestions,
        "reasoning": reasoning,
        "critic_type": critic_type,
    }


def fallback_architecture_evaluation(
    critic_type: ArchitectureCriticType, spec, intent
) -> dict[str, Any]:
    """
    Fallback evaluation when LLM fails for architecture critics.
    """
    score = 70.0
    issues = []
    suggestions = []
    reasoning = "Fallback evaluation due to LLM parse failure"

    paradigm = getattr(spec, "paradigm", None)
    if paradigm and hasattr(paradigm, "value"):
        paradigm_val = paradigm.value
    else:
        paradigm_val = str(paradigm)

    if critic_type == ArchitectureCriticType.SCALABILITY:
        if paradigm_val == "monolith":
            score -= 20
            issues.append("Monolith limits horizontal scaling")
            suggestions.append("Consider modular design for future splitting")
        elif paradigm_val == "serverless":
            score += 10

    elif critic_type == ArchitectureCriticType.PRAGMATISM:
        if hasattr(intent, "implicit_requirements") and intent.implicit_requirements:
            budget = getattr(intent.implicit_requirements, "budget", None)
            if budget == "budget_conscious" and paradigm_val == "microservices":
                score -= 20
                issues.append("Microservices over-engineering for budget")
                suggestions.append("Start with modular monolith")

    elif critic_type == ArchitectureCriticType.COST:
        if paradigm_val == "microservices":
            score -= 15
            issues.append("Higher operational costs")
            suggestions.append("Consider managed services")

    elif critic_type == ArchitectureCriticType.ROBUSTNESS:
        comm = getattr(spec, "communication", None)
        if comm and hasattr(comm, "value") and comm.value in ("message_queue", "event_bus"):
            score += 10

    score = max(0.0, min(100.0, score))
    approved = score >= 70.0

    return {
        "score": score,
        "approved": approved,
        "issues": issues,
        "suggestions": suggestions,
        "reasoning": reasoning,
        "critic_type": critic_type,
    }
