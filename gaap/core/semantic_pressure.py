"""
Semantic Pressure System for GAAP

Provides linguistic constraint checking and enforcement for LLM outputs.

Classes:
    - Constraint: Single constraint definition
    - ConstraintViolation: Violation of a constraint
    - SemanticConstraints: Constraint checking and enforcement

Usage:
    from gaap.core.semantic_pressure import SemanticConstraints

    constraints = SemanticConstraints()
    violations = constraints.check_text("We need to ensure robust performance")
    pressure_prompt = constraints.apply_pressure_prompt()
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    HINT = auto()


@dataclass
class Constraint:
    """
    Single constraint definition.

    Attributes:
        pattern: Regex pattern to match violations
        requirement: What is required instead
        severity: Violation severity
        description: Human-readable description
        category: Category of constraint
    """

    pattern: str
    requirement: str
    severity: ConstraintSeverity = ConstraintSeverity.WARNING
    description: str = ""
    category: str = "general"

    def __post_init__(self) -> None:
        self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """Check if the pattern matches text."""
        return bool(self._compiled_pattern.search(text))

    def find_all(self, text: str) -> list[str]:
        """Find all matches in text."""
        return self._compiled_pattern.findall(text)


@dataclass
class ConstraintViolation:
    """
    A violation of a semantic constraint.

    Attributes:
        constraint: The violated constraint
        matched_text: The text that violated the constraint
        position: Position in text (start, end)
        suggestion: Suggested fix
        context: Surrounding context
    """

    constraint: Constraint
    matched_text: str = ""
    position: tuple[int, int] = (0, 0)
    suggestion: str = ""
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.constraint.pattern,
            "requirement": self.constraint.requirement,
            "severity": self.constraint.severity.name,
            "matched_text": self.matched_text,
            "suggestion": self.suggestion,
            "context": self.context,
        }


class SemanticConstraints:
    """
    Linguistic constraints for LLM output quality.

    Enforces specific language patterns to improve output
    precision and actionability.

    Attributes:
        BANNED_VAGUE_TERMS: Terms that should be avoided
        REQUIRE_METRIC_TERMS: Terms requiring quantitative backing
    """

    BANNED_VAGUE_TERMS: list[str] = [
        "ensure",
        "robust",
        "efficient",
        "properly",
        "correctly",
        "appropriate",
        "reasonable",
        "adequate",
        "suitable",
        "sufficient",
        "optimal",
        "effective",
        "seamless",
        "seamlessly",
        "streamlined",
        "comprehensive",
        "intuitive",
        "user-friendly",
        "high-quality",
        "best practices",
        "state-of-the-art",
        "cutting-edge",
        "enterprise-grade",
    ]

    REQUIRE_METRIC_TERMS: list[str] = [
        "fast",
        "slow",
        "big",
        "small",
        "large",
        "scalable",
        "performant",
        "lightweight",
        "heavy",
        "complex",
        "simple",
    ]

    QUANTIFIERS: list[str] = [
        "significant",
        "substantial",
        "considerable",
        "moderate",
        "minor",
        "major",
        "dramatic",
        "noticeable",
    ]

    def __init__(self) -> None:
        self._logger = get_logger("gaap.core.semantic_pressure")
        self._constraints = self._build_constraints()
        self._violation_count = 0
        self._check_count = 0

    def _build_constraints(self) -> list[Constraint]:
        """Build all constraints."""
        constraints = []

        for term in self.BANNED_VAGUE_TERMS:
            constraints.append(
                Constraint(
                    pattern=rf"\b{term}\b",
                    requirement=f"Avoid vague term '{term}'. Use specific, measurable language.",
                    severity=ConstraintSeverity.WARNING,
                    description=f"Vague term detected: {term}",
                    category="vague",
                )
            )

        for term in self.REQUIRE_METRIC_TERMS:
            constraints.append(
                Constraint(
                    pattern=rf"\b{term}\b",
                    requirement=f"Term '{term}' requires quantitative backing. Specify exact metrics.",
                    severity=ConstraintSeverity.INFO,
                    description=f"Metric term needs quantification: {term}",
                    category="metric",
                )
            )

        for term in self.QUANTIFIERS:
            constraints.append(
                Constraint(
                    pattern=rf"\b{term}\b",
                    requirement=f"Quantifier '{term}' should be replaced with specific numbers or percentages.",
                    severity=ConstraintSeverity.HINT,
                    description=f"Vague quantifier: {term}",
                    category="quantifier",
                )
            )

        constraints.append(
            Constraint(
                pattern=r"\bTODO\b|\bFIXME\b|\bXXX\b",
                requirement="Remove placeholder comments before final output.",
                severity=ConstraintSeverity.ERROR,
                description="Placeholder comment detected",
                category="placeholder",
            )
        )

        constraints.append(
            Constraint(
                pattern=r"\bshould\b(?!\s+be\s+able\s+to)",
                requirement="Replace 'should' with 'will' or 'must' for clarity, or explain uncertainty.",
                severity=ConstraintSeverity.INFO,
                description="Uncertain modal verb",
                category="uncertainty",
            )
        )

        constraints.append(
            Constraint(
                pattern=r"\bmay\s+(?:cause|lead\s+to|result\s+in)\b",
                requirement="Replace uncertainty with concrete risk assessment.",
                severity=ConstraintSeverity.HINT,
                description="Vague risk statement",
                category="risk",
            )
        )

        constraints.append(
            Constraint(
                pattern=r"\betc\.?\b|\band\s+so\s+on\b|\band\s+more\b",
                requirement="Complete the list explicitly instead of using placeholders.",
                severity=ConstraintSeverity.INFO,
                description="Incomplete list indicator",
                category="incomplete",
            )
        )

        constraints.append(
            Constraint(
                pattern=r"\bworks?\s+(?:as\s+expected|fine|correctly|properly)\b",
                requirement="Specify exact behavior with test cases or metrics.",
                severity=ConstraintSeverity.WARNING,
                description="Vague success statement",
                category="success",
            )
        )

        return constraints

    def check_text(self, text: str) -> list[ConstraintViolation]:
        """
        Check text for constraint violations.

        Args:
            text: Text to check

        Returns:
            List of violations found
        """
        self._check_count += 1
        violations = []

        for constraint in self._constraints:
            matches = constraint.find_all(text)
            for match in matches:
                matched_text = match if isinstance(match, str) else match[0]

                start = text.lower().find(matched_text.lower())
                end = start + len(matched_text)

                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                context = text[context_start:context_end]

                suggestion = self._generate_suggestion(constraint, matched_text)

                violation = ConstraintViolation(
                    constraint=constraint,
                    matched_text=matched_text,
                    position=(start, end),
                    suggestion=suggestion,
                    context=context,
                )
                violations.append(violation)
                self._violation_count += 1

        if violations:
            self._logger.debug(f"Found {len(violations)} violations in text ({len(text)} chars)")

        return violations

    def check_text_severity(
        self,
        text: str,
        min_severity: ConstraintSeverity = ConstraintSeverity.WARNING,
    ) -> list[ConstraintViolation]:
        """
        Check text for violations at or above a severity level.

        Args:
            text: Text to check
            min_severity: Minimum severity to include

        Returns:
            Filtered list of violations
        """
        all_violations = self.check_text(text)

        severity_order = [
            ConstraintSeverity.ERROR,
            ConstraintSeverity.WARNING,
            ConstraintSeverity.INFO,
            ConstraintSeverity.HINT,
        ]

        min_idx = severity_order.index(min_severity)
        allowed = severity_order[: min_idx + 1]

        return [v for v in all_violations if v.constraint.severity in allowed]

    def apply_pressure_prompt(self) -> str:
        """
        Generate a prompt addition to apply semantic pressure.

        Returns:
            String to add to system prompt
        """
        return """
## Output Quality Requirements

Your outputs must be:

1. **Specific**: Avoid vague terms. Replace:
   - "ensure" → Specify the exact check or mechanism
   - "robust" → List specific error handling and edge cases
   - "efficient" → Provide time/space complexity
   - "properly/correctly" → Describe exact expected behavior

2. **Measurable**: When using comparative terms, provide metrics:
   - "fast" → "O(n log n) with benchmarks showing X ms for Y items"
   - "scalable" → "Handles up to N concurrent users with M latency"
   - "lightweight" → "X KB bundle size, Y dependencies"

3. **Actionable**: Every recommendation should have:
   - A concrete implementation step
   - A verification method
   - A fallback or error handling approach

4. **Complete**: Never use:
   - "etc.", "and so on", "and more" — list all items explicitly
   - "TODO", "FIXME" — provide the actual implementation

5. **Precise Risk Assessment**: Replace "may cause X" with:
   - "Has Y% probability of X when Z conditions apply"
   - "Known failure mode: X, mitigation: Y"

When uncertain, state the uncertainty explicitly with:
- What is known
- What is unknown
- Recommended investigation steps
"""

    def apply_pressure_prompt_short(self) -> str:
        """Generate a shorter pressure prompt for tight context windows."""
        return """
CRITICAL: Avoid vague language.
- "ensure" → specify exact check
- "robust" → list error handling
- "efficient" → give O() complexity
- "properly/correctly" → describe exact behavior
Always provide specific metrics, concrete steps, and complete lists (no "etc.").
"""

    def fix_text(self, text: str) -> tuple[str, list[ConstraintViolation]]:
        """
        Attempt to fix violations in text.

        Args:
            text: Text to fix

        Returns:
            Tuple of (fixed text, remaining violations)
        """
        violations = self.check_text(text)

        fixed_text = text
        unfixable = []

        for violation in violations:
            if violation.suggestion:
                fixed_text = fixed_text.replace(
                    violation.matched_text,
                    f"[SPECIFIC:{violation.constraint.category}]",
                )
            else:
                unfixable.append(violation)

        return fixed_text, unfixable

    def _generate_suggestion(
        self,
        constraint: Constraint,
        matched_text: str,
    ) -> str:
        """Generate a specific suggestion for a violation."""
        suggestions: dict[str, dict[str, str]] = {
            "vague": {
                "ensure": "verify with: <specific check>",
                "robust": "handle errors: <list error cases>",
                "efficient": "optimize to O(<complexity>)",
                "properly": "with correct behavior: <describe>",
                "correctly": "as specified: <reference spec>",
                "appropriate": "suitable for: <list criteria>",
            },
            "metric": {
                "fast": "<X ms for Y operations>",
                "scalable": "<N users, M latency>",
                "lightweight": "<X KB, Y dependencies>",
                "performant": "<X throughput, Y latency>",
            },
            "quantifier": {
                "significant": "<X% improvement>",
                "substantial": "<X times faster>",
                "moderate": "<X units>",
            },
        }

        category_suggestions = suggestions.get(constraint.category, {})
        return category_suggestions.get(
            matched_text.lower(),
            constraint.requirement,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get constraint checking statistics."""
        return {
            "total_checks": self._check_count,
            "total_violations": self._violation_count,
            "violation_rate": (
                self._violation_count / self._check_count if self._check_count > 0 else 0
            ),
            "constraints_count": len(self._constraints),
        }

    def add_custom_constraint(
        self,
        pattern: str,
        requirement: str,
        severity: ConstraintSeverity = ConstraintSeverity.WARNING,
        category: str = "custom",
    ) -> None:
        """Add a custom constraint."""
        constraint = Constraint(
            pattern=pattern,
            requirement=requirement,
            severity=severity,
            category=category,
        )
        self._constraints.append(constraint)
        self._logger.info(f"Added custom constraint: {pattern}")

    def get_constraints_by_category(self) -> dict[str, list[Constraint]]:
        """Get constraints grouped by category."""
        categories: dict[str, list[Constraint]] = {}
        for constraint in self._constraints:
            if constraint.category not in categories:
                categories[constraint.category] = []
            categories[constraint.category].append(constraint)
        return categories


def create_semantic_constraints() -> SemanticConstraints:
    """Factory function to create SemanticConstraints."""
    return SemanticConstraints()
