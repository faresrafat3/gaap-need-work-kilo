"""
Interest Calculator - Debt Priority Scoring
===========================================

Calculates "interest" scores for technical debt items,
prioritizing them based on multiple factors.

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.maintenance.debt_config import DebtConfig, DebtPriority
from gaap.maintenance.debt_scanner import DebtItem

logger = logging.getLogger("gaap.maintenance.interest")


@dataclass
class InterestFactors:
    """Factors that influence debt interest score."""

    file_criticality: float = 0.0
    code_age_days: int = 0
    reference_count: int = 0
    test_coverage: float = 0.5
    complexity_score: int = 0
    is_tested: bool = False
    last_modified_days: int = 0
    change_frequency: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_criticality": self.file_criticality,
            "code_age_days": self.code_age_days,
            "reference_count": self.reference_count,
            "test_coverage": self.test_coverage,
            "complexity_score": self.complexity_score,
            "is_tested": self.is_tested,
            "last_modified_days": self.last_modified_days,
            "change_frequency": self.change_frequency,
        }


@dataclass
class InterestReport:
    """Report of interest calculations."""

    total_items: int = 0
    total_interest: float = 0.0
    high_interest_count: int = 0
    critical_interest_count: int = 0
    by_type: dict[str, float] = field(default_factory=dict)
    top_items: list[tuple[DebtItem, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_interest": round(self.total_interest, 4),
            "high_interest_count": self.high_interest_count,
            "critical_interest_count": self.critical_interest_count,
            "by_type": {k: round(v, 4) for k, v in self.by_type.items()},
            "top_items": [
                {"item": item.to_dict(), "interest": round(interest, 4)}
                for item, interest in self.top_items
            ],
        }


class InterestCalculator:
    """
    Calculates interest scores for technical debt.

    Interest = weighted combination of:
    - File criticality (is it in a critical path?)
    - Code age (how long has this debt existed?)
    - Reference count (how many other files depend on this?)
    - Test coverage (is this code tested?)
    """

    def __init__(self, config: DebtConfig | None = None):
        self._config = config or DebtConfig()
        self._logger = logger

        self._git_available = self._check_git()

    @property
    def config(self) -> DebtConfig:
        return self._config

    def calculate(
        self,
        debt: DebtItem,
        factors: InterestFactors | None = None,
    ) -> float:
        """
        Calculate interest score for a debt item.

        Args:
            debt: The debt item to calculate interest for
            factors: Optional pre-computed factors

        Returns:
            Interest score between 0.0 and 1.0
        """
        if factors is None:
            factors = self._compute_factors(debt)

        weights = {
            "criticality": self._config.criticality_weight,
            "age": self._config.age_weight,
            "reference": self._config.reference_weight,
            "coverage": self._config.coverage_weight,
        }

        criticality_score = factors.file_criticality

        age_score = min(factors.code_age_days / 365, 1.0)

        reference_score = min(factors.reference_count / 10, 1.0)

        coverage_score = 1.0 - factors.test_coverage

        interest = (
            weights["criticality"] * criticality_score
            + weights["age"] * age_score
            + weights["reference"] * reference_score
            + weights["coverage"] * coverage_score
        )

        if debt.complexity and debt.complexity > self._config.complexity_warning:
            complexity_bonus = min((debt.complexity - self._config.complexity_warning) / 10, 0.2)
            interest += complexity_bonus

        if debt.priority == DebtPriority.CRITICAL:
            interest += 0.2
        elif debt.priority == DebtPriority.HIGH:
            interest += 0.1

        return min(max(interest, 0.0), 1.0)

    def calculate_batch(
        self,
        debts: list[DebtItem],
        project_root: Path | str | None = None,
    ) -> list[tuple[DebtItem, float]]:
        """
        Calculate interest scores for multiple debt items.

        Args:
            debts: List of debt items
            project_root: Project root for computing factors

        Returns:
            List of (debt_item, interest_score) tuples
        """
        results: list[tuple[DebtItem, float]] = []
        root = Path(project_root) if project_root else Path.cwd()

        for debt in debts:
            factors = self._compute_factors(debt, root)
            interest = self.calculate(debt, factors)
            debt.interest_score = interest
            results.append((debt, interest))

        return results

    def prioritize(
        self,
        debts: list[DebtItem],
        project_root: Path | str | None = None,
    ) -> list[DebtItem]:
        """
        Sort debt items by interest score (highest first).

        Args:
            debts: List of debt items
            project_root: Project root for computing factors

        Returns:
            Sorted list of debt items
        """
        scored = self.calculate_batch(debts, project_root)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored]

    def get_top_debt(
        self,
        debts: list[DebtItem],
        n: int = 10,
        project_root: Path | str | None = None,
    ) -> list[tuple[DebtItem, float]]:
        """
        Get top N debt items by interest.

        Args:
            debts: List of debt items
            n: Number of items to return
            project_root: Project root for computing factors

        Returns:
            List of (debt_item, interest_score) tuples
        """
        scored = self.calculate_batch(debts, project_root)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def generate_report(
        self,
        debts: list[DebtItem],
        project_root: Path | str | None = None,
    ) -> InterestReport:
        """
        Generate a comprehensive interest report.

        Args:
            debts: List of debt items
            project_root: Project root for computing factors

        Returns:
            InterestReport with statistics and top items
        """
        scored = self.calculate_batch(debts, project_root)

        report = InterestReport()
        report.total_items = len(scored)

        for item, interest in scored:
            report.total_interest += interest

            type_name = item.type.name
            report.by_type[type_name] = report.by_type.get(type_name, 0.0) + interest

            if interest >= self._config.interest_threshold_critical:
                report.critical_interest_count += 1
            elif interest >= self._config.interest_threshold_high:
                report.high_interest_count += 1

        scored.sort(key=lambda x: x[1], reverse=True)
        report.top_items = scored[:10]

        return report

    def _compute_factors(
        self,
        debt: DebtItem,
        project_root: Path | None = None,
    ) -> InterestFactors:
        """Compute interest factors for a debt item."""
        factors = InterestFactors()

        factors.file_criticality = self._calculate_criticality(debt.file_path)

        if project_root:
            factors.code_age_days = self._get_file_age(debt.file_path, project_root)
            factors.last_modified_days = self._get_last_modified(debt.file_path, project_root)
            factors.reference_count = self._count_references(debt, project_root)

        if debt.complexity:
            factors.complexity_score = debt.complexity

        return factors

    def _calculate_criticality(self, file_path: str) -> float:
        """Calculate how critical a file is (0.0 to 1.0)."""
        path_lower = file_path.lower()

        for critical in self._config.critical_files:
            if critical.lower() in path_lower:
                return 1.0

        if "test" in path_lower:
            return 0.3
        elif "util" in path_lower or "helper" in path_lower:
            return 0.2

        depth = file_path.count("/")
        if depth <= 1:
            return 0.8
        elif depth <= 2:
            return 0.6
        else:
            return 0.4

    def _get_file_age(self, file_path: str, project_root: Path) -> int:
        """Get the age of a file in days."""
        if not self._git_available:
            return 0

        try:
            result = subprocess.run(
                ["git", "log", "--follow", "--format=%at", "--", file_path],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                timestamps = result.stdout.strip().split("\n")
                if timestamps:
                    oldest = int(timestamps[-1])
                    age = datetime.now() - datetime.fromtimestamp(oldest)
                    return age.days
        except Exception:
            pass

        return 0

    def _get_last_modified(self, file_path: str, project_root: Path) -> int:
        """Get days since last modification."""
        if not self._git_available:
            return 0

        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%at", "--", file_path],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                timestamp = int(result.stdout.strip())
                age = datetime.now() - datetime.fromtimestamp(timestamp)
                return age.days
        except Exception:
            pass

        return 0

    def _count_references(self, debt: DebtItem, project_root: Path) -> int:
        """Count references to a function or file."""
        if not debt.function_name or not self._git_available:
            return 0

        try:
            result = subprocess.run(
                ["git", "grep", "-l", debt.function_name],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                return len([f for f in files if f])
        except Exception:
            pass

        return 0

    def _check_git(self) -> bool:
        """Check if git is available."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False


def create_interest_calculator(config: DebtConfig | None = None) -> InterestCalculator:
    """Create an InterestCalculator instance."""
    return InterestCalculator(config=config)
