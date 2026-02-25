"""
Debt Configuration - Settings for Technical Debt Analysis
==========================================================

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md

Configuration presets for debt scanning, interest calculation,
and refinancing operations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class DebtType(Enum):
    """أنواع الـ Technical Debt"""

    TODO = auto()
    FIXME = auto()
    XXX = auto()
    HACK = auto()
    BUG = auto()
    COMPLEXITY = auto()
    DUPLICATE = auto()
    DEAD_CODE = auto()
    LONG_FUNCTION = auto()


class DebtPriority(Enum):
    """أولوية الـ Debt"""

    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


class ProposalStatus(Enum):
    """حالة Refactoring Proposal"""

    PENDING = auto()
    IN_PROGRESS = auto()
    READY_FOR_REVIEW = auto()
    APPROVED = auto()
    MERGED = auto()
    REJECTED = auto()
    ABANDONED = auto()


@dataclass
class DebtConfig:
    """
    Configuration for Technical Debt Agent.

    Controls scanning behavior, thresholds, and safety constraints.
    """

    enabled: bool = True

    complexity_warning: int = 10
    complexity_critical: int = 15

    long_function_lines: int = 50
    long_function_warning: int = 100

    duplicate_min_lines: int = 6
    duplicate_similarity_threshold: float = 0.8

    markers: list[str] = field(default_factory=lambda: ["TODO", "FIXME", "XXX", "HACK", "BUG"])

    marker_priorities: dict[str, DebtPriority] = field(
        default_factory=lambda: {
            "FIXME": DebtPriority.CRITICAL,
            "BUG": DebtPriority.CRITICAL,
            "XXX": DebtPriority.HIGH,
            "HACK": DebtPriority.HIGH,
            "TODO": DebtPriority.MEDIUM,
        }
    )

    criticality_weight: float = 0.4
    age_weight: float = 0.2
    reference_weight: float = 0.2
    coverage_weight: float = 0.2

    critical_files: list[str] = field(
        default_factory=lambda: [
            "auth",
            "security",
            "main",
            "router",
            "api",
            "core",
            "layer0",
            "layer1",
            "layer2",
            "layer3",
        ]
    )

    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/tests/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/migrations/**",
        ]
    )

    never_push_to_main: bool = True
    branch_prefix: str = "refactor/debt-"

    llm_enabled: bool = True
    llm_max_proposal_tokens: int = 1000

    max_debt_items: int = 100
    interest_threshold_high: float = 0.7
    interest_threshold_critical: float = 0.85

    storage_path: str = ".gaap/debt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "complexity_warning": self.complexity_warning,
            "complexity_critical": self.complexity_critical,
            "long_function_lines": self.long_function_lines,
            "long_function_warning": self.long_function_warning,
            "duplicate_min_lines": self.duplicate_min_lines,
            "duplicate_similarity_threshold": self.duplicate_similarity_threshold,
            "markers": self.markers,
            "marker_priorities": {k: v.name for k, v in self.marker_priorities.items()},
            "criticality_weight": self.criticality_weight,
            "age_weight": self.age_weight,
            "reference_weight": self.reference_weight,
            "coverage_weight": self.coverage_weight,
            "critical_files": self.critical_files,
            "exclude_patterns": self.exclude_patterns,
            "never_push_to_main": self.never_push_to_main,
            "branch_prefix": self.branch_prefix,
            "llm_enabled": self.llm_enabled,
            "llm_max_proposal_tokens": self.llm_max_proposal_tokens,
            "max_debt_items": self.max_debt_items,
            "interest_threshold_high": self.interest_threshold_high,
            "interest_threshold_critical": self.interest_threshold_critical,
            "storage_path": self.storage_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtConfig":
        marker_priorities = {}
        for k, v in data.get("marker_priorities", {}).items():
            try:
                marker_priorities[k] = DebtPriority[v]
            except KeyError:
                marker_priorities[k] = DebtPriority.MEDIUM

        return cls(
            enabled=data.get("enabled", True),
            complexity_warning=data.get("complexity_warning", 10),
            complexity_critical=data.get("complexity_critical", 15),
            long_function_lines=data.get("long_function_lines", 50),
            long_function_warning=data.get("long_function_warning", 100),
            duplicate_min_lines=data.get("duplicate_min_lines", 6),
            duplicate_similarity_threshold=data.get("duplicate_similarity_threshold", 0.8),
            markers=data.get("markers", ["TODO", "FIXME", "XXX", "HACK", "BUG"]),
            marker_priorities=marker_priorities,
            criticality_weight=data.get("criticality_weight", 0.4),
            age_weight=data.get("age_weight", 0.2),
            reference_weight=data.get("reference_weight", 0.2),
            coverage_weight=data.get("coverage_weight", 0.2),
            critical_files=data.get("critical_files", []),
            exclude_patterns=data.get("exclude_patterns", []),
            never_push_to_main=data.get("never_push_to_main", True),
            branch_prefix=data.get("branch_prefix", "refactor/debt-"),
            llm_enabled=data.get("llm_enabled", True),
            llm_max_proposal_tokens=data.get("llm_max_proposal_tokens", 1000),
            max_debt_items=data.get("max_debt_items", 100),
            interest_threshold_high=data.get("interest_threshold_high", 0.7),
            interest_threshold_critical=data.get("interest_threshold_critical", 0.85),
            storage_path=data.get("storage_path", ".gaap/debt"),
        )

    @classmethod
    def conservative(cls) -> "DebtConfig":
        """Conservative config - only critical issues"""
        return cls(
            enabled=True,
            complexity_warning=15,
            complexity_critical=20,
            long_function_lines=100,
            markers=["FIXME", "BUG"],
            interest_threshold_high=0.8,
            interest_threshold_critical=0.9,
        )

    @classmethod
    def aggressive(cls) -> "DebtConfig":
        """Aggressive config - catch everything"""
        return cls(
            enabled=True,
            complexity_warning=8,
            complexity_critical=12,
            long_function_lines=30,
            long_function_warning=60,
            duplicate_min_lines=4,
            markers=["TODO", "FIXME", "XXX", "HACK", "BUG", "NOTE", "OPTIMIZE"],
            interest_threshold_high=0.5,
            interest_threshold_critical=0.7,
        )

    @classmethod
    def development(cls) -> "DebtConfig":
        """Development config - balanced for active development"""
        return cls(
            enabled=True,
            complexity_warning=10,
            complexity_critical=15,
            long_function_lines=50,
            markers=["TODO", "FIXME", "XXX", "HACK", "BUG"],
            llm_enabled=True,
        )


def create_debt_config(preset: str = "default", **kwargs: Any) -> DebtConfig:
    """Create a DebtConfig with optional preset."""
    presets: dict[str, Callable[[], DebtConfig]] = {
        "default": DebtConfig,
        "conservative": DebtConfig.conservative,
        "aggressive": DebtConfig.aggressive,
        "development": DebtConfig.development,
    }

    factory = presets.get(preset, DebtConfig)
    config = factory()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
