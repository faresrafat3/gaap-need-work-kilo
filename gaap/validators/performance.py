"""
Performance Validator - Code Quality Metrics
Implements: docs/evolution_plan_2026/41_VALIDATORS_AUDIT_SPEC.md

Features:
- Cyclomatic Complexity (via Radon)
- Maintainability Index
- Raw metrics (LOC, comments, etc.)
- Quality thresholds
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


logger = logging.getLogger("gaap.validators.performance")


class MetricType(Enum):
    CYCLOMATIC_COMPLEXITY = auto()
    MAINTAINABILITY_INDEX = auto()
    LINES_OF_CODE = auto()
    COMMENT_RATIO = auto()
    COGNITIVE_COMPLEXITY = auto()


@dataclass
class ComplexityResult:
    name: str
    type: str  # function, method, class
    complexity: int
    rank: str  # A-F
    lineno: int
    col_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "complexity": self.complexity,
            "rank": self.rank,
            "line": self.lineno,
        }


@dataclass
class PerformanceReport:
    is_acceptable: bool
    complexity_results: list[ComplexityResult] = field(default_factory=list)
    maintainability_index: float = 0.0
    total_loc: int = 0
    total_comments: int = 0
    comment_ratio: float = 0.0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    scan_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "maintainability_index": self.maintainability_index,
            "total_loc": self.total_loc,
            "comment_ratio": self.comment_ratio,
            "avg_complexity": self.avg_complexity,
            "max_complexity": self.max_complexity,
            "issues_count": len(self.issues),
            "issues": self.issues,
            "scan_time_ms": self.scan_time_ms,
        }


@dataclass
class PerformanceConfig:
    max_complexity: int = 10
    min_maintainability: float = 20.0
    min_comment_ratio: float = 0.1
    fail_on_violation: bool = True

    @classmethod
    def strict(cls) -> PerformanceConfig:
        return cls(
            max_complexity=8,
            min_maintainability=25.0,
            min_comment_ratio=0.15,
        )

    @classmethod
    def relaxed(cls) -> PerformanceConfig:
        return cls(
            max_complexity=15,
            min_maintainability=15.0,
            min_comment_ratio=0.05,
        )

    @classmethod
    def default(cls) -> PerformanceConfig:
        return cls()


class PerformanceValidator:
    """
    Code quality metrics validator.

    Features:
    - Cyclomatic complexity via AST
    - Maintainability index calculation
    - Comment ratio analysis
    - Configurable thresholds

    Usage:
        validator = PerformanceValidator()
        report = validator.validate(code)
        print(f"Maintainability: {report.maintainability_index}")
    """

    RANK_THRESHOLDS = [
        (1, 5, "A"),
        (6, 10, "B"),
        (11, 20, "C"),
        (21, 30, "D"),
        (31, 40, "E"),
        (41, float("inf"), "F"),
    ]

    def __init__(self, config: PerformanceConfig | None = None) -> None:
        self.config = config or PerformanceConfig.default()
        self._logger = logger
        self._radon_available = False

        try:
            import radon.complexity as radon_cc
            import radon.metrics as radon_metrics

            self._radon_cc = radon_cc
            self._radon_metrics = radon_metrics
            self._radon_available = True
        except ImportError:
            self._logger.warning(
                "Radon not installed. Using built-in complexity calculation. "
                "Install with: pip install radon"
            )

    def validate(self, code: str, filename: str = "<string>") -> PerformanceReport:
        import time

        start_time = time.time()

        issues: list[dict[str, Any]] = []
        complexity_results: list[ComplexityResult] = []
        maintainability_index = 0.0
        total_loc = 0
        total_comments = 0

        if self._radon_available:
            complexity_results = self._radon_complexity(code)
            maintainability_index = self._radon_maintainability(code)
        else:
            complexity_results = self._builtin_complexity(code)
            maintainability_index = self._builtin_maintainability(code)

        loc_result = self._count_loc(code)
        total_loc = loc_result["loc"]
        total_comments = loc_result["comments"]
        comment_ratio = loc_result["comment_ratio"]

        for result in complexity_results:
            if result.complexity > self.config.max_complexity:
                issues.append(
                    {
                        "type": "complexity_exceeded",
                        "message": f"{result.type} '{result.name}' has complexity {result.complexity} "
                        f"(max: {self.config.max_complexity})",
                        "line": result.lineno,
                        "severity": "high"
                        if result.complexity > self.config.max_complexity * 2
                        else "medium",
                    }
                )

        if maintainability_index < self.config.min_maintainability:
            issues.append(
                {
                    "type": "low_maintainability",
                    "message": f"Maintainability index {maintainability_index:.1f} is below "
                    f"threshold {self.config.min_maintainability}",
                    "severity": "medium",
                }
            )

        if comment_ratio < self.config.min_comment_ratio and total_loc > 20:
            issues.append(
                {
                    "type": "low_comment_ratio",
                    "message": f"Comment ratio {comment_ratio:.1%} is below "
                    f"threshold {self.config.min_comment_ratio:.1%}",
                    "severity": "low",
                }
            )

        complexities = [r.complexity for r in complexity_results]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        max_complexity = max(complexities) if complexities else 0

        scan_time = (time.time() - start_time) * 1000

        is_acceptable = len([i for i in issues if i["severity"] in ("critical", "high")]) == 0

        if issues:
            self._logger.warning(f"Performance validator found {len(issues)} issues")

        return PerformanceReport(
            is_acceptable=is_acceptable,
            complexity_results=complexity_results,
            maintainability_index=maintainability_index,
            total_loc=total_loc,
            total_comments=total_comments,
            comment_ratio=comment_ratio,
            avg_complexity=avg_complexity,
            max_complexity=max_complexity,
            issues=issues,
            scan_time_ms=scan_time,
        )

    def _get_rank(self, complexity: int) -> str:
        for min_val, max_val, rank in self.RANK_THRESHOLDS:
            if min_val <= complexity <= max_val:
                return rank
        return "F"

    def _radon_complexity(self, code: str) -> list[ComplexityResult]:
        results: list[ComplexityResult] = []

        try:
            cc_results = self._radon_cc.cc_visit(code)
            for item in cc_results:
                results.append(
                    ComplexityResult(
                        name=item.name,
                        type=item.classname.lower() if item.classname else "function",
                        complexity=item.complexity,
                        rank=item.letter,
                        lineno=item.lineno,
                        col_offset=item.col_offset,
                    )
                )
        except Exception as e:
            self._logger.debug(f"Radon complexity failed: {e}")

        return results

    def _radon_maintainability(self, code: str) -> float:
        try:
            mi_result = self._radon_metrics.mi_visit(code, True)
            return mi_result if isinstance(mi_result, float) else 0.0
        except Exception as e:
            self._logger.debug(f"Radon maintainability failed: {e}")
            return 0.0

    def _builtin_complexity(self, code: str) -> list[ComplexityResult]:
        results: list[ComplexityResult] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return results

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                complexity = self._calculate_complexity(node)
                results.append(
                    ComplexityResult(
                        name=node.name,
                        type="method" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        complexity=complexity,
                        rank=self._get_rank(complexity),
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    )
                )

        return results

    def _calculate_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity

    def _builtin_maintainability(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0

        loc = len(code.splitlines())

        num_functions = sum(
            1 for _ in ast.walk(tree) if isinstance(_, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

        num_classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))

        volume = loc * (1 + num_functions + num_classes * 2)

        complexity_results = self._builtin_complexity(code)
        avg_complexity = (
            sum(r.complexity for r in complexity_results) / len(complexity_results)
            if complexity_results
            else 1
        )

        import math

        mi = max(0, 171 - 5.2 * math.log(volume + 1) - 0.23 * avg_complexity - 16.2)

        return min(100, max(0, mi))

    def _count_loc(self, code: str) -> dict[str, Any]:
        lines = code.splitlines()
        total_loc = len(lines)

        blank_lines = sum(1 for line in lines if not line.strip())

        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_lines += 1
            elif '"""' in stripped or "'''" in stripped:
                comment_lines += 1

        code_lines = total_loc - blank_lines - comment_lines
        comment_ratio = comment_lines / code_lines if code_lines > 0 else 0.0

        return {
            "loc": code_lines,
            "blank": blank_lines,
            "comments": comment_lines,
            "total": total_loc,
            "comment_ratio": comment_ratio,
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "config": {
                "max_complexity": self.config.max_complexity,
                "min_maintainability": self.config.min_maintainability,
                "min_comment_ratio": self.config.min_comment_ratio,
            },
            "radon_available": self._radon_available,
        }


def create_performance_validator(
    max_complexity: int = 10,
    min_maintainability: float = 20.0,
) -> PerformanceValidator:
    config = PerformanceConfig(
        max_complexity=max_complexity,
        min_maintainability=min_maintainability,
    )
    return PerformanceValidator(config)
