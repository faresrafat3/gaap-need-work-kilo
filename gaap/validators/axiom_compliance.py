"""
Axiom Compliance Validator - Project Constitution Enforcement
Implements: docs/evolution_plan_2026/41_VALIDATORS_AUDIT_SPEC.md

Features:
- Positive constraints (must use X)
- Negative constraints (must not use X)
- Project axiom validation
- Integration with gaap/core/axioms.py
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


logger = logging.getLogger("gaap.validators.axiom_compliance")


class ConstraintType(Enum):
    MUST_USE = auto()
    MUST_NOT_USE = auto()
    MUST_IMPORT = auto()
    MUST_NOT_IMPORT = auto()
    MUST_BE_ASYNC = auto()
    MUST_HAVE_TYPE_HINTS = auto()
    MUST_HAVE_DOCSTRING = auto()
    MAX_FUNCTION_LENGTH = auto()
    MAX_FILE_LENGTH = auto()
    NAMING_CONVENTION = auto()


@dataclass
class Constraint:
    name: str
    constraint_type: ConstraintType
    value: str | int | list[str]
    description: str = ""
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.constraint_type.name,
            "value": str(self.value),
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class ComplianceIssue:
    constraint_name: str
    message: str
    line: int
    severity: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint_name,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class ComplianceResult:
    is_compliant: bool
    issues: list[ComplianceIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    scan_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "issues_count": len(self.issues),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "issues": [i.to_dict() for i in self.issues],
            "scan_time_ms": self.scan_time_ms,
        }


DEFAULT_CONSTRAINTS: list[Constraint] = [
    Constraint(
        name="no_sync_requests",
        constraint_type=ConstraintType.MUST_NOT_IMPORT,
        value="requests",
        description="Use aiohttp instead of requests for async support",
        severity="high",
    ),
    Constraint(
        name="async_database",
        constraint_type=ConstraintType.MUST_IMPORT,
        value="asyncpg",
        description="Database operations should use async drivers",
        severity="medium",
    ),
    Constraint(
        name="no_pickle",
        constraint_type=ConstraintType.MUST_NOT_IMPORT,
        value="pickle",
        description="Pickle is unsafe for untrusted data",
        severity="high",
    ),
    Constraint(
        name="async_functions",
        constraint_type=ConstraintType.MUST_BE_ASYNC,
        value="true",
        description="Functions doing I/O should be async",
        severity="medium",
    ),
    Constraint(
        name="type_hints",
        constraint_type=ConstraintType.MUST_HAVE_TYPE_HINTS,
        value="true",
        description="Functions should have type hints",
        severity="low",
    ),
    Constraint(
        name="max_function_length",
        constraint_type=ConstraintType.MAX_FUNCTION_LENGTH,
        value=50,
        description="Functions should not exceed 50 lines",
        severity="medium",
    ),
    Constraint(
        name="snake_case_functions",
        constraint_type=ConstraintType.NAMING_CONVENTION,
        value="snake_case",
        description="Function names should use snake_case",
        severity="low",
    ),
]


class AxiomComplianceValidator:
    """
    Validates code against project axioms and constraints.

    Features:
    - Positive constraints (must use X)
    - Negative constraints (must not use X)
    - Async enforcement
    - Type hint checking
    - Function length limits
    - Naming convention checking

    Usage:
        validator = AxiomComplianceValidator()
        result = validator.validate(code)
        print(f"Compliant: {result.is_compliant}")
    """

    def __init__(
        self,
        constraints: list[Constraint] | None = None,
        strict_mode: bool = False,
    ) -> None:
        self.constraints = constraints or DEFAULT_CONSTRAINTS.copy()
        self.strict_mode = strict_mode
        self._logger = logger

    def validate(self, code: str, filename: str = "<string>") -> ComplianceResult:
        import time

        start_time = time.time()

        issues: list[ComplianceIssue] = []
        checks_passed = 0
        checks_failed = 0

        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            issues.append(
                ComplianceIssue(
                    constraint_name="syntax",
                    message=f"Syntax error: {e.msg}",
                    line=e.lineno or 1,
                    severity="critical",
                )
            )
            return ComplianceResult(
                is_compliant=False,
                issues=issues,
                checks_failed=1,
            )

        for constraint in self.constraints:
            passed, constraint_issues = self._check_constraint(tree, code, constraint)
            if passed:
                checks_passed += 1
            else:
                checks_failed += 1
                issues.extend(constraint_issues)

        scan_time = (time.time() - start_time) * 1000

        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])

        is_compliant = critical_issues == 0 and (self.strict_mode or high_issues == 0)

        if issues:
            self._logger.warning(
                f"Axiom compliance found {len(issues)} issues "
                f"({checks_passed} passed, {checks_failed} failed)"
            )

        return ComplianceResult(
            is_compliant=is_compliant,
            issues=issues,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            scan_time_ms=scan_time,
        )

    def _check_constraint(
        self,
        tree: ast.AST,
        code: str,
        constraint: Constraint,
    ) -> tuple[bool, list[ComplianceIssue]]:
        issues: list[ComplianceIssue] = []

        if constraint.constraint_type == ConstraintType.MUST_NOT_IMPORT:
            if self._has_import(tree, str(constraint.value)):
                issues.append(
                    ComplianceIssue(
                        constraint_name=constraint.name,
                        message=f"Must not import '{constraint.value}': {constraint.description}",
                        line=1,
                        severity=constraint.severity,
                        suggestion=f"Use an alternative to '{constraint.value}'",
                    )
                )

        elif constraint.constraint_type == ConstraintType.MUST_USE:
            if not self._has_usage(tree, str(constraint.value)):
                issues.append(
                    ComplianceIssue(
                        constraint_name=constraint.name,
                        message=f"Must use '{constraint.value}': {constraint.description}",
                        line=1,
                        severity=constraint.severity,
                    )
                )

        elif constraint.constraint_type == ConstraintType.MUST_BE_ASYNC:
            issues.extend(self._check_async_functions(tree, constraint))

        elif constraint.constraint_type == ConstraintType.MUST_HAVE_TYPE_HINTS:
            issues.extend(self._check_type_hints(tree, constraint))

        elif constraint.constraint_type == ConstraintType.MAX_FUNCTION_LENGTH:
            max_lines = constraint.value if isinstance(constraint.value, int) else 50
            issues.extend(self._check_function_length(tree, max_lines, constraint))

        elif constraint.constraint_type == ConstraintType.NAMING_CONVENTION:
            issues.extend(self._check_naming(tree, str(constraint.value), constraint))

        return len(issues) == 0, issues

    def _has_import(self, tree: ast.AST, module_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module_name or alias.name.startswith(f"{module_name}."):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == module_name or node.module.startswith(f"{module_name}.")
                ):
                    return True
        return False

    def _has_usage(self, tree: ast.AST, name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                return True
            if isinstance(node, ast.Attribute) and node.attr == name:
                return True
        return False

    def _check_async_functions(
        self,
        tree: ast.AST,
        constraint: Constraint,
    ) -> list[ComplianceIssue]:
        issues: list[ComplianceIssue] = []

        io_patterns = {"request", "fetch", "read", "write", "save", "load", "query", "execute"}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                for call in ast.walk(node):
                    if isinstance(call, ast.Call):
                        func_name = self._get_call_name(call)
                        if any(pattern in func_name.lower() for pattern in io_patterns):
                            issues.append(
                                ComplianceIssue(
                                    constraint_name=constraint.name,
                                    message=f"Function '{node.name}' does I/O but is not async",
                                    line=node.lineno,
                                    severity=constraint.severity,
                                    suggestion="Add 'async' keyword to function definition",
                                )
                            )
                            break

        return issues

    def _check_type_hints(
        self,
        tree: ast.AST,
        constraint: Constraint,
    ) -> list[ComplianceIssue]:
        issues: list[ComplianceIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is None:
                    has_return = any(
                        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)
                    )
                    if has_return:
                        issues.append(
                            ComplianceIssue(
                                constraint_name=constraint.name,
                                message=f"Function '{node.name}' lacks return type hint",
                                line=node.lineno,
                                severity=constraint.severity,
                            )
                        )

                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != "self" and arg.arg != "cls":
                        issues.append(
                            ComplianceIssue(
                                constraint_name=constraint.name,
                                message=f"Argument '{arg.arg}' in '{node.name}' lacks type hint",
                                line=node.lineno,
                                severity="low",
                            )
                        )

        return issues

    def _check_function_length(
        self,
        tree: ast.AST,
        max_lines: int,
        constraint: Constraint,
    ) -> list[ComplianceIssue]:
        issues: list[ComplianceIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "end_lineno") and node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                    if length > max_lines:
                        issues.append(
                            ComplianceIssue(
                                constraint_name=constraint.name,
                                message=f"Function '{node.name}' is {length} lines (max: {max_lines})",
                                line=node.lineno,
                                severity=constraint.severity,
                                suggestion="Consider breaking into smaller functions",
                            )
                        )

        return issues

    def _check_naming(
        self,
        tree: ast.AST,
        convention: str,
        constraint: Constraint,
    ) -> list[ComplianceIssue]:
        issues: list[ComplianceIssue] = []

        if convention == "snake_case":
            pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        elif convention == "camelCase":
            pattern = re.compile(r"^[a-z][a-zA-Z0-9]*$")
        elif convention == "PascalCase":
            pattern = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
        else:
            return issues

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not pattern.match(node.name):
                    if not node.name.startswith("_"):
                        issues.append(
                            ComplianceIssue(
                                constraint_name=constraint.name,
                                message=f"Function '{node.name}' doesn't follow {convention}",
                                line=node.lineno,
                                severity=constraint.severity,
                            )
                        )

        return issues

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def remove_constraint(self, name: str) -> bool:
        for i, c in enumerate(self.constraints):
            if c.name == name:
                self.constraints.pop(i)
                return True
        return False

    def get_constraints(self) -> list[Constraint]:
        return self.constraints.copy()

    def get_stats(self) -> dict[str, Any]:
        return {
            "constraints_count": len(self.constraints),
            "strict_mode": self.strict_mode,
            "constraints": [c.to_dict() for c in self.constraints],
        }


def create_axiom_validator(
    strict: bool = False,
    custom_constraints: list[Constraint] | None = None,
) -> AxiomComplianceValidator:
    return AxiomComplianceValidator(
        constraints=custom_constraints,
        strict_mode=strict,
    )
