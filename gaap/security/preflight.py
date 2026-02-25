"""
Pre-Flight Check - Zero-Trust Code Validation
Implements: docs/evolution_plan_2026/24_LAYER3_AUDIT_SPEC.md

Validates code BEFORE execution:
- Lesson injection from memory
- Static analysis (ruff, bandit)
- Policy checks (banned imports)
- Security scanning
"""

import ast
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class CheckSeverity(Enum):
    """مستوى خطورة النتيجة"""

    PASS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class CheckResult:
    """نتيجة فحص"""

    check_name: str
    severity: CheckSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS, CheckSeverity.WARNING)


@dataclass
class PreFlightReport:
    """تقرير الفحص المسبق"""

    task_id: str
    overall_passed: bool
    results: list[CheckResult] = field(default_factory=list)
    lessons_injected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scan_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.overall_passed,
            "lessons_count": len(self.lessons_injected),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "duration_ms": self.scan_duration_ms,
        }


BANNED_IMPORTS = {
    "socket": "Network access not allowed without explicit permission",
    "subprocess": "Use DockerSandbox instead",
    "os.system": "Use subprocess with sandbox instead",
    "eval": "Code injection risk",
    "exec": "Code injection risk",
    "compile": "Code injection risk",
    "__import__": "Dynamic imports not allowed",
    "importlib.import_module": "Dynamic imports not allowed",
    "pickle.loads": "Deserialization attack risk",
    "marshal.loads": "Deserialization attack risk",
    "shelve.open": "Potential security risk",
}

ALLOWED_WITH_PERMISSION = {
    "requests": "HTTP requests allowed with network permission",
    "urllib": "HTTP requests allowed with network permission",
    "httpx": "HTTP requests allowed with network permission",
    "aiohttp": "HTTP requests allowed with network permission",
}


from gaap.core.logging import get_standard_logger as get_logger


class PreFlightCheck:
    """
    Pre-Flight Validator for Code Execution

    Runs before any code is executed to:
    1. Inject relevant lessons from memory
    2. Run static analysis
    3. Check security policies
    4. Validate against banned patterns
    """

    def __init__(
        self,
        memory: Any = None,
        strict_mode: bool = True,
        run_bandit: bool = True,
        run_ruff: bool = True,
    ) -> None:
        self._memory = memory
        self._strict_mode = strict_mode
        self._run_bandit = run_bandit
        self._run_ruff = run_ruff
        self._logger = get_logger("gaap.preflight")

        self._total_checks = 0
        self._total_failures = 0

    def check(
        self,
        code: str,
        task_id: str,
        task_description: str | None = None,
        capability_token: dict[str, Any] | None = None,
    ) -> PreFlightReport:
        """
        Run all pre-flight checks

        Args:
            code: Code to validate
            task_id: Task identifier
            task_description: Task description for lesson retrieval
            capability_token: Permission token for restricted operations

        Returns:
            PreFlightReport with all check results
        """
        start_time = time.time()
        results: list[CheckResult] = []
        lessons: list[str] = []

        self._total_checks += 1

        lessons = self._inject_lessons(task_description or task_id)
        results.append(self._check_syntax(code))
        results.append(self._check_banned_imports(code, capability_token))
        results.append(self._check_dangerous_patterns(code))

        if self._run_ruff:
            results.append(self._run_ruff_check(code))

        if self._run_bandit:
            results.append(self._run_bandit_check(code))

        warnings = [r.message for r in results if r.severity == CheckSeverity.WARNING]
        errors = [
            r.message
            for r in results
            if r.severity in (CheckSeverity.ERROR, CheckSeverity.CRITICAL)
        ]

        overall_passed = len(errors) == 0

        if not overall_passed:
            self._total_failures += 1

        report = PreFlightReport(
            task_id=task_id,
            overall_passed=overall_passed,
            results=results,
            lessons_injected=lessons,
            warnings=warnings,
            errors=errors,
            scan_duration_ms=(time.time() - start_time) * 1000,
        )

        self._logger.info(
            f"Pre-flight check for {task_id}: {'PASSED' if overall_passed else 'FAILED'} "
            f"({len(warnings)} warnings, {len(errors)} errors)"
        )

        return report

    def _inject_lessons(self, task_description: str) -> list[str]:
        """Inject relevant lessons from memory"""
        lessons: list[str] = []

        if not self._memory:
            return lessons

        try:
            if hasattr(self._memory, "retrieve_lessons"):
                lessons = self._memory.retrieve_lessons(task_description, k=3)
            elif hasattr(self._memory, "retrieve"):
                results = self._memory.retrieve(task_description, k=3)
                for r in results:
                    if hasattr(r, "content"):
                        lessons.append(r.content[:200])
                    elif isinstance(r, dict) and "content" in r:
                        lessons.append(r["content"][:200])
                    elif isinstance(r, str):
                        lessons.append(r[:200])

            if lessons:
                self._logger.debug(f"Injected {len(lessons)} lessons for: {task_description[:50]}")

        except Exception as e:
            self._logger.debug(f"Failed to inject lessons: {e}")

        return lessons[:3]

    def _check_syntax(self, code: str) -> CheckResult:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return CheckResult(
                check_name="syntax",
                severity=CheckSeverity.PASS,
                message="Code parses successfully",
            )
        except SyntaxError as e:
            return CheckResult(
                check_name="syntax",
                severity=CheckSeverity.ERROR,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                details={"line": e.lineno, "offset": e.offset},
                suggestions=["Fix the syntax error before execution"],
            )

    def _check_banned_imports(
        self, code: str, capability_token: dict[str, Any] | None
    ) -> CheckResult:
        """Check for banned imports"""
        violations = []
        warnings = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in BANNED_IMPORTS:
                            violations.append(f"import {module}: {BANNED_IMPORTS[module]}")
                        elif module in ALLOWED_WITH_PERMISSION:
                            if not capability_token or not capability_token.get("network"):
                                warnings.append(
                                    f"import {module}: {ALLOWED_WITH_PERMISSION[module]}"
                                )

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        if module in BANNED_IMPORTS:
                            violations.append(f"from {module}: {BANNED_IMPORTS[module]}")
                        elif module in ALLOWED_WITH_PERMISSION:
                            if not capability_token or not capability_token.get("network"):
                                warnings.append(f"from {module}: {ALLOWED_WITH_PERMISSION[module]}")

        except SyntaxError:
            pass

        if violations:
            return CheckResult(
                check_name="banned_imports",
                severity=CheckSeverity.CRITICAL,
                message=f"Banned imports detected: {'; '.join(violations)}",
                details={"violations": violations},
                suggestions=["Remove banned imports or request explicit permission"],
            )

        if warnings:
            return CheckResult(
                check_name="banned_imports",
                severity=CheckSeverity.WARNING,
                message=f"Imports require permission: {'; '.join(warnings)}",
                details={"warnings": warnings},
                suggestions=["Add network permission to capability token"],
            )

        return CheckResult(
            check_name="banned_imports",
            severity=CheckSeverity.PASS,
            message="No banned imports detected",
        )

    def _check_dangerous_patterns(self, code: str) -> CheckResult:
        """Check for dangerous code patterns"""
        dangerous_patterns = [
            (r"eval\s*\(", "eval() detected - code injection risk"),
            (r"exec\s*\(", "exec() detected - code injection risk"),
            (r"compile\s*\(", "compile() detected - code injection risk"),
            (r"__import__\s*\(", "__import__() detected - dynamic import risk"),
            (r"pickle\.loads", "pickle.loads detected - deserialization risk"),
            (r"marshal\.loads", "marshal.loads detected - deserialization risk"),
            (r"subprocess\..*shell\s*=\s*True", "shell=True detected - command injection risk"),
            (r"os\.system\s*\(", "os.system() detected - use subprocess instead"),
        ]

        findings = []
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                findings.append(message)

        if findings:
            return CheckResult(
                check_name="dangerous_patterns",
                severity=CheckSeverity.ERROR,
                message=f"Dangerous patterns detected: {'; '.join(findings)}",
                details={"patterns": findings},
                suggestions=["Remove or sanitize dangerous code patterns"],
            )

        return CheckResult(
            check_name="dangerous_patterns",
            severity=CheckSeverity.PASS,
            message="No dangerous patterns detected",
        )

    def _run_ruff_check(self, code: str) -> CheckResult:
        """Run ruff linter"""
        try:
            result = subprocess.run(
                ["ruff", "check", "-", "--output-format=json"],
                input=code.encode(),
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0:
                return CheckResult(
                    check_name="ruff",
                    severity=CheckSeverity.PASS,
                    message="Ruff check passed",
                )

            import json

            issues = json.loads(result.stdout) if result.stdout else []

            errors = [i for i in issues if i.get("severity") == "error"]
            warnings = [i for i in issues if i.get("severity") != "error"]

            if errors:
                return CheckResult(
                    check_name="ruff",
                    severity=CheckSeverity.WARNING,
                    message=f"Ruff found {len(errors)} errors and {len(warnings)} warnings",
                    details={"errors": errors[:5], "warnings": warnings[:5]},
                    suggestions=["Fix linting issues before execution"],
                )

            return CheckResult(
                check_name="ruff",
                severity=CheckSeverity.WARNING,
                message=f"Ruff found {len(warnings)} warnings",
                details={"warnings": warnings[:5]},
            )

        except FileNotFoundError:
            return CheckResult(
                check_name="ruff",
                severity=CheckSeverity.PASS,
                message="Ruff not installed, skipping",
            )
        except Exception as e:
            return CheckResult(
                check_name="ruff",
                severity=CheckSeverity.WARNING,
                message=f"Ruff check failed: {str(e)[:50]}",
            )

    def _run_bandit_check(self, code: str) -> CheckResult:
        """Run bandit security scanner"""
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["bandit", "-f", "json", "-r", temp_path],
                capture_output=True,
                timeout=30,
            )

            import json
            import os

            try:
                os.unlink(temp_path)
            except OSError:
                pass

            output = json.loads(result.stdout) if result.stdout else {}
            results = output.get("results", [])

            high_severity = [r for r in results if r.get("issue_severity") == "HIGH"]
            medium_severity = [r for r in results if r.get("issue_severity") == "MEDIUM"]

            if high_severity:
                return CheckResult(
                    check_name="bandit",
                    severity=CheckSeverity.ERROR,
                    message=f"Bandit found {len(high_severity)} high severity issues",
                    details={"issues": high_severity[:3]},
                    suggestions=["Fix security issues before execution"],
                )

            if medium_severity:
                return CheckResult(
                    check_name="bandit",
                    severity=CheckSeverity.WARNING,
                    message=f"Bandit found {len(medium_severity)} medium severity issues",
                    details={"issues": medium_severity[:3]},
                )

            return CheckResult(
                check_name="bandit",
                severity=CheckSeverity.PASS,
                message="Bandit check passed",
            )

        except FileNotFoundError:
            return CheckResult(
                check_name="bandit",
                severity=CheckSeverity.PASS,
                message="Bandit not installed, skipping",
            )
        except Exception as e:
            return CheckResult(
                check_name="bandit",
                severity=CheckSeverity.WARNING,
                message=f"Bandit check failed: {str(e)[:50]}",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get pre-flight statistics"""
        return {
            "total_checks": self._total_checks,
            "total_failures": self._total_failures,
            "pass_rate": 1 - (self._total_failures / max(self._total_checks, 1)),
        }


def create_preflight_check(memory: Any = None, strict: bool = True) -> PreFlightCheck:
    """Create a PreFlightCheck instance"""
    return PreFlightCheck(memory=memory, strict_mode=strict)
