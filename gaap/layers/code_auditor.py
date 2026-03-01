"""
Code Auditor - Static Analysis Before Execution
================================================

Evolution 2026: Pre-execution code audit.

Key Features:
- Ruff linting
- Bandit security checks
- Banned imports detection
- Policy enforcement
- Configurable severity thresholds

Runs AFTER code generation but BEFORE sandbox execution.
"""

import ast
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger
from gaap.layers.layer2_tactical import AtomicTask
from gaap.layers.layer3_config import Layer3Config

logger = get_logger("gaap.layer3.auditor")


class IssueSeverity(Enum):
    """Severity level for audit issues"""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class IssueType(Enum):
    """Type of audit issue"""

    LINT = auto()
    SECURITY = auto()
    POLICY = auto()
    SYNTAX = auto()
    IMPORT = auto()


@dataclass
class AuditIssue:
    """A single issue found during audit"""

    issue_type: IssueType
    severity: IssueSeverity
    message: str
    line: int = 0
    column: int = 0
    code: str = ""
    fix: str = ""

    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type.name,
            "severity": self.severity.name,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "fix": self.fix,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class AuditResult:
    """Result of code audit"""

    passed: bool
    issues: list[AuditIssue] = field(default_factory=list)

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    audit_time_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "audit_time_ms": self.audit_time_ms,
            "tools_used": self.tools_used,
        }

    def get_errors(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    def get_warnings(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def get_critical(self) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]


class CodeAuditor:
    """
    Static code analysis before execution.

    Runs multiple analysis tools:
    - AST parsing for syntax
    - Import analysis for banned modules
    - Ruff for linting
    - Bandit for security
    """

    def __init__(
        self,
        config: Layer3Config | None = None,
    ):
        self._config = config or Layer3Config()
        self._audit_config = self._config.audit
        self._logger = logger

        self._audits_run = 0
        self._issues_found = 0
        self._blocks = 0

    async def audit(
        self,
        code: str,
        task: AtomicTask | None = None,
    ) -> AuditResult:
        """
        Run full audit on code.

        Args:
            code: Python code to audit
            task: Task context (optional)

        Returns:
            AuditResult with issues and pass/fail status
        """

        if not self._audit_config.enabled:
            return AuditResult(passed=True, tools_used=["disabled"])

        start_time = time.time()
        self._audits_run += 1

        issues: list[AuditIssue] = []
        tools_used: list[str] = []

        syntax_issues = self._check_syntax(code)
        issues.extend(syntax_issues)

        if not syntax_issues:
            import_issues = self._check_banned_imports(code)
            issues.extend(import_issues)

            function_issues = self._check_banned_functions(code)
            issues.extend(function_issues)

            if "ruff" in self._audit_config.tools:
                ruff_issues = self._run_ruff(code)
                issues.extend(ruff_issues)
                tools_used.append("ruff")

            if "bandit" in self._audit_config.tools:
                bandit_issues = self._run_bandit(code)
                issues.extend(bandit_issues)
                tools_used.append("bandit")

            if "mypy" in self._audit_config.tools:
                mypy_issues = self._run_mypy(code)
                issues.extend(mypy_issues)
                tools_used.append("mypy")

        issues = issues[: self._audit_config.max_issues]

        error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)
        info_count = sum(1 for i in issues if i.severity == IssueSeverity.INFO)
        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)

        passed = True
        if self._audit_config.fail_on_errors and (error_count > 0 or critical_count > 0):
            passed = False
            self._blocks += 1

        if self._audit_config.fail_on_warnings and warning_count > 0:
            passed = False

        self._issues_found += len(issues)

        audit_time = (time.time() - start_time) * 1000

        return AuditResult(
            passed=passed,
            issues=issues,
            error_count=error_count + critical_count,
            warning_count=warning_count,
            info_count=info_count,
            audit_time_ms=audit_time,
            tools_used=tools_used if tools_used else ["syntax", "import"],
        )

    def _check_syntax(self, code: str) -> list[AuditIssue]:
        """Check for syntax errors"""

        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                AuditIssue(
                    issue_type=IssueType.SYNTAX,
                    severity=IssueSeverity.ERROR,
                    message=f"Syntax error: {e.msg}",
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    code="SYNTAX001",
                    source="ast",
                )
            )

        return issues

    def _check_banned_imports(self, code: str) -> list[AuditIssue]:
        """Check for banned imports"""

        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        if module_name in self._audit_config.banned_imports:
                            issues.append(
                                AuditIssue(
                                    issue_type=IssueType.IMPORT,
                                    severity=IssueSeverity.CRITICAL,
                                    message=f"Banned import: {alias.name}",
                                    line=node.lineno,
                                    code="IMPORT001",
                                    fix=f"Remove or replace import '{alias.name}'",
                                    source="policy",
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split(".")[0]
                        if module_name in self._audit_config.banned_imports:
                            issues.append(
                                AuditIssue(
                                    issue_type=IssueType.IMPORT,
                                    severity=IssueSeverity.CRITICAL,
                                    message=f"Banned import from: {node.module}",
                                    line=node.lineno,
                                    code="IMPORT002",
                                    fix=f"Remove or replace import from '{node.module}'",
                                    source="policy",
                                )
                            )

        except SyntaxError:
            pass

        return issues

    def _check_banned_functions(self, code: str) -> list[AuditIssue]:
        """Check for banned function calls"""

        issues = []

        for banned in self._audit_config.banned_functions:
            pattern = re.escape(banned)
            for match in re.finditer(pattern, code):
                line_num = code[: match.start()].count("\n") + 1
                issues.append(
                    AuditIssue(
                        issue_type=IssueType.SECURITY,
                        severity=IssueSeverity.CRITICAL,
                        message=f"Banned function call: {banned}",
                        line=line_num,
                        code="FUNC001",
                        fix=f"Remove or replace '{banned}' with a safer alternative",
                        source="policy",
                    )
                )

        return issues

    def _run_ruff(self, code: str) -> list[AuditIssue]:
        """Run ruff linter"""

        issues = []

        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                import json

                try:
                    ruff_issues = json.loads(result.stdout)
                    for issue in ruff_issues:
                        severity = IssueSeverity.WARNING
                        if issue.get("severity") == "error":
                            severity = IssueSeverity.ERROR

                        issues.append(
                            AuditIssue(
                                issue_type=IssueType.LINT,
                                severity=severity,
                                message=issue.get("message", ""),
                                line=issue.get("location", {}).get("row", 0),
                                column=issue.get("location", {}).get("column", 0),
                                code=issue.get("code", ""),
                                fix=issue.get("fix", ""),
                                source="ruff",
                            )
                        )
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            self._logger.debug("ruff not found, skipping")
        except subprocess.TimeoutExpired:
            self._logger.warning("ruff timed out")
        except Exception as e:
            self._logger.debug(f"ruff failed: {e}")

        return issues

    def _run_bandit(self, code: str) -> list[AuditIssue]:
        """Run bandit security checker"""

        issues = []

        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                import json

                try:
                    bandit_result = json.loads(result.stdout)
                    for issue in bandit_result.get("results", []):
                        severity_map = {
                            "LOW": IssueSeverity.INFO,
                            "MEDIUM": IssueSeverity.WARNING,
                            "HIGH": IssueSeverity.ERROR,
                        }

                        severity = severity_map.get(
                            issue.get("issue_severity", "MEDIUM"),
                            IssueSeverity.WARNING,
                        )

                        issues.append(
                            AuditIssue(
                                issue_type=IssueType.SECURITY,
                                severity=severity,
                                message=issue.get("issue_text", ""),
                                line=issue.get("line_number", 0),
                                code=issue.get("test_id", ""),
                                fix=issue.get("more_info", ""),
                                source="bandit",
                                metadata={
                                    "confidence": issue.get("issue_confidence"),
                                    "cwe": issue.get("cwe", {}),
                                },
                            )
                        )
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            self._logger.debug("bandit not found, skipping")
        except subprocess.TimeoutExpired:
            self._logger.warning("bandit timed out")
        except Exception as e:
            self._logger.debug(f"bandit failed: {e}")

        return issues

    def _run_mypy(self, code: str) -> list[AuditIssue]:
        """Run mypy type checker"""

        issues = []

        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["mypy", "--output=json", temp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            import os

            os.unlink(temp_path)

            if result.stdout:
                import json

                try:
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            mypy_issue = json.loads(line)
                            severity = IssueSeverity.WARNING
                            if mypy_issue.get("severity") == "error":
                                severity = IssueSeverity.ERROR

                            issues.append(
                                AuditIssue(
                                    issue_type=IssueType.LINT,
                                    severity=severity,
                                    message=mypy_issue.get("message", ""),
                                    line=mypy_issue.get("line", 0),
                                    column=mypy_issue.get("column", 0),
                                    code=mypy_issue.get("code", ""),
                                    source="mypy",
                                )
                            )
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            self._logger.debug("mypy not found, skipping")
        except subprocess.TimeoutExpired:
            self._logger.warning("mypy timed out")
        except Exception as e:
            self._logger.debug(f"mypy failed: {e}")

        return issues

    def quick_audit(self, code: str) -> bool:
        """
        Quick audit for banned patterns only.

        Faster than full audit, suitable for hot path.
        """

        for banned in self._audit_config.banned_imports:
            if f"import {banned}" in code or f"from {banned}" in code:
                return False

        for banned in self._audit_config.banned_functions:
            if banned in code:
                return False

        try:
            ast.parse(code)
        except SyntaxError:
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get auditor statistics"""

        return {
            "audits_run": self._audits_run,
            "issues_found": self._issues_found,
            "executions_blocked": self._blocks,
            "block_rate": self._blocks / max(self._audits_run, 1),
        }


def create_code_auditor(
    config: Layer3Config | None = None,
) -> CodeAuditor:
    """Factory function to create CodeAuditor"""

    return CodeAuditor(config=config)
