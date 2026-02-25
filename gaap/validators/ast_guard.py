"""
AST Guard - Security Pattern Detection via AST Analysis
Implements: docs/evolution_plan_2026/41_VALIDATORS_AUDIT_SPEC.md

Features:
- AST pattern matching for dangerous code patterns
- Call chain tracing
- Dangerous import detection
- Shell injection detection
- Code injection detection
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


logger = logging.getLogger("gaap.validators.ast_guard")


class ASTIssueType(Enum):
    DANGEROUS_FUNCTION = auto()
    DANGEROUS_IMPORT = auto()
    SHELL_INJECTION = auto()
    CODE_INJECTION = auto()
    UNSAFE_DESERIALIZATION = auto()
    SQL_INJECTION_RISK = auto()
    PATH_TRAVERSAL = auto()
    HARDCODED_SECRET = auto()
    ASSERT_STATEMENT = auto()
    EXCEPT_BARE = auto()
    GLOBAL_STATEMENT = auto()


@dataclass
class ASTIssue:
    issue_type: ASTIssueType
    message: str
    line: int
    column: int = 0
    severity: str = "medium"
    node_type: str = ""
    code_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.issue_type.name,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "node_type": self.node_type,
        }


@dataclass
class ASTScanResult:
    is_safe: bool
    issues: list[ASTIssue] = field(default_factory=list)
    scan_time_ms: float = 0.0
    lines_scanned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "issues_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
            "scan_time_ms": self.scan_time_ms,
            "lines_scanned": self.lines_scanned,
        }


DANGEROUS_FUNCTIONS: dict[str, str] = {
    "eval": "Code injection risk: eval() can execute arbitrary code",
    "exec": "Code injection risk: exec() can execute arbitrary code",
    "compile": "Code injection risk: compile() can create executable code",
    "execfile": "Code injection risk: execfile() executes file contents",
    "__import__": "Dynamic import can load arbitrary modules",
    "input": "In Python 2, input() evaluates input as code",
    "breakpoint": "Breakpoint can expose sensitive state in production",
}

DANGEROUS_IMPORTS: dict[str, str] = {
    "pickle": "Pickle can execute arbitrary code during deserialization",
    "marshal": "Marshal can execute arbitrary code during deserialization",
    "shelve": "Shelve uses pickle internally",
    "subprocess": "Subprocess with shell=True can lead to shell injection",
    "os.system": "os.system can lead to shell injection",
    "os.popen": "os.popen can lead to shell injection",
    "commands": "Deprecated module, use subprocess instead",
}

SHELL_TRUE_FUNCTIONS = {"call", "run", "Popen", "check_output", "check_call"}


class ASTGuard:
    """
    AST-based security pattern detector.

    Features:
    - Detects dangerous function calls (eval, exec, compile)
    - Detects unsafe imports (pickle, marshal)
    - Detects shell=True in subprocess calls
    - Detects hardcoded secrets
    - Detects bare except clauses

    Usage:
        guard = ASTGuard()
        result = guard.scan("eval(user_input)")
        print(result.is_safe)  # False
    """

    def __init__(
        self,
        strict_mode: bool = True,
        detect_secrets: bool = True,
    ) -> None:
        self.strict_mode = strict_mode
        self.detect_secrets = detect_secrets
        self._logger = logger

    def scan(self, code: str, filename: str = "<string>") -> ASTScanResult:
        import time

        start_time = time.time()

        issues: list[ASTIssue] = []
        lines_scanned = len(code.splitlines())

        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.DANGEROUS_FUNCTION,
                    message=f"Syntax error: {e.msg}",
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    severity="low",
                )
            )
            return ASTScanResult(is_safe=False, issues=issues, lines_scanned=lines_scanned)

        for node in ast.walk(tree):
            self._check_node(node, issues, code)

        if self.detect_secrets:
            self._detect_secrets(code, issues)

        scan_time = (time.time() - start_time) * 1000

        is_safe = len([i for i in issues if i.severity in ("critical", "high")]) == 0

        if issues:
            self._logger.warning(f"AST Guard found {len(issues)} issues")

        return ASTScanResult(
            is_safe=is_safe,
            issues=issues,
            scan_time_ms=scan_time,
            lines_scanned=lines_scanned,
        )

    def _check_node(self, node: ast.AST, issues: list[ASTIssue], code: str) -> None:
        if isinstance(node, ast.Call):
            self._check_call(node, issues)
        elif isinstance(node, ast.Import):
            self._check_import(node, issues)
        elif isinstance(node, ast.ImportFrom):
            self._check_import_from(node, issues)
        elif isinstance(node, ast.Assert):
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.ASSERT_STATEMENT,
                    message="Assert statements may be stripped in optimized mode",
                    line=node.lineno,
                    severity="low",
                    node_type="Assert",
                )
            )
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.EXCEPT_BARE,
                    message="Bare except clause catches all exceptions including KeyboardInterrupt",
                    line=node.lineno,
                    severity="medium" if self.strict_mode else "low",
                    node_type="ExceptHandler",
                )
            )
        elif isinstance(node, ast.Global):
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.GLOBAL_STATEMENT,
                    message="Global statement can make code harder to reason about",
                    line=node.lineno,
                    severity="low",
                    node_type="Global",
                )
            )

    def _check_call(self, node: ast.Call, issues: list[ASTIssue]) -> None:
        func_name = self._get_func_name(node)

        if func_name in DANGEROUS_FUNCTIONS:
            severity = "critical" if func_name in ("eval", "exec") else "high"
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.DANGEROUS_FUNCTION,
                    message=DANGEROUS_FUNCTIONS[func_name],
                    line=node.lineno,
                    column=node.col_offset,
                    severity=severity,
                    node_type="Call",
                )
            )

        if func_name in ("eval", "exec", "compile"):
            for arg in node.args:
                if not isinstance(arg, ast.Constant):
                    issues.append(
                        ASTIssue(
                            issue_type=ASTIssueType.CODE_INJECTION,
                            message=f"{func_name}() with non-constant argument is highly dangerous",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="critical",
                            node_type="Call",
                        )
                    )

        if func_name == "subprocess.call" or func_name in SHELL_TRUE_FUNCTIONS:
            if func_name in SHELL_TRUE_FUNCTIONS and self._has_shell_true(node):
                issues.append(
                    ASTIssue(
                        issue_type=ASTIssueType.SHELL_INJECTION,
                        message=f"subprocess.{func_name} with shell=True is vulnerable to injection",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="high",
                        node_type="Call",
                    )
                )

        if func_name in ("pickle.loads", "pickle.load", "marshal.loads", "marshal.load"):
            issues.append(
                ASTIssue(
                    issue_type=ASTIssueType.UNSAFE_DESERIALIZATION,
                    message=f"{func_name} can execute arbitrary code during deserialization",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="high",
                    node_type="Call",
                )
            )

        if func_name in ("open", "file"):
            for arg in node.args[:1]:
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                    issues.append(
                        ASTIssue(
                            issue_type=ASTIssueType.PATH_TRAVERSAL,
                            message="Potential path traversal: file path constructed from variables",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="medium",
                            node_type="Call",
                        )
                    )

    def _check_import(self, node: ast.Import, issues: list[ASTIssue]) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in DANGEROUS_IMPORTS:
                issues.append(
                    ASTIssue(
                        issue_type=ASTIssueType.DANGEROUS_IMPORT,
                        message=DANGEROUS_IMPORTS[module_name],
                        line=node.lineno,
                        severity="high" if module_name in ("pickle", "marshal") else "medium",
                        node_type="Import",
                    )
                )

    def _check_import_from(self, node: ast.ImportFrom, issues: list[ASTIssue]) -> None:
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in DANGEROUS_IMPORTS:
                issues.append(
                    ASTIssue(
                        issue_type=ASTIssueType.DANGEROUS_IMPORT,
                        message=DANGEROUS_IMPORTS[module_name],
                        line=node.lineno,
                        severity="high" if module_name in ("pickle", "marshal") else "medium",
                        node_type="ImportFrom",
                    )
                )

    def _get_func_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _has_shell_true(self, node: ast.Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "shell":
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    return True
                if isinstance(keyword.value, ast.NameConstant) and keyword.value.value is True:
                    return True
        return False

    def _detect_secrets(self, code: str, issues: list[ASTIssue]) -> None:
        secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']+(["\']|)', "password"),
            (r'(?i)(api_key|apikey|api-key)\s*=\s*["\'][^"\']+(["\']|)', "API key"),
            (r'(?i)(secret|token)\s*=\s*["\'][^"\']+(["\']|)', "secret/token"),
            (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
            (r"AKIA[0-9A-Z]{16}", "AWS access key"),
            (r"ghp_[a-zA-Z0-9]{36}", "GitHub token"),
        ]

        for pattern, secret_type in secret_patterns:
            for match in re.finditer(pattern, code):
                line_num = code[: match.start()].count("\n") + 1
                issues.append(
                    ASTIssue(
                        issue_type=ASTIssueType.HARDCODED_SECRET,
                        message=f"Potential hardcoded {secret_type} detected",
                        line=line_num,
                        severity="high",
                        node_type="Literal",
                    )
                )

    def get_stats(self) -> dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "detect_secrets": self.detect_secrets,
            "dangerous_functions": list(DANGEROUS_FUNCTIONS.keys()),
            "dangerous_imports": list(DANGEROUS_IMPORTS.keys()),
        }


def create_ast_guard(
    strict: bool = True,
    detect_secrets: bool = True,
) -> ASTGuard:
    return ASTGuard(strict_mode=strict, detect_secrets=detect_secrets)
