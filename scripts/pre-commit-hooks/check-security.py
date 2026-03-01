#!/usr/bin/env python3
"""
Security scanner pre-commit hook.

Scans Python files for common security issues:
- Hardcoded secrets and API keys
- Dangerous function usage (eval, exec)
- SQL injection patterns
- Unsafe imports
- Suspicious patterns
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Optional


# Patterns to detect hardcoded secrets
SECRET_PATTERNS = {
    "api_key": re.compile(
        r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][a-zA-Z0-9_-]{16,}["\']',
        re.IGNORECASE,
    ),
    "secret_key": re.compile(
        r'(?i)(secret[_-]?key|secret)\s*[=:]\s*["\'][a-zA-Z0-9_-]{16,}["\']',
        re.IGNORECASE,
    ),
    "password": re.compile(
        r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']',
        re.IGNORECASE,
    ),
    "token": re.compile(
        r'(?i)(token|auth_token|access_token)\s*[=:]\s*["\'][a-zA-Z0-9_-]{20,}["\']',
        re.IGNORECASE,
    ),
    "private_key": re.compile(
        r'(?i)(private[_-]?key|rsa_key)\s*[=:]\s*["\']',
        re.IGNORECASE,
    ),
}

# Dangerous functions/patterns
DANGEROUS_PATTERNS = {
    "eval": re.compile(r"\beval\s*\("),
    "exec": re.compile(r"\bexec\s*\("),
    "compile": re.compile(r"\bcompile\s*\("),
    "__import__": re.compile(r"\b__import__\s*\("),
    "subprocess_shell": re.compile(
        r"subprocess\.\w+.*shell\s*=\s*True",
        re.IGNORECASE,
    ),
    "input_unsafe": re.compile(r"\binput\s*\("),
    "pickle_loads": re.compile(r"pickle\.loads?\s*\("),
    "yaml_unsafe": re.compile(
        r"yaml\.load\s*\([^)]*\)(?!\s*#\s*noqa)",
    ),
    "marshal_loads": re.compile(r"marshal\.loads?\s*\("),
}

# SQL injection patterns
SQL_INJECTION_PATTERNS = {
    "string_format_sql": re.compile(
        r'(?:execute|cursor\.execute|run_query)\s*\(\s*["\'].*%s.*["\']\s*%',
        re.IGNORECASE,
    ),
    "f_string_sql": re.compile(
        r'(?:execute|cursor\.execute|run_query)\s*\(\s*f["\']',
        re.IGNORECASE,
    ),
    "format_sql": re.compile(
        r'(?:execute|cursor\.execute|run_query)\s*\(\s*["\'].*\{.*\}.*["\']\.format\s*\(',
        re.IGNORECASE,
    ),
    "concat_sql": re.compile(
        r'(?:execute|cursor\.execute|run_query)\s*\(\s*["\'].*\+\s*\w+\s*\+',
        re.IGNORECASE,
    ),
}

# Unsafe imports
UNSAFE_IMPORTS = {
    "pickle",
    "marshal",
    "subprocess",
    "os.system",
    "ctypes",
    "code",
    "commands",
}

# Allowlist for false positives
ALLOWLIST = {
    "password": ["get_password", "set_password", "password_hash", "check_password"],
    "token": ["get_token", "set_token", "csrf_token", "xsrf_token"],
    "api_key": ["get_api_key", "API_KEY_PATH", "api_key_file"],
}


class SecurityIssue:
    """Represents a security issue found in code."""

    def __init__(
        self,
        file: Path,
        line: int,
        category: str,
        message: str,
        severity: str = "medium",
    ):
        self.file = file
        self.line = line
        self.category = category
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(self.severity, "⚪")
        return f"{severity_icon} {self.file}:{self.line} [{self.category.upper()}] {self.message}"


class SecurityScanner(ast.NodeVisitor):
    """AST-based security scanner."""

    def __init__(self, file_path: Path, source: str):
        self.file_path = file_path
        self.source = source
        self.issues: list[SecurityIssue] = []
        self.lines = source.split("\n")

    def scan(self) -> list[SecurityIssue]:
        """Run all security checks."""
        self._check_patterns()
        self._check_ast()
        return self.issues

    def _check_patterns(self) -> None:
        """Check for pattern-based security issues."""
        for i, line in enumerate(self.lines, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('""') or stripped.startswith("''"):
                continue

            # Check for secrets
            for secret_type, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    if not self._is_allowlisted(line, secret_type):
                        self.issues.append(
                            SecurityIssue(
                                self.file_path,
                                i,
                                "secret",
                                f"Potential hardcoded {secret_type} detected",
                                "high",
                            )
                        )

            # Check for dangerous patterns
            for danger_type, pattern in DANGEROUS_PATTERNS.items():
                if pattern.search(line):
                    if "# noqa" not in line and "# nosec" not in line:
                        self.issues.append(
                            SecurityIssue(
                                self.file_path,
                                i,
                                "dangerous",
                                f"Dangerous pattern: {danger_type}",
                                "high",
                            )
                        )

            # Check for SQL injection
            for sql_type, pattern in SQL_INJECTION_PATTERNS.items():
                if pattern.search(line):
                    self.issues.append(
                        SecurityIssue(
                            self.file_path,
                            i,
                            "sql_injection",
                            f"Potential SQL injection: {sql_type}",
                            "high",
                        )
                    )

    def _is_allowlisted(self, line: str, secret_type: str) -> bool:
        """Check if a line is in the allowlist."""
        allowlist = ALLOWLIST.get(secret_type, [])
        return any(allowed in line for allowed in allowlist)

    def _check_ast(self) -> None:
        """Check AST for security issues."""
        try:
            tree = ast.parse(self.source)
            self.visit(tree)
        except SyntaxError:
            pass  # Let other tools handle syntax errors

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in UNSAFE_IMPORTS:
                self.issues.append(
                    SecurityIssue(
                        self.file_path,
                        node.lineno,
                        "unsafe_import",
                        f"Potentially unsafe import: {module}",
                        "medium",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports."""
        if node.module:
            module = node.module.split(".")[0]
            if module in UNSAFE_IMPORTS:
                self.issues.append(
                    SecurityIssue(
                        self.file_path,
                        node.lineno,
                        "unsafe_import",
                        f"Potentially unsafe import: {module}",
                        "medium",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for eval/exec calls
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                self.issues.append(
                    SecurityIssue(
                        self.file_path,
                        node.lineno,
                        "dangerous_call",
                        f"Dangerous function call: {node.func.id}()",
                        "high",
                    )
                )
        self.generic_visit(node)


def scan_file(file_path: Path) -> list[SecurityIssue]:
    """Scan a single file for security issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return [SecurityIssue(file_path, 0, "error", f"Could not read file: {e}", "low")]

    scanner = SecurityScanner(file_path, source)
    return scanner.scan()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Security scanner for Python files")
    parser.add_argument(
        "files",
        nargs="+",
        help="Python files to scan",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Fail on medium severity issues (default: only high)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Patterns to exclude from scanning",
    )

    args = parser.parse_args(argv)

    all_issues: list[SecurityIssue] = []

    for file_path_str in args.files:
        file_path = Path(file_path_str)

        # Skip excluded patterns
        if any(exclude in str(file_path) for exclude in args.exclude):
            continue

        # Skip non-Python files
        if not file_path.suffix == ".py":
            continue

        issues = scan_file(file_path)
        all_issues.extend(issues)

    if not all_issues:
        print("✅ Security scan passed - no issues found")
        return 0

    # Group issues by severity
    high_issues = [i for i in all_issues if i.severity == "high"]
    medium_issues = [i for i in all_issues if i.severity == "medium"]
    low_issues = [i for i in all_issues if i.severity == "low"]

    print()
    print("=" * 70)
    print("🔒 Security Scan Results")
    print("=" * 70)

    for issue in all_issues:
        print(issue)

    print("=" * 70)
    print(
        f"Found: {len(high_issues)} high, {len(medium_issues)} medium, "
        f"{len(low_issues)} low severity issues"
    )

    if high_issues or (args.fail_on_warning and medium_issues):
        print()
        print("❌ Security scan failed!")
        print()
        print("To suppress false positives:")
        print("  - Add '# nosec' comment to the line")
        print("  - Add '# noqa' comment for specific linters")
        print()
        print("Review the findings carefully before committing.")
        return 1

    print()
    print("⚠️  Security warnings found (not blocking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
