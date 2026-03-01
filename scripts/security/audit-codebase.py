#!/usr/bin/env python3
"""
GAAP Custom Security Auditor
============================

A comprehensive security audit tool that scans the codebase for:
- Hardcoded secrets and credentials
- Unsafe eval/exec usage
- API keys in code
- TODO/FIXME security markers
- Insecure patterns
- Suspicious imports

Usage:
    python scripts/security/audit-codebase.py [--output FILE] [--format {json,text}]

Exit codes:
    0 - No issues found
    1 - Issues found (count in output)
    2 - Error during scan
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SecurityFinding:
    """Represents a security finding."""

    rule: str
    severity: str  # critical, high, medium, low
    file: str
    line: int
    column: int
    message: str
    code_snippet: str = ""
    remediation: str = ""


@dataclass
class AuditReport:
    """Complete audit report."""

    timestamp: str
    duration_ms: float
    files_scanned: int
    critical: list[SecurityFinding] = field(default_factory=list)
    high: list[SecurityFinding] = field(default_factory=list)
    medium: list[SecurityFinding] = field(default_factory=list)
    low: list[SecurityFinding] = field(default_factory=list)
    info: list[SecurityFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "files_scanned": self.files_scanned,
            "summary": {
                "critical": len(self.critical),
                "high": len(self.high),
                "medium": len(self.medium),
                "low": len(self.low),
                "info": len(self.info),
                "total": len(self.critical)
                + len(self.high)
                + len(self.medium)
                + len(self.low)
                + len(self.info),
            },
            "findings": {
                "critical": [self._finding_to_dict(f) for f in self.critical],
                "high": [self._finding_to_dict(f) for f in self.high],
                "medium": [self._finding_to_dict(f) for f in self.medium],
                "low": [self._finding_to_dict(f) for f in self.low],
                "info": [self._finding_to_dict(f) for f in self.info],
            },
        }

    @staticmethod
    def _finding_to_dict(finding: SecurityFinding) -> dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            "rule": finding.rule,
            "severity": finding.severity,
            "file": finding.file,
            "line": finding.line,
            "column": finding.column,
            "message": finding.message,
            "code_snippet": finding.code_snippet,
            "remediation": finding.remediation,
        }


class SecurityAuditor:
    """Main security auditor class."""

    # Secret patterns to detect
    SECRET_PATTERNS = {
        "aws_access_key": (
            re.compile(r"AKIA[0-9A-Z]{16}"),
            "AWS Access Key ID detected",
            "Remove hardcoded AWS credentials, use environment variables or AWS IAM roles",
        ),
        "aws_secret_key": (
            re.compile(r'["\'][0-9a-zA-Z/+]{40}["\']'),
            "Possible AWS Secret Key detected",
            "Remove hardcoded AWS credentials, use environment variables or AWS IAM roles",
        ),
        "private_key": (
            re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
            "Private key detected",
            "Remove hardcoded private keys, use secure key management (e.g., HashiCorp Vault)",
        ),
        "api_key_generic": (
            re.compile(r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9_\-]{16,}["\']'),
            "Generic API key detected",
            "Remove hardcoded API keys, use environment variables or secrets management",
        ),
        "secret_generic": (
            re.compile(r'(?i)(secret[_-]?key|secretkey)\s*[:=]\s*["\'][a-zA-Z0-9_\-]{16,}["\']'),
            "Generic secret key detected",
            "Remove hardcoded secrets, use environment variables or secrets management",
        ),
        "password_in_code": (
            re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{4,}["\']'),
            "Possible hardcoded password detected",
            "Remove hardcoded passwords, use environment variables or secrets management",
        ),
        "token_generic": (
            re.compile(r'(?i)(token|bearer)\s*[:=]\s*["\'][a-zA-Z0-9_\-\.]{20,}["\']'),
            "Generic token detected",
            "Remove hardcoded tokens, use environment variables or secrets management",
        ),
        "github_token": (
            re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
            "GitHub token detected",
            "Remove hardcoded GitHub tokens, use environment variables",
        ),
        "slack_token": (
            re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"),
            "Slack token detected",
            "Remove hardcoded Slack tokens, use environment variables",
        ),
        "jwt_token": (
            re.compile(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),
            "Possible JWT token detected",
            "Remove hardcoded JWT tokens, use secure token generation and storage",
        ),
        "database_url": (
            re.compile(r'(?i)(postgres|mysql|mongodb|redis)://[^\s"\']+:[^\s"\']+@[^\s"\']+'),
            "Database connection string with credentials detected",
            "Remove hardcoded database URLs, use environment variables or connection pooling with secrets",
        ),
    }

    # Unsafe patterns
    UNSAFE_PATTERNS = {
        "eval_usage": (
            re.compile(r"\beval\s*\("),
            "Use of eval() detected",
            "Avoid eval(), use ast.literal_eval for safe evaluation or json.loads for JSON",
        ),
        "exec_usage": (
            re.compile(r"\bexec\s*\("),
            "Use of exec() detected",
            "Avoid exec(), it allows arbitrary code execution. Refactor to use safer alternatives",
        ),
        "compile_usage": (
            re.compile(r"\bcompile\s*\("),
            "Use of compile() detected",
            "Review compile() usage carefully, ensure input is sanitized",
        ),
        "subprocess_shell": (
            re.compile(r"subprocess\.\w+.*shell\s*=\s*True"),
            "subprocess with shell=True detected",
            "Avoid shell=True, use shell=False and pass command as list. If necessary, sanitize all inputs",
        ),
        "pickle_load": (
            re.compile(r"pickle\.load|pickle\.loads"),
            "Unsafe pickle usage detected",
            "Avoid pickle for untrusted data, use json or msgpack. If necessary, implement signing",
        ),
        "yaml_load": (
            re.compile(r"yaml\.load\s*\([^)]*\)(?!.*Loader)"),
            "Unsafe yaml.load() without Loader specified",
            "Use yaml.safe_load() instead of yaml.load() or specify SafeLoader",
        ),
        "input_usage": (
            re.compile(r"\binput\s*\("),
            "Use of input() detected",
            "In Python 2, input() evaluates. In Python 3, consider validation. Review for security",
        ),
        "format_sql": (
            re.compile(r"\.format\s*\([^)]*\).*(?:SELECT|INSERT|UPDATE|DELETE|WHERE)"),
            "Possible SQL injection via string formatting",
            "Use parameterized queries with SQL libraries, never format SQL strings",
        ),
        "fstring_sql": (
            re.compile(r'f["\'][^"\']*(?:SELECT|INSERT|UPDATE|DELETE)[^"\']*\{[^}]+\}'),
            "Possible SQL injection via f-string",
            "Use parameterized queries, never use f-strings for SQL",
        ),
        "hardcoded_ip": (
            re.compile(
                r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
            ),
            "Hardcoded IP address detected",
            "Consider using DNS names or configuration files for IP addresses",
        ),
    }

    # Security markers in comments
    SECURITY_MARKERS = {
        "todo_security": (
            re.compile(r"(?i)#.*TODO.*SECUR"),
            "TODO with security mentioned",
            "Review and address security TODO before production",
        ),
        "fixme_security": (
            re.compile(r"(?i)#.*FIXME.*SECUR"),
            "FIXME with security mentioned",
            "Review and address security FIXME before production",
        ),
        "hack_security": (
            re.compile(r"(?i)#.*HACK.*SECUR"),
            "HACK with security mentioned",
            "Review security hack, implement proper solution",
        ),
        "temporary_security": (
            re.compile(r"(?i)#.*TEMP.*SECUR"),
            "Temporary security measure detected",
            "Remove temporary security measures before production",
        ),
        "insecure_disable": (
            re.compile(r"(?i)#.*(?:disable|skip|bypass).*verif|certif|ssl|tls"),
            "Potential security verification bypass",
            "Review SSL/TLS/certificate verification bypass",
        ),
    }

    # Suspicious imports
    SUSPICIOUS_IMPORTS = {
        "pickle": ("pickle", "high", "Unsafe deserialization library"),
        "subprocess": ("subprocess", "medium", "Process spawning - ensure input validation"),
        "os_system": ("os.system", "high", "Shell command execution - extremely dangerous"),
        "tempfile_mktemp": ("tempfile.mktemp", "medium", "Insecure temporary file creation"),
        "ftplib": ("ftplib", "low", "Unencrypted FTP - consider FTPS or SFTP"),
        "telnetlib": ("telnetlib", "high", "Unencrypted telnet - never use in production"),
        "ssl_wrap": ("ssl.wrap_socket", "medium", "Deprecated SSL function"),
    }

    def __init__(self, root_path: str = ".") -> None:
        self.root_path = Path(root_path)
        self.findings: list[SecurityFinding] = []
        self.files_scanned = 0
        self.exclude_patterns = [
            r"\.venv",
            r"venv",
            r"\.git",
            r"__pycache__",
            r"\.pytest_cache",
            r"\.mypy_cache",
            r"build",
            r"dist",
            r"\.eggs",
            r"node_modules",
            r"\.tox",
            r"\.coverage",
            r"htmlcov",
            r"\.next",
            r"scripts/security",  # Exclude self
        ]

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if re.search(pattern, path_str):
                return True
        return False

    def scan_file(self, file_path: Path) -> list[SecurityFinding]:
        """Scan a single file for security issues."""
        findings = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            return [
                SecurityFinding(
                    rule="file_read_error",
                    severity="info",
                    file=str(file_path),
                    line=0,
                    column=0,
                    message=f"Could not read file: {e}",
                )
            ]

        # Scan for secret patterns
        for pattern_name, (pattern, message, remediation) in self.SECRET_PATTERNS.items():
            for match in pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                col_num = match.start() - content.rfind("\n", 0, match.start())
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                # Skip if in comment or string that's clearly not a secret
                if self._is_likely_false_positive(line_content, pattern_name):
                    continue

                findings.append(
                    SecurityFinding(
                        rule=f"secret:{pattern_name}",
                        severity="critical"
                        if pattern_name in ["private_key", "aws_secret_key"]
                        else "high",
                        file=str(file_path),
                        line=line_num,
                        column=col_num,
                        message=message,
                        code_snippet=line_content.strip()[:100],
                        remediation=remediation,
                    )
                )

        # Scan for unsafe patterns
        for pattern_name, (pattern, message, remediation) in self.UNSAFE_PATTERNS.items():
            for match in pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                col_num = match.start() - content.rfind("\n", 0, match.start())
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                severity = "critical" if pattern_name in ["eval_usage", "exec_usage"] else "high"
                if pattern_name in ["hardcoded_ip", "input_usage"]:
                    severity = "low"

                findings.append(
                    SecurityFinding(
                        rule=f"unsafe:{pattern_name}",
                        severity=severity,
                        file=str(file_path),
                        line=line_num,
                        column=col_num,
                        message=message,
                        code_snippet=line_content.strip()[:100],
                        remediation=remediation,
                    )
                )

        # Scan for security markers
        for pattern_name, (pattern, message, remediation) in self.SECURITY_MARKERS.items():
            for match in pattern.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                col_num = match.start() - content.rfind("\n", 0, match.start())
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                findings.append(
                    SecurityFinding(
                        rule=f"marker:{pattern_name}",
                        severity="medium",
                        file=str(file_path),
                        line=line_num,
                        column=col_num,
                        message=message,
                        code_snippet=line_content.strip()[:100],
                        remediation=remediation,
                    )
                )

        # AST-based analysis for imports and function calls
        try:
            tree = ast.parse(content)
            findings.extend(self._analyze_ast(tree, file_path, lines))
        except SyntaxError:
            pass  # Skip files with syntax errors

        return findings

    def _is_likely_false_positive(self, line: str, pattern_name: str) -> bool:
        """Check if a match is likely a false positive."""
        # Skip if clearly in a comment explaining the pattern
        if line.strip().startswith("#") and "example" in line.lower():
            return True
        if line.strip().startswith("#") and "placeholder" in line.lower():
            return True
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            return True

        # Skip test files with mock data
        if "test" in line.lower() and "mock" in line.lower():
            return True
        if "example" in line.lower() or "dummy" in line.lower():
            return True
        if "placeholder" in line.lower() or "changeme" in line.lower():
            return True

        # Skip environment variable references
        if "os.environ" in line or "os.getenv" in line:
            return True

        # Skip configuration templates
        if pattern_name == "database_url" and ("${" in line or "{{" in line):
            return True

        return False

    def _analyze_ast(
        self, tree: ast.AST, file_path: Path, lines: list[str]
    ) -> list[SecurityFinding]:
        """Analyze AST for security issues."""
        findings = []

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    finding = self._check_import(alias.name, file_path, node, lines)
                    if finding:
                        findings.append(finding)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    finding = self._check_import(full_name, file_path, node, lines)
                    if finding:
                        findings.append(finding)

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                finding = self._check_dangerous_call(node, file_path, lines)
                if finding:
                    findings.append(finding)

        return findings

    def _check_import(
        self, name: str, file_path: Path, node: ast.AST, lines: list[str]
    ) -> SecurityFinding | None:
        """Check if an import is suspicious."""
        for check_name, (import_name, severity, message) in self.SUSPICIOUS_IMPORTS.items():
            if import_name in name:
                return SecurityFinding(
                    rule=f"import:{check_name}",
                    severity=severity,
                    file=str(file_path),
                    line=node.lineno,
                    column=node.col_offset,
                    message=message,
                    code_snippet=lines[node.lineno - 1].strip()[:100]
                    if node.lineno <= len(lines)
                    else "",
                    remediation=f"Review usage of {import_name} and ensure it's necessary and secure",
                )
        return None

    def _check_dangerous_call(
        self, node: ast.Call, file_path: Path, lines: list[str]
    ) -> SecurityFinding | None:
        """Check for dangerous function calls."""
        # Check for hashlib.md5
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "md5" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "hashlib":
                    return SecurityFinding(
                        rule="crypto:weak_hash",
                        severity="medium",
                        file=str(file_path),
                        line=node.lineno,
                        column=node.col_offset,
                        message="Use of MD5 hash detected",
                        code_snippet=lines[node.lineno - 1].strip()[:100]
                        if node.lineno <= len(lines)
                        else "",
                        remediation="Use SHA-256 or stronger for cryptographic purposes",
                    )

            # Check for random (not cryptographically secure)
            if node.func.attr in ["random", "randint", "choice", "shuffle"]:
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "random":
                    return SecurityFinding(
                        rule="crypto:insecure_random",
                        severity="medium",
                        file=str(file_path),
                        line=node.lineno,
                        column=node.col_offset,
                        message="Use of random module for potentially cryptographic purpose",
                        code_snippet=lines[node.lineno - 1].strip()[:100]
                        if node.lineno <= len(lines)
                        else "",
                        remediation="Use secrets module for cryptographic randomness",
                    )

        return None

    def scan_directory(self, directory: Path | None = None) -> AuditReport:
        """Scan entire directory for security issues."""
        import time

        start_time = time.time()

        if directory is None:
            directory = self.root_path

        all_findings = []
        files_scanned = 0

        # Find all Python files
        for py_file in directory.rglob("*.py"):
            if self.should_exclude(py_file):
                continue

            findings = self.scan_file(py_file)
            all_findings.extend(findings)
            files_scanned += 1

        # Categorize findings
        report = AuditReport(
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=(time.time() - start_time) * 1000,
            files_scanned=files_scanned,
        )

        for finding in all_findings:
            if finding.severity == "critical":
                report.critical.append(finding)
            elif finding.severity == "high":
                report.high.append(finding)
            elif finding.severity == "medium":
                report.medium.append(finding)
            elif finding.severity == "low":
                report.low.append(finding)
            else:
                report.info.append(finding)

        return report


def format_text_report(report: AuditReport) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 80,
        "GAAP SECURITY AUDIT REPORT",
        "=" * 80,
        "",
        f"Timestamp: {report.timestamp}",
        f"Duration: {report.duration_ms:.2f}ms",
        f"Files Scanned: {report.files_scanned}",
        "",
        "-" * 80,
        "SUMMARY",
        "-" * 80,
        f"  Critical: {len(report.critical)}",
        f"  High:     {len(report.high)}",
        f"  Medium:   {len(report.medium)}",
        f"  Low:      {len(report.low)}",
        f"  Info:     {len(report.info)}",
        f"  Total:    {len(report.critical) + len(report.high) + len(report.medium) + len(report.low) + len(report.info)}",
        "",
    ]

    severity_order = [
        ("CRITICAL", report.critical),
        ("HIGH", report.high),
        ("MEDIUM", report.medium),
        ("LOW", report.low),
    ]

    for severity, findings in severity_order:
        if findings:
            lines.extend(
                [
                    "-" * 80,
                    f"{severity} FINDINGS ({len(findings)})",
                    "-" * 80,
                    "",
                ]
            )

            for i, finding in enumerate(findings, 1):
                lines.extend(
                    [
                        f"  [{i}] {finding.rule}",
                        f"      File: {finding.file}:{finding.line}:{finding.column}",
                        f"      Message: {finding.message}",
                    ]
                )
                if finding.code_snippet:
                    lines.append(f"      Code: {finding.code_snippet}")
                if finding.remediation:
                    lines.append(f"      Fix: {finding.remediation}")
                lines.append("")

    lines.extend(
        [
            "-" * 80,
            "END OF REPORT",
            "-" * 80,
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GAAP Custom Security Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/security/audit-codebase.py
  python scripts/security/audit-codebase.py --output report.json --format json
  python scripts/security/audit-codebase.py --output report.txt --format text
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=".",
        help="Root path to scan (default: current directory)",
    )
    parser.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium", "low"],
        default="critical",
        help="Minimum severity to fail on (default: critical)",
    )

    args = parser.parse_args()

    try:
        auditor = SecurityAuditor(root_path=args.path)
        report = auditor.scan_directory()

        # Generate output
        if args.format == "json":
            output = json.dumps(report.to_dict(), indent=2)
        else:
            output = format_text_report(report)

        # Write or print output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Report written to: {args.output}")
        else:
            print(output)

        # Determine exit code
        severity_levels = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }
        fail_level = severity_levels[args.fail_on]

        total_issues = 0
        if fail_level <= 4:
            total_issues += len(report.critical)
        if fail_level <= 3:
            total_issues += len(report.high)
        if fail_level <= 2:
            total_issues += len(report.medium)
        if fail_level <= 1:
            total_issues += len(report.low)

        if total_issues > 0:
            print(
                f"\n❌ Security audit failed with {total_issues} issue(s) at '{args.fail_on}' level or higher"
            )
            return 1

        print("\n✅ Security audit passed")
        return 0

    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error during audit: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
