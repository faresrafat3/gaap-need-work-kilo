import re
from typing import Any

from gaap.validators.base import BaseValidator, QualityGate, ValidationResult


class SecurityValidator(BaseValidator):
    """مدقق الأمان"""

    DANGEROUS_PATTERNS = [
        (
            r"eval\s*\(",
            "Use of eval() - code injection risk",
            "Avoid eval(), use safer alternatives",
        ),
        (
            r"exec\s*\(",
            "Use of exec() - code injection risk",
            "Avoid exec(), use safer alternatives",
        ),
        (
            r"__import__\s*\(\s*['\"]os['\"]",
            "Dynamic OS import - potential attack vector",
            "Use explicit imports",
        ),
        (
            r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
            "subprocess with shell=True - shell injection",
            "Use shell=False with list args",
        ),
        (
            r"subprocess\.run\s*\([^)]*shell\s*=\s*True",
            "subprocess with shell=True - shell injection",
            "Use shell=False with list args",
        ),
        (
            r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True",
            "subprocess with shell=True - shell injection",
            "Use shell=False with list args",
        ),
        (r"os\.system\s*\(", "os.system() - shell injection risk", "Use subprocess module instead"),
        (
            r"pickle\.loads\s*\(",
            "pickle.loads() - deserialization vulnerability",
            "Use JSON for data exchange",
        ),
        (
            r"yaml\.load\s*\([^)]*(?!Loader\s*=\s*yaml\.SafeLoader)",
            "yaml.load() without SafeLoader",
            "Use yaml.safe_load()",
        ),
        (
            r"xml\.parse\s*\([^)]*untrusted",
            "XML parsing from untrusted source",
            "Use defusedxml library",
        ),
    ]

    SECRET_PATTERNS = [
        (
            r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
            "Hardcoded API key",
            "Use environment variables",
        ),
        (
            r"password\s*=\s*['\"][^'\"]+['\"]",
            "Hardcoded password",
            "Use environment variables or secrets manager",
        ),
        (
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            "Hardcoded secret",
            "Use environment variables or secrets manager",
        ),
        (
            r"token\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
            "Hardcoded token",
            "Use environment variables",
        ),
        (r"private[_-]?key\s*=\s*['\"]", "Hardcoded private key", "Use secure key management"),
        (r"aws[_-]?access[_-]?key", "AWS access key detected", "Use IAM roles instead"),
        (
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            "Private key detected",
            "Use secure key management",
        ),
    ]

    VULNERABLE_PATTERNS = [
        (
            r"SELECT\s+.*\+.*FROM",
            "SQL string concatenation - SQL injection",
            "Use parameterized queries",
        ),
        (
            r"INSERT\s+INTO\s+.*\+",
            "SQL string concatenation - SQL injection",
            "Use parameterized queries",
        ),
        (
            r"execute\s*\(\s*['\"].*%",
            "SQL string with % formatting - SQL injection",
            "Use parameterized queries",
        ),
        (
            r"f['\"].*SELECT.*\{",
            "f-string SQL query - SQL injection risk",
            "Use parameterized queries",
        ),
        (
            r"format\s*\(\s*['\"].*SELECT",
            "format() SQL query - SQL injection risk",
            "Use parameterized queries",
        ),
        (r"<script[^>]*>", "Potential XSS - script tag in HTML", "Sanitize HTML output"),
        (
            r"innerHTML\s*=",
            "Direct innerHTML assignment - XSS risk",
            "Use textContent or sanitize input",
        ),
        (r"document\.write\s*\(", "document.write() - XSS risk", "Use DOM manipulation instead"),
    ]

    @property
    def gate(self) -> QualityGate:
        return QualityGate.SECURITY_SCAN

    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من الأمان"""
        artifact_str = str(artifact) if artifact else ""

        if not artifact_str.strip():
            return self._record_result(
                ValidationResult.passed("Empty artifact - no security check needed")
            )

        issues = []
        suggestions = []

        for pattern, issue, suggestion in self.DANGEROUS_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                issues.append(issue)
                suggestions.append(suggestion)

        for pattern, issue, suggestion in self.SECRET_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                issues.append(issue)
                suggestions.append(suggestion)

        for pattern, issue, suggestion in self.VULNERABLE_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                issues.append(issue)
                suggestions.append(suggestion)

        if issues:
            unique_issues = list(set(issues))
            unique_suggestions = list(set(suggestions))
            return self._record_result(
                ValidationResult.failed(
                    f"Security issues found: {len(unique_issues)}",
                    suggestions=unique_suggestions,
                    details={"issues": unique_issues},
                )
            )

        return self._record_result(ValidationResult.passed("No security issues detected"))
