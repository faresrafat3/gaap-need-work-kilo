import re
from typing import Any

from gaap.validators.base import BaseValidator, QualityGate, ValidationResult


class ComplianceValidator(BaseValidator):
    """مدقق الامتثال"""

    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN detected", "Remove or mask SSN"),
        (r"\b\d{9}\b", "Possible SSN (9 digits)", "Remove or mask"),
        (r"\b[A-Z]{2}\d{6,9}\b", "Passport number detected", "Remove or mask"),
        (r"\b\d{16}\b", "Credit card number detected", "Use payment processor instead"),
        (r"credit[_-]?card", "Credit card reference", "Avoid storing card numbers"),
        (
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "Credit card format detected",
            "Use payment processor",
        ),
    ]

    REGULATORY_PATTERNS = [
        (r"gdpr", "GDPR reference detected", "Ensure GDPR compliance"),
        (r"hipaa", "HIPAA reference detected", "Ensure HIPAA compliance"),
        (r"pci[_-]?d?ss?", "PCI-DSS reference detected", "Ensure PCI compliance"),
        (r"data[_-]?retention", "Data retention policy reference", "Define retention policy"),
        (
            r"right[_-]?to[_-]?be[_-]?forgotten",
            "Right to be forgotten reference",
            "Implement deletion capability",
        ),
        (r"consent", "Consent reference detected", "Implement consent management"),
    ]

    SENSITIVE_DATA_PATTERNS = [
        (
            r"password\s*=\s*['\"][^'\"]+['\"]",
            "Plain text password in code",
            "Use secure password handling",
        ),
        (
            r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
            "API key in code",
            "Use environment variables",
        ),
        (
            r"bearer\s+[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
            "Bearer token in code",
            "Use secure token storage",
        ),
        (r"secret[_-]?key\s*=", "Secret key in code", "Use secrets manager"),
        (r"private[_-]?key", "Private key in code", "Use secure key management"),
    ]

    @property
    def gate(self) -> QualityGate:
        return QualityGate.COMPLIANCE_CHECK

    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من الامتثال"""
        artifact_str = str(artifact) if artifact else ""

        if not artifact_str.strip():
            return self._record_result(ValidationResult.success("Empty artifact"))

        issues = []
        suggestions = []
        warnings = []

        for pattern, issue, suggestion in self.PII_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                issues.append(issue)
                suggestions.append(suggestion)

        for pattern, issue, suggestion in self.REGULATORY_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                warnings.append(f"Regulatory reference: {issue}")
                suggestions.append(suggestion)

        for pattern, issue, suggestion in self.SENSITIVE_DATA_PATTERNS:
            if re.search(pattern, artifact_str, re.IGNORECASE):
                issues.append(issue)
                suggestions.append(suggestion)

        if issues:
            unique_issues = list(set(issues))
            unique_suggestions = list(set(suggestions))
            return self._record_result(
                ValidationResult.failed(
                    f"Compliance issues found: {len(unique_issues)}",
                    suggestions=unique_suggestions,
                    details={"issues": unique_issues, "warnings": warnings},
                )
            )

        if warnings:
            return ValidationResult.warning(
                "Regulatory considerations detected",
                suggestions=list(set(suggestions)) if suggestions else None,
                details={"warnings": warnings},
            )

        return self._record_result(ValidationResult.success("No compliance issues detected"))
