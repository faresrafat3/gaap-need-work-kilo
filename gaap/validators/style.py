import re
from typing import Any

from gaap.validators.base import BaseValidator, QualityGate, ValidationResult


class StyleValidator(BaseValidator):
    """مدقق الأسلوب"""

    @property
    def gate(self) -> QualityGate:
        return QualityGate.STYLE_CHECK

    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من الأسلوب"""
        artifact_str = str(artifact) if artifact else ""

        if not artifact_str.strip():
            return self._record_result(ValidationResult.passed("Empty artifact"))

        language = context.get("language", "python") if context else "python"

        if language == "python":
            return self._record_result(self._validate_python(artifact_str))
        elif language in ("javascript", "typescript"):
            return self._record_result(self._validate_js(artifact_str))

        return self._record_result(ValidationResult.passed("Language not supported"))

    def _validate_python(self, code: str) -> ValidationResult:
        """تحليل أسلوب Python"""
        issues = []
        suggestions = []

        lines = code.split("\n")

        if len(lines) > 200:
            issues.append(f"File too long: {len(lines)} lines")
            suggestions.append("Consider splitting into multiple modules")

        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i} exceeds 120 characters ({len(line)} chars)")
                suggestions.append("Keep lines under 120 characters")

            if line.strip().startswith("\t"):
                issues.append(f"Line {i}: Uses tabs instead of spaces")
                suggestions.append("Use 4 spaces for indentation")

            if re.match(r"^\s+$", line) and i > 1 and i < len(lines):
                if not re.match(r"^\s+$", lines[i - 2]):
                    issues.append(f"Line {i}: Multiple consecutive blank lines")
                    suggestions.append("Use single blank line")

        if "import *" in code:
            issues.append("Wildcard imports detected (import *)")
            suggestions.append("Use explicit imports")

        if re.search(r"[a-z][A-Z]", code):
            camel_case_vars = re.findall(r"[a-z][A-Z][a-z]", code)
            if camel_case_vars:
                issues.append(f"CamelCase naming detected: {camel_case_vars[0]}")
                suggestions.append("Use snake_case for variables and functions")

        if code.count("    ") == 0 and len(lines) > 10:
            issues.append("No indentation detected")

        if issues:
            return ValidationResult.warning(
                f"Style issues found: {len(issues)}",
                suggestions=list(set(suggestions)),
                details={"issues": issues[:10]},
            )

        return ValidationResult.passed("Code style looks good")

    def _validate_js(self, code: str) -> ValidationResult:
        """تحليل أسلوب JavaScript"""
        issues = []
        suggestions = []

        lines = code.split("\n")

        if len(lines) > 300:
            issues.append(f"File too long: {len(lines)} lines")
            suggestions.append("Consider splitting into multiple files")

        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i} exceeds 120 characters")

            if line.strip().endswith(" ") and line.strip():
                issues.append(f"Line {i}: Trailing whitespace")

        if "var " in code:
            issues.append("Use of 'var' detected")
            suggestions.append("Use 'const' or 'let' instead")

        if re.search(r"==[^=]", code):
            issues.append("Use of == instead of ===")
            suggestions.append("Use === for strict equality")

        if "function(" in code and "=>" not in code:
            issues.append("Traditional function syntax detected")
            suggestions.append("Consider using arrow functions for callbacks")

        if issues:
            return ValidationResult.warning(
                f"Style issues found: {len(issues)}",
                suggestions=list(set(suggestions)),
                details={"issues": issues[:10]},
            )

        return ValidationResult.passed("Code style looks good")
