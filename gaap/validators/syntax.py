import ast
import re
from typing import Any

from gaap.validators.base import BaseValidator, QualityGate, ValidationResult


class SyntaxValidator(BaseValidator):
    """مدقق الصيغة اللغوية"""

    @property
    def gate(self) -> QualityGate:
        return QualityGate.SYNTAX_CHECK

    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من صحة الصيغة"""
        artifact_str = str(artifact) if artifact else ""

        if not artifact_str.strip():
            return self._record_result(
                ValidationResult.failed(
                    "Artifact is empty", suggestions=["Provide non-empty content"]
                )
            )

        language = context.get("language", "python") if context else "python"

        if language == "python":
            return self._record_result(self._validate_python(artifact_str))
        elif language in ("javascript", "typescript", "js", "ts"):
            return self._record_result(self._validate_js(artifact_str))
        elif language in ("html", "css", "json", "sql"):
            return self._record_result(self._validate_generic(artifact_str, language))

        return self._record_result(
            ValidationResult.passed("Language not supported for syntax check")
        )

    def _validate_python(self, code: str) -> ValidationResult:
        """التحقق من صيغة Python"""
        errors = []
        suggestions = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            suggestions.append(f"Fix the syntax error: {e.msg}")
            return ValidationResult.failed(
                "Python syntax validation failed",
                suggestions=suggestions,
                details={"errors": errors},
            )

        except Exception as e:
            return ValidationResult.failed(
                f"Unexpected parsing error: {str(e)}",
                suggestions=["Review code for hidden characters or encoding issues"],
            )

        return ValidationResult.passed("Python syntax is valid")

    def _validate_js(self, code: str) -> ValidationResult:
        """التحقق من صيغة JavaScript/TypeScript"""
        errors = []
        suggestions = []

        patterns = [
            (r"\{[^}]*$", "Unclosed brace"),
            (r"\[[^\]]*$", "Unclosed bracket"),
            (r"\([^)]*$", "Unclosed parenthesis"),
            (r"function\s+\w+\s*\([^)]*$", "Unclosed function declaration"),
            (r"const\s+\w+\s*=\s*[^;]*$", "Incomplete const assignment"),
            (r"let\s+\w+\s*=\s*[^;]*$", "Incomplete let assignment"),
            (r"var\s+\w+\s*=\s*[^;]*$", "Incomplete var assignment"),
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern, error_msg in patterns:
                if re.search(pattern, line):
                    errors.append(f"Line {i}: Potential {error_msg}")

        for char in "{}()[]":
            open_count = code.count(char)
            close_count = 0
            if char in "{([":
                continue
            elif char == "}":
                close_count = code.count("}")
                open_count = code.count("{")
            elif char == ")":
                close_count = code.count(")")
                open_count = code.count("(")
            elif char == "]":
                close_count = code.count("]")
                open_count = code.count("[")

            if open_count != close_count:
                errors.append(
                    f"Unbalanced brackets: {char} (open: {open_count}, close: {close_count})"
                )

        if errors:
            suggestions.append("Check bracket matching and incomplete statements")
            return ValidationResult.failed(
                "JavaScript syntax validation failed",
                suggestions=suggestions,
                details={"errors": errors},
            )

        return ValidationResult.passed("JavaScript syntax appears valid")

    def _validate_generic(self, code: str, language: str) -> ValidationResult:
        """التحقق العام للصيغ"""
        errors = []

        if language == "json":
            try:
                import json

                json.loads(code)
            except json.JSONDecodeError as e:
                errors.append(f"JSON error: {e}")
                return ValidationResult.failed(
                    "JSON validation failed",
                    suggestions=["Fix JSON syntax errors"],
                    details={"errors": errors},
                )

        elif language == "html":
            unclosed_tags = re.findall(r"<(\w+)[^>]*>(?!.*</\1>)", code)
            if unclosed_tags:
                errors.append(f"Unclosed tags: {unclosed_tags[:5]}")

        elif language == "sql":
            keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "FROM", "WHERE"]
            if not any(kw in code.upper() for kw in keywords):
                errors.append("No SQL keywords detected")

        if errors:
            return ValidationResult.warning(
                f"{language.upper()} has some issues",
                suggestions=["Review code for syntax issues"],
                details={"errors": errors},
            )

        return ValidationResult.passed(f"{language.upper()} syntax is valid")
