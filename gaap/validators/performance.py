import ast
import re
from typing import Any

from gaap.validators.base import BaseValidator, QualityGate, ValidationResult


class PerformanceValidator(BaseValidator):
    """مدقق الأداء"""

    @property
    def gate(self) -> QualityGate:
        return QualityGate.PERFORMANCE_CHECK

    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من الأداء"""
        artifact_str = str(artifact) if artifact else ""

        if not artifact_str.strip():
            return self._record_result(ValidationResult.success("Empty artifact"))

        language = context.get("language", "python") if context else "python"

        if language == "python":
            return self._record_result(self._validate_python(artifact_str))
        elif language in ("javascript", "typescript"):
            return self._record_result(self._validate_js(artifact_str))

        return self._record_result(ValidationResult.success("Language not supported"))

    def _validate_python(self, code: str) -> ValidationResult:
        """تحليل أداء Python"""
        issues = []
        suggestions = []

        if "for " in code and " in " in code:
            nested_loops = self._count_nested_loops(code)
            if nested_loops >= 2:
                issues.append(f"Nested loops detected ({nested_loops} levels)")
                suggestions.append("Consider algorithmic optimization for nested loops")

        if "+=" in code and "for " in code:
            issues.append("String concatenation in loop detected")
            suggestions.append("Use list.append() + join() or f-string")

        if "while True:" in code and "break" not in code:
            issues.append("Potential infinite loop")
            suggestions.append("Ensure loop has exit condition")

        if ".append(" not in code and "[" in code and "for " in code and "+=" in code:
            issues.append("List comprehension could be more efficient")
            suggestions.append("Consider using list comprehension instead of loop")

        if "range(len(" in code:
            issues.append("Using range(len()) pattern")
            suggestions.append("Use enumerate() instead")

        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree))
        except SyntaxError:
            pass

        if issues:
            return ValidationResult.warning(
                f"Performance concerns: {len(issues)} issues",
                suggestions=list(set(suggestions)),
                details={"issues": issues},
            )

        return ValidationResult.success("No performance issues detected")

    def _validate_js(self, code: str) -> ValidationResult:
        """تحليل أداء JavaScript"""
        issues = []
        suggestions = []

        nested_loops = self._count_nested_loops(code)
        if nested_loops >= 2:
            issues.append(f"Nested loops detected ({nested_loops} levels)")
            suggestions.append("Consider algorithmic optimization")

        if re.search(r"\+\s*=\s*['\"].*['\"]\s*\+", code):
            issues.append("String concatenation in loop")
            suggestions.append("Use array.join() or template literals")

        if re.search(r"\.innerHTML\s*=", code):
            issues.append("Direct DOM manipulation can cause reflows")
            suggestions.append("Consider using DocumentFragment or batch updates")

        if "async" not in code and ("fetch(" in code or "await" in code):
            issues.append("Async operation without async/await")
            suggestions.append("Ensure proper async handling")

        if issues:
            return ValidationResult.warning(
                f"Performance concerns: {len(issues)} issues",
                suggestions=list(set(suggestions)),
                details={"issues": issues},
            )

        return ValidationResult.success("No performance issues detected")

    def _count_nested_loops(self, code: str) -> int:
        """عدد الـ nested loops"""
        max_depth = 0
        current_depth = 0

        for char in code:
            if char == "for" or char == "while":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "\n" and current_depth > 0:
                if "for" not in code[max(0, code.find(char) - 10) : code.find(char)]:
                    current_depth = max(0, current_depth - 1)

        return max_depth

    def _analyze_ast(self, tree: ast.AST) -> list[str]:
        """تحليل AST للكشف عن مشاكل الأداء"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 50:
                issues.append(f"Large function '{node.name}' ({len(node.body)} statements)")

        return issues
