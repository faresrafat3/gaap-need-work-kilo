from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Severity(Enum):
    """مستوى الخطورة"""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


class QualityGate(Enum):
    """بوابات الجودة"""

    SYNTAX_CHECK = auto()
    SECURITY_SCAN = auto()
    PERFORMANCE_CHECK = auto()
    STYLE_CHECK = auto()
    COMPLIANCE_CHECK = auto()


@dataclass
class ValidationResult:
    """نتيجة التحقق"""

    passed: bool
    severity: Severity = Severity.INFO
    message: str = ""
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passed(cls, message: str = "Validation passed", **kwargs) -> "ValidationResult":
        return cls(passed=True, severity=Severity.INFO, message=message, **kwargs)

    @classmethod
    def warning(
        cls, message: str, suggestions: list[str] | None = None, **kwargs
    ) -> "ValidationResult":
        return cls(
            passed=True,
            severity=Severity.WARNING,
            message=message,
            suggestions=suggestions or [],
            **kwargs,
        )

    @classmethod
    def failed(
        cls, message: str, suggestions: list[str] | None = None, **kwargs
    ) -> "ValidationResult":
        return cls(
            passed=False,
            severity=Severity.ERROR,
            message=message,
            suggestions=suggestions or [],
            **kwargs,
        )


class BaseValidator(ABC):
    """الفئة الأساسية للمدققين"""

    def __init__(self, enabled: bool = True, fail_on_error: bool = True):
        self.enabled = enabled
        self.fail_on_error = fail_on_error
        self._validation_count = 0
        self._failures = 0

    @property
    @abstractmethod
    def gate(self) -> QualityGate:
        """نوع البوابة"""
        pass

    @abstractmethod
    async def validate(
        self, artifact: Any, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """التحقق من الـ artifact"""
        pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "validator": self.__class__.__name__,
            "gate": self.gate.name,
            "enabled": self.enabled,
            "validations": self._validation_count,
            "failures": self._failures,
        }

    def _record_result(self, result: ValidationResult) -> ValidationResult:
        self._validation_count += 1
        if not result.passed:
            self._failures += 1
        return result
