from gaap.validators.base import (
    BaseValidator,
    QualityGate,
    Severity,
    ValidationResult,
)
from gaap.validators.compliance import ComplianceValidator
from gaap.validators.performance import PerformanceValidator
from gaap.validators.security import SecurityValidator
from gaap.validators.style import StyleValidator
from gaap.validators.syntax import SyntaxValidator

__all__ = [
    "BaseValidator",
    "QualityGate",
    "Severity",
    "ValidationResult",
    "SyntaxValidator",
    "SecurityValidator",
    "PerformanceValidator",
    "StyleValidator",
    "ComplianceValidator",
]


def create_validators(
    enable_syntax: bool = True,
    enable_security: bool = True,
    enable_performance: bool = True,
    enable_style: bool = True,
    enable_compliance: bool = True,
    fail_on_error: bool = True,
) -> dict[QualityGate, BaseValidator]:
    """إنشاء جميع الـ validators"""
    validators = {}

    if enable_syntax:
        validators[QualityGate.SYNTAX_CHECK] = SyntaxValidator(fail_on_error=fail_on_error)

    if enable_security:
        validators[QualityGate.SECURITY_SCAN] = SecurityValidator(fail_on_error=fail_on_error)

    if enable_performance:
        validators[QualityGate.PERFORMANCE_CHECK] = PerformanceValidator(
            fail_on_error=fail_on_error
        )

    if enable_style:
        validators[QualityGate.STYLE_CHECK] = StyleValidator(fail_on_error=fail_on_error)

    if enable_compliance:
        validators[QualityGate.COMPLIANCE_CHECK] = ComplianceValidator(fail_on_error=fail_on_error)

    return validators
